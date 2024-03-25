import logging
import json, os, copy
import pandas as pd
import random

from transformers import XLMTokenizer
from torch import LongTensor
from torch.nn import CrossEntropyLoss

from src.consts import NER_TYPES, CLS_TOKEN, SEP_TOKEN, X_TOKEN
from src.data_processors.utils import convert2ids

logger = logging.getLogger(__name__)

pad_token_label_id = 0  # CrossEntropyLoss().ignore_index


class NERExample(object):
    """
    A single training/test example feature for NER task formulated as a token classification task.
    Args:
      :param unique_id: str
            Unique id for the example.
      :param text: string
            The untokenized text of the joined words (of the sentence).
      :param words: List<string>
            List of words
      :param language: (Optional) string
            The language of all words (assuming all are from the same language).
      :param labels: List<int>
            The NER class labels of each word.
      :param tokens: (Optional) List<str>
            List of tokens of the text.
      :param input_ids: (Optional) sequence_length torch long tensor
            Indices of input sequence tokens in the vocabulary.
            They are numerical representations of tokens that build the input sequence.
      :param input_mask: (Optional) torch long tensor
            1 marks that input ids are actual word, 0 marks padding.
      :param token_type_ids: (Optional) torch long tensor
            Used to differentiate between text_a and text_b.
    """

    def __init__(
        self,
        unique_id,
        text,
        language,
        words,
        label,
        tokens=None,
        input_ids=None,
        input_mask=None,
        token_type_ids=None,
    ):
        """Constructor"""
        self.unique_id = unique_id
        self.text = text
        self.language = language
        self.words = words
        self.label = label
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.length = len(input_ids)

    def __repr__(self):
        """String representation of the different attributes of the class."""
        return (
            " unique id: "
            + self.unique_id
            + " text: "
            + self.text
            + " language: "
            + self.language
            + " label: "
            + " ".join(
                [
                    NER_TYPES["panx"][label] if label != -100 else X_TOKEN
                    for label in self.label
                ]
            )
        )

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def get_text(self):
        return self.text


def extract_features(tokenizer, words, labels, lang, class_types):
    tokens = [CLS_TOKEN]
    label_ids = [1]
    for word, label in zip(words, labels):
        if isinstance(tokenizer, XLMTokenizer):
            word_tokens = tokenizer.tokenize(word, lang=lang)
        else:
            word_tokens = tokenizer.tokenize(word)
        if len(word) != 0 and len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend([class_types.index(label)] + [1] * (len(word_tokens) - 1))

    tokens += [SEP_TOKEN]
    label_ids += [1]

    # print(
    #     "words:",
    #     words,
    #     " tokens:",
    #     tokens,
    #     " labels:",
    #     labels,
    #     " label_ids:",
    #     label_ids,
    # )

    input_ids, input_mask, token_type_ids = convert2ids(tokenizer, tokens)

    return input_ids, input_mask, token_type_ids, tokens, label_ids


def _parse_xner(data_path, tokenizer, split, lang, class_types):
    # From XTREME
    process_egs = []
    process_egs_dict = {}
    if not os.path.exists(data_path):
        logger.info("[Warming] file {} not exists".format(data_path))
        return []

    i = 0
    id_ = split + "_" + lang + "_" + str(i)
    subword_len_counter = 0

    with open(data_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    (
                        input_ids,
                        input_mask,
                        token_type_ids,
                        tokens,
                        label_ids,
                    ) = extract_features(tokenizer, words, labels, lang, class_types)

                    example = NERExample(
                        unique_id=id_,
                        text=" ".join(words),
                        language=lang,
                        words=words,
                        label=label_ids,
                        tokens=tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        token_type_ids=token_type_ids,
                    )
                    process_egs.append(example)
                    process_egs_dict.update({id_: example})

                    i += 1
                    id_ = split + "_" + lang + "_" + str(i)
                    words = []
                    labels = []
                    subword_len_counter = 0
                else:
                    print(
                        f"guid_index",
                        id_,
                        words,
                        labels,
                        subword_len_counter,
                    )
            else:
                splits = line.split("\t")
                word = splits[0]

                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            (
                input_ids,
                input_mask,
                token_type_ids,
                tokens,
                label_ids,
            ) = extract_features(tokenizer, words, labels, lang, class_types)
            example = NERExample(
                unique_id=id_,
                text=" ".join(words),
                language=lang,
                words=words,
                label=label_ids,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
            )

            process_egs.append(example)
            process_egs_dict.update({id_: example})

    return process_egs, process_egs_dict


class Processor(object):
    """
    Processor of natural language inference
    """

    def __init__(self, args, tokenizer):
        self.data_format = args.data_format
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length  # TODO not used here
        self.class_types = NER_TYPES[args.data_name]

    def read_split(self, lang, split_name):
        file_path = os.path.join(
            os.path.join(self.data_root, "PERLANG", lang), split_name
        )

        process_egs, process_egs_dict = _parse_xner(
            file_path,
            self.tokenizer,
            split_name,
            lang,
            self.class_types,
        )

        return process_egs, process_egs_dict

    def next_batch(
        self, dataset, batch_size, data_split=None, identifier=None, identifiers=None
    ):
        if identifiers:
            examples_all = [dataset.get_item_by_id(id_) for id_ in identifiers]
            examples = random.choices(examples_all, k=batch_size)
        else:
            if identifier:
                examples = [dataset.get_item_by_id(identifier)]
            else:
                examples = data_split.next_items(batch_size)

        input_ids_l, lengths_l, input_masks_l, token_type_ids_l, labels_l = (
            [],
            [],
            [],
            [],
            [],
        )

        # Find the maximum length of input_ids across all examples
        max_sent_len = max([example.length for example in examples])

        # Padding
        for example in examples:
            input_ids_l.append(
                example.input_ids
                + [pad_token_label_id] * (max_sent_len - example.length)
            )

            input_masks_l.append(
                example.input_mask
                + [pad_token_label_id] * (max_sent_len - example.length)
            )

            token_type_ids_l.append(
                example.token_type_ids
                + [pad_token_label_id] * (max_sent_len - example.length)
            )

            labels_l.append(
                example.label + [pad_token_label_id] * (max_sent_len - example.length)
            )
            lengths_l.append(example.length)

        return (
            {
                "input_ids": LongTensor(input_ids_l),
                "lengths": LongTensor(lengths_l),
                "input_masks": LongTensor(input_masks_l),
                "token_type_ids": LongTensor(token_type_ids_l),
                "labels": LongTensor(labels_l),
            },
            examples,
            examples,
        )
