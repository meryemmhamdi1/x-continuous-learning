import logging
import json, os, copy
import pandas as pd
import random

from transformers import XLMTokenizer
from torch import LongTensor

from consts import NLI_TYPES
from data_processors.utils import _read_tsv

logger = logging.getLogger(__name__)


class NLIExample(object):
    """
    A single training/test example for simple inference task.
    Given text_a and text_b, the purpose is to find the inference class (role of text_b with respect to text_a).
    Args:
      :param unique_id: str
            Unique id for the example.
      :param text_a: string
            The untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.
      :param text_b: (Optional) string
            The untokenized text of the second sequence. Only must be specified for sequence pair tasks.
      :param language: (Optional) string
            The language of text_a and text_b we assume we have one common language for both text_a and text_b
      :param label: int
            The class label of the example.
      :param tokens: (Optional) List<str>
            List of tokens of the utterance.
      :param length: (Optional) int
            Actual length of each utterance (in terms of number of tokens)
            Not really needed in this case.
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
        label,
        text_b=None,
        tokens=None,
        length=None,
        input_ids=None,
        input_mask=None,
        token_type_ids=None,
    ):
        """Constructor"""
        self.unique_id = unique_id
        self.text_a = text
        self.text_b = text_b
        self.language = language
        self.label = label
        self.tokens = tokens
        self.length = length
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids

    def __repr__(self):
        """String representation of the different attributes of the class."""
        return (
            " unique id: "
            + self.unique_id
            + " text_a: "
            + self.text_a
            + " text_b: "
            + self.text_b
            + " language: "
            + self.language
            + " label: "
            + self.label
        )

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def get_text(self):
        return self.text_a + " " + self.text_b


def _parse_xnli(
    data_path,
    tokenizer,
    max_seq_length,
    pad_token,
    mask_padding_with_zero,
    pad_token_segment_id,
    split,
    lang,
    class_types,
):
    process_egs = []
    process_egs_dict = {}
    lines = _read_tsv(data_path)
    for i, line in enumerate(lines):
        if i == 0:
            continue

        id_ = split + "_" + lang + "_" + str(i)
        text_a = line[0]
        text_b = line[1]
        label = class_types.index(line[2])

        if isinstance(tokenizer, XLMTokenizer):
            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_seq_length,
                lang=lang,
            )
        else:
            inputs = tokenizer.encode_plus(
                text_a, text_b, add_special_tokens=True, max_length=max_seq_length
            )
        # inputs = tokenizer(text_a, text_b, truncation='only_first', max_length=max_seq_length)

        input_ids = inputs["input_ids"]
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + (
            [pad_token_segment_id] * padding_length
        )  # pad_token_segment_id is 0

        assert (
            len(input_ids) == max_seq_length
        ), "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert (
            len(input_mask) == max_seq_length
        ), "Error with input length {} vs {}".format(len(input_mask), max_seq_length)
        assert (
            len(token_type_ids) == max_seq_length
        ), "Error with input length {} vs {}".format(
            len(token_type_ids), max_seq_length
        )

        example = NLIExample(
            unique_id=id_,
            text=text_a,
            text_b=text_b,
            label=label,
            language=lang,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        process_egs.append(example)
        process_egs_dict.update({id_: example})

    print("Saving the datasets for visualization purposes:")
    unique_ids = []
    text_as = []
    text_bs = []
    gold_labels = []
    input_ids_l = []
    input_masks_l = []
    token_type_ids_l = []
    for example in process_egs:
        unique_ids.append(example.unique_id)
        text_as.append(example.text)
        text_bs.append(example.text_b)
        gold_labels.append(example.label)
        input_ids_l.append(example.input_ids)
        input_masks_l.append(example.input_mask)
        token_type_ids_l.append(example.token_type_ids)

    df = pd.DataFrame(
        data={
            "unique_id": unique_ids,
            "text_a": text_as,
            "sentence2s": text_bs,
            "gold_labels": gold_labels,
            "input_ids": input_ids_l,
            "input_masks": input_masks_l,
            "token_type_ids": token_type_ids_l,
        }
    )
    df.to_csv(data_path + "OWN_processed_" + split + ".csv", index=False)
    return process_egs, process_egs_dict


class Processor(object):
    """
    Processor of natural language inference
    """

    def __init__(self, args, tokenizer):
        self.data_format = args.data_format
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.pad_token = args.pad_token
        self.mask_padding_with_zero = args.mask_padding_with_zero
        self.pad_token_segment_id = args.pad_token_segment_id
        self.class_types = NLI_TYPES

    def read_split(self, lang, split_name):
        file_path = os.path.join(
            os.path.join(self.data_root, "PERLANG", lang), split_name
        )

        process_egs, process_egs_dict = _parse_xnli(
            file_path + "." + self.data_format,
            self.tokenizer,
            self.max_seq_length,
            self.pad_token,
            self.mask_padding_with_zero,
            self.pad_token_segment_id,
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

        for example in examples:
            input_ids_l.append(example.input_ids)
            lengths_l.append(example.length)
            input_masks_l.append(example.input_mask)
            token_type_ids_l.append(example.token_type_ids)
            labels_l.append(example.label)

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
