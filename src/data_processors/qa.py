""" Largely adapted from https://github.com/google-research/xtreme/blob/8ad18d8f5d765cab56fd50e2abe369191de9cc08/third_party/run_squad.py"""
import logging
import json, os
from tqdm import tqdm
import numpy as np
import torch
import random
from multiprocessing import Pool, cpu_count

from functools import partial
from src.consts import QA_TYPES
from src.data_processors.utils import (
    _is_whitespace,
    whitespace_tokenize,
    _improve_answer_span,
    _check_is_max_context,
    _new_check_is_max_context,
)

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        unique_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
        language="en",
    ):
        self.unique_id = unique_id
        self.qas_id = unique_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        self.language = language

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(
                    start_position_character + len(answer_text) - 1,
                    len(char_to_word_offset) - 1,
                )
            ]


class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        langs,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.langs = langs


def squad_convert_example_to_features(
    example,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    lang2id,
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            # logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if lang2id is None:
            sub_tokens = tokenizer.tokenize(token)
        else:
            sub_tokens = tokenizer.tokenize(token, lang=example.language)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            tokenizer,
            example.answer_text,
            lang=example.language,
            lang2id=lang2id,
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, max_length=max_query_length
    )
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_sentences_pair
    )

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        # print("truncated_query:", truncated_query)
        # print("len(spans):", len(spans), " doc_stride:", doc_stride, " len(all_doc_tokens):", len(all_doc_tokens)) #" span_doc_tokens:", len(span_doc_tokens), " len(example.doc_tokens): ", len(example.doc_tokens)

        # print("stride: ", max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens)
        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length
            - doc_stride
            - len(truncated_query)
            - sequence_pair_added_tokens,
            truncation_strategy="only_second"
            if tokenizer.padding_side == "right"
            else "only_first",
            return_token_type_ids=True,
        )
        # print("overflowing_tokens:", encoded_dict["overflowing_tokens"])

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][
                : encoded_dict["input_ids"].index(tokenizer.pad_token_id)
            ]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = (
                len(truncated_query) + sequence_added_tokens + i
                if tokenizer.padding_side == "right"
                else i
            )
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = (
            len(truncated_query) + sequence_added_tokens
        )
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if (
            "overflowing_tokens" not in encoded_dict
            or len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(
                spans, doc_span_index, doc_span_index * doc_stride + j
            )
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"]
                + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if lang2id is not None:
            lid = lang2id.get(example.language, lang2id["en"])
        else:
            lid = 0
        langs = [lid] * max_seq_length

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                langs=langs,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    threads=1,
    lang2id=None,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    # threads = min(threads, cpu_count())
    # with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
    #     annotate_ = partial(
    #         squad_convert_example_to_features,
    #         max_seq_length=max_seq_length,
    #         doc_stride=doc_stride,
    #         max_query_length=max_query_length,
    #         is_training=is_training,
    #         lang2id=lang2id
    #     )
    #     features = list(
    #         tqdm(
    #             p.imap(annotate_, examples, chunksize=32),
    #             total=len(examples),
    #             desc="convert squad examples to features",
    #         )
    #     )

    not_found_answers = 0
    all_unique_ids = []
    for i, example in enumerate(examples):
        feature = squad_convert_example_to_features(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
            lang2id=lang2id,
        )

        if len(feature) == 0:
            not_found_answers += 1
        features.append(feature)
        all_unique_ids.append(example.unique_id)

    # print("not_found_answers:", not_found_answers)
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for (
        example_features
    ) in (
        features
    ):  # tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    return features


def _parse_squad(data_path, tokenizer, args, split, lang):
    if split in ["train", "valid"]:
        evaluate = False
    else:
        evaluate = True

    process_egs_dict = {}
    with open(data_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in tqdm(input_data):
        title = entry["title"] if "title" in entry else ""
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = split + "_" + lang + "_" + str(qa["id"])
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]

                example = SquadExample(
                    unique_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                    language=lang,
                )

                examples.append(example)

    for i, example in enumerate(examples):
        process_egs_dict.update({examples[i].unique_id: examples[i]})

    return examples, process_egs_dict


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(
        self,
        unique_id,
        start_logits,
        end_logits,
        start_top_index=None,
        end_top_index=None,
        cls_logits=None,
    ):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


class Processor(object):
    """
    Processor of multilingual task-oriented dialog benchmarks including MTOP, MultiATIS
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.data_name = args.data_name
        self.data_format = args.data_format
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.class_types = QA_TYPES
        # self.class_types = []

    def read_split(self, lang, split_name):
        file_path = os.path.join(
            os.path.join(self.data_root, "PERLANG", lang), split_name
        )

        process_egs, process_egs_dict = _parse_squad(
            file_path + "." + self.data_format,
            self.tokenizer,
            self.args,
            split_name,
            lang,
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

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_seq_length,
            doc_stride=self.args.doc_stride,
            max_query_length=self.args.max_query_length,
            is_training=True,
            threads=1,
            lang2id=None,  # TODO NONE for now but should replace this with config.lang2id
        )

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_masks = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        start_positions = torch.tensor(
            [f.start_position for f in features], dtype=torch.long
        )
        end_positions = torch.tensor(
            [f.end_position for f in features], dtype=torch.long
        )
        cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        langs = torch.tensor([f.langs for f in features], dtype=torch.long)

        # print(
        #     "input_ids.shape:",
        #     input_ids.shape,
        #     " attention_masks.shape:",
        #     attention_masks.shape,
        #     " token_type_ids.shape:",
        #     token_type_ids.shape,
        #     " start_positions.shape:",
        #     start_positions.shape,
        # )

        return (
            {
                "input_ids": input_ids,
                "input_masks": attention_masks,
                "token_type_ids": token_type_ids,
                "start_positions": start_positions,
                "end_positions": end_positions,
            },
            examples,
            features,
        )
