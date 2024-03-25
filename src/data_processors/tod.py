import logging
import csv, json, re, os, copy
from torch import LongTensor
import random

from src.consts import INTENT_TYPES, SLOT_TYPES, CLS_TOKEN, SEP_TOKEN, X_TOKEN
from src.data_processors.utils import clean_text, convert2ids

logger = logging.getLogger(__name__)


class TODExample(object):
    """
    A single training/test example for task-oriented dialogue task.
    Args:
      :param unique_id: str
            Unique id for the example.
      :param text: str
            The untokenized text of the dialog utterance.
      :param language: str
            The language of the utterance.
      :param label: int
            The intent label of the example (indexed by the fixed list of intent ontologies in ../consts.py)
      :param slot_labels: (Optional) List<int>
            Slot labels in BIO annotation one-to-one mapped to the subtokens converted to ids
            Optional depending on whether slots are used.
      :param tokens: (Optional) List<str>
            List of tokens of the utterance.
      :param length: (Optional) int
            Actual length of each utterance (in terms of number of tokens).
      :param input_ids: (Optional) torch long tensor TODO double check the type of this
            Indices of input sequence tokens in the vocabulary.
            They are numerical representations of tokens that build the input sequence.
      :param input_mask: (Optional) torch long tensor TODO doubel check the type of this
            1 marks that input ids are actual word, 0 marks padding
      :param token_type_ids: (Optional) torch long tensor TODO doubel check the type of this
            This is not needed for this TOD task definition.
    """

    def __init__(
        self,
        unique_id,
        text,
        language,
        label,
        slot_labels=None,
        tokens=None,
        input_ids=None,
        input_mask=None,
        token_type_ids=None,
    ):
        """Constructor"""
        self.unique_id = unique_id
        self.text = text
        self.language = language
        self.label = label
        self.slot_labels = slot_labels
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.length = len(self.input_ids)

    def __repr__(self):
        """String representation of all elements in "the example."""
        return (
            " unique id: "
            + self.unique_id
            + " utterance: "
            + self.text
            + " language: "
            + self.language
            + " intent label: "
            + str(self.label)
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


def _parse_schuster(
    data_path,
    tokenizer,
    split,
    lang,
    intent_set=[],
    slot_set=["O", X_TOKEN],
):
    """
    Taken from https://github.com/zliucr/mixed-language-training
    Input:
        data_path: the path of data
        intent_set: set of intent (empty if it is train data)
        slot_set: set of slot type (empty if it is train data)
    Output:
        data_tsv: {"text": [[token1, token2, ...], ...], "slot": [[slot_type1, slot_type2, ...], ...],
                  "intent": [intent_type, ...]}
        intent_set: set of intent
        slot_set: set of slot type
    """
    slot_type_list = ["alarm", "datetime", "location", "reminder", "weather"]
    process_egs = []
    process_egs_dict = {}
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            id_ = split + "_" + lang + "_" + str(i)
            intent = line[0]
            if intent not in intent_set:
                intent_set.append(intent)

            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != "":
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    slot_item = {
                        "start": item_splits[0],
                        "end": item_splits[1],
                        "slot": item_splits[2],
                    }
                    flag = False
                    for slot_type in slot_type_list:
                        if slot_type in slot_item["slot"]:
                            flag = True

                    if not flag:
                        slot_flag = False
                        break
                    slot_line.append(slot_item)

            if not slot_flag:
                # slot flag not correct
                continue

            token_part = json.loads(line[4])
            tokens = clean_text(token_part["tokenizations"][0]["tokens"], lang)
            tokenSpans = token_part["tokenizations"][0]["tokenSpans"]

            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start = tokenspan["start"]
                    # if int(start) >= int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(slot_item["start"]) < int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel:
                    slots.append("O")

            assert len(slots) == len(tokens)

            sub_tokens = [CLS_TOKEN]
            sub_slots = [X_TOKEN]
            for j, token in enumerate(tokens):
                sub_sub_tokens = tokenizer.tokenize(token)
                sub_tokens += sub_sub_tokens
                for k, sub_token in enumerate(sub_sub_tokens):
                    if k == 0:
                        sub_slots.append(slots[j])
                    else:
                        sub_slots.append(X_TOKEN)

            sub_tokens += [SEP_TOKEN]
            sub_slots.append(X_TOKEN)
            assert len(sub_slots) == len(sub_tokens)

            input_ids, input_mask, token_type_ids = convert2ids(tokenizer, sub_tokens)

            slot_labels_ids = [slot_set.index(slot) for slot in sub_slots]

            example = TODExample(
                unique_id=id_,
                text=" ".join(tokens),
                language=lang,
                label=intent,
                slot_labels=slot_labels_ids,
                tokens=sub_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
            )

            process_egs.append(example)
            process_egs_dict.update({id_: example})

    return process_egs, process_egs_dict, intent_set, slot_set


def _parse_jarvis(data_path, tokenizer, split, lang, intent_set=[]):
    """
    Processes json formatted text (was used for in-house Adobe data)
    """
    process_egs = []
    process_egs_dict = {}
    with open(data_path) as fp:
        i = 0
        for entry in json.load(fp):
            intent = entry["intent"]
            if intent not in intent_set:
                intent_set.append(intent)
            words = entry["text"].lower().strip().split(" ")
            if len(words) >= 3 and words[-2].endswith("?"):
                words[-2] = words[-2][:-1]
            tokenized_words = (
                [CLS_TOKEN] + tokenizer.tokenize(" ".join(words)) + [SEP_TOKEN]
            )
            id_ = split + "_" + lang + "_" + str(i)

            input_ids, input_mask, token_type_ids = convert2ids(
                tokenizer, list(tokenized_words)
            )

            example = TODExample(
                unique_id=id_,
                text=" ".join(words),
                language=lang,
                label=intent,
                tokens=list(tokenized_words),
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
            )

            process_egs.append(example)
            process_egs_dict.update({id_: example})
            i += 1

    return process_egs, process_egs_dict, intent_set


def _parse_mtop(
    data_path, tokenizer, split, lang, intent_set=[], slot_set=["O", X_TOKEN]
):
    """
    To process the flat representation of MTOP by taking the top level in the hierarchical representation
    """
    process_egs = []
    process_egs_dict = {}
    distinct_domains = []
    distinct_slots = []
    domain_intent_slot_dict = {}
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            domain = line[3]
            if domain not in domain_intent_slot_dict:
                domain_intent_slot_dict.update({domain: {}})

            intent_str = domain + ":" + line[0].split(":")[1]

            intent = intent_set.index(intent_str)

            slot_splits = re.split(",|，", line[1])
            utterance = line[2]

            if intent_str not in intent_set:
                intent_set.append(intent_str)

            if domain not in distinct_domains:
                distinct_domains.append(domain)

            locale = line[4]
            decoupled_form = line[5]

            slot_line = []
            if line[1] != "":
                for item in slot_splits:
                    if item != "":
                        item_splits = item.split(":")
                        assert len(item_splits) == 4
                        slot_item = {
                            "start": item_splits[0],
                            "end": item_splits[1],
                            "slot": item_splits[3],
                        }
                        slot_line.append(slot_item)

            token_part = json.loads(line[6])

            tokens = token_part["tokens"]
            tokenSpans = token_part["tokenSpans"]
            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    if slot_item["slot"] not in distinct_slots:
                        distinct_slots.append(slot_item["slot"])

                    if intent_str not in domain_intent_slot_dict[domain]:
                        domain_intent_slot_dict[domain].update({intent_str: []})

                    if (
                        slot_item["slot"]
                        not in domain_intent_slot_dict[domain][intent_str]
                    ):
                        domain_intent_slot_dict[domain][intent_str].append(
                            slot_item["slot"]
                        )

                    start = tokenspan["start"]
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(slot_item["start"]) < int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel:
                    slots.append("O")

            assert len(slots) == len(tokens)

            sub_tokens = [CLS_TOKEN]
            sub_slots = [X_TOKEN]
            for j, token in enumerate(tokens):
                sub_sub_tokens = tokenizer.tokenize(token)
                sub_tokens += sub_sub_tokens
                for k, sub_token in enumerate(sub_sub_tokens):
                    if k == 0:
                        sub_slots.append(slots[j])
                    else:
                        sub_slots.append(X_TOKEN)

            sub_tokens += [SEP_TOKEN]
            sub_slots.append(X_TOKEN)
            assert len(sub_slots) == len(sub_tokens)

            id_ = split + "_" + lang + "_" + str(i)

            input_ids, input_mask, token_type_ids = convert2ids(tokenizer, sub_tokens)

            slot_labels_ids = [slot_set.index(slot) for slot in sub_slots]

            example = TODExample(
                unique_id=id_,
                text=" ".join(tokens),
                language=lang,
                label=intent,
                slot_labels=slot_labels_ids,
                tokens=sub_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
            )

            process_egs.append(example)
            process_egs_dict.update({id_: example})

    return process_egs, process_egs_dict, intent_set, slot_set


def _parse_multi_atis(
    data_path, tokenizer, split, lang, intent_set=[], slot_set=["O", X_TOKEN]
):
    """
    To process the flat representation of ATIS++ by taking the top level in the hierarchical representation
    """

    process_egs = []
    process_egs_dict = {}
    distinct_domains = []
    distinct_slots = []
    skipped = 0
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            intent_str = line[3]

            if intent_str not in intent_set:
                intent_set.append(intent_str)

            intent = intent_set.index(intent_str)

            slots = line[2].split(" ")
            utterance = line[1]

            if intent_str not in intent_set:
                intent_set.append(intent_str)

            tokens = utterance.split()
            if len(slots) > len(tokens):
                slots = slots[: len(tokens)]
            elif len(slots) < len(tokens):
                tokens = tokens[: len(slots)]

            sub_tokens = [CLS_TOKEN]
            sub_slots = [X_TOKEN]
            for j, token in enumerate(tokens):
                bert_toks = tokenizer.tokenize(token)
                sub_tokens += bert_toks
                tag = slots[j]
                if tag not in slot_set:
                    slot_set.append(tag)

                if tag.startswith("B"):
                    cont_tag = "I" + tag[1:]
                    if cont_tag not in slot_set:
                        slot_set.append(cont_tag)
                    sub_slots.extend([tag] + [cont_tag] * (len(bert_toks) - 1))
                else:
                    sub_slots.extend([tag] * len(bert_toks))

            sub_tokens += [SEP_TOKEN]
            sub_slots.append(X_TOKEN)
            assert len(sub_slots) == len(sub_tokens)

            id_ = split + "_" + lang + "_" + str(i)

            input_ids, input_mask, token_type_ids = convert2ids(tokenizer, sub_tokens)

            slot_labels_ids = [slot_set.index(slot) for slot in sub_slots]

            example = TODExample(
                unique_id=id_,
                text=" ".join(tokens),
                language=lang,
                label=intent,
                slot_labels=slot_labels_ids,
                tokens=sub_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
            )

            process_egs.append(example)
            process_egs_dict.update({id_: example})

    return process_egs, process_egs_dict, intent_set, slot_set


class Processor(object):
    """
    Processor of multilingual task-oriented dialog benchmarks including MTOP, MultiATIS
    """

    def __init__(self, args, tokenizer):
        self.data_name = args.data_name
        self.data_format = args.data_format
        self.data_root = args.data_root
        if self.data_name == "multiatis":
            self.data_root = os.path.join(self.data_root, "multi_atis_all")
        self.use_slots = args.use_slots
        self.tokenizer = tokenizer
        self.class_types = INTENT_TYPES[self.data_name]
        self.slot_types = SLOT_TYPES[self.data_name]
        self.func_map = {
            "schuster": _parse_schuster,  # tsv
            "jarvis": _parse_jarvis,  # json
            "mtop": _parse_mtop,  # txt
            "multiatis": _parse_multi_atis,  # tsv
        }

    def read_split(self, lang, split_name):
        file_path = os.path.join(os.path.join(self.data_root, lang), split_name)

        (
            process_egs,
            process_egs_dict,
            self.class_types,
            self.slot_types,
        ) = self.func_map[self.data_name](
            file_path + "." + self.data_format,
            self.tokenizer,
            split_name,
            lang,
            self.class_types,
            self.slot_types,
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

        (
            input_ids_l,
            lengths_l,
            input_masks_l,
            token_type_ids_l,
            intent_labels_l,
            slot_labels_l,
        ) = ([], [], [], [], [], [])

        # Find the maximum length of input_ids across all examples
        max_sent_len = max([example.length for example in examples])

        # print(
        #     "BEFORE ",
        #     " len(examples[0].input_ids):",
        #     len(examples[0].input_ids),
        #     " length:",
        #     examples[0].length,
        # )

        # Padding
        for example in examples:
            input_ids_l.append(
                example.input_ids + [0] * (max_sent_len - example.length)
            )

            input_masks_l.append(
                example.input_mask + [0] * (max_sent_len - example.length)
            )

            token_type_ids_l.append(
                example.token_type_ids + [0] * (max_sent_len - example.length)
            )

            intent_labels_l.append(example.label)

            lengths_l.append(example.length)

            if self.use_slots:
                slot_labels_l.append(
                    example.slot_labels + [0] * (max_sent_len - example.length)
                )

        # print(
        #     "AFTER",
        #     " len(examples[0].input_ids):",
        #     len(examples[0].input_ids),
        #     " length:",
        #     examples[0].length,
        #     " input_ids_l[0]:",
        #     input_ids_l[0],
        #     " len(input_ids_l[0]):",
        #     len(input_ids_l[0]),
        # )

        return (
            {
                "input_ids": LongTensor(input_ids_l),
                "lengths": LongTensor(lengths_l),
                "input_masks": LongTensor(input_masks_l),
                "token_type_ids": LongTensor(token_type_ids_l),
                "labels": LongTensor(intent_labels_l),
                "slot_labels": LongTensor(slot_labels_l),
            },
            examples,
            examples,
        )
