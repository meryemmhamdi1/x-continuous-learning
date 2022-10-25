import logging
from consts import INTENT_TYPES, SLOT_TYPES
import csv, json, re, os
from torch import LongTensor
from processors.utils import clean_text

logger = logging.getLogger(__name__)

def _parse_schuster(data_path, tokenizer, split, lang, intent_set=[], slot_set=["O", "X"], seen_examples=[]):
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
            intent = line[0]
            if intent not in intent_set:
                intent_set.append(intent)

            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != '':
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2]}
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

            sub_tokens = ['[CLS]']
            sub_slots = ['X']
            for j, token in enumerate(tokens):
                sub_sub_tokens = tokenizer.tokenize(token)
                sub_tokens += sub_sub_tokens
                for k, sub_token in enumerate(sub_sub_tokens):
                    if k == 0:
                        sub_slots.append(slots[j])
                    else:
                        sub_slots.append('X')

            sub_tokens += ['[SEP]']
            sub_slots.append('X')
            assert len(sub_slots) == len(sub_tokens)

            id_ = split + "_" + lang + "_" + str(i)

            process_egs.append((' '.join(tokens), sub_tokens, intent, sub_slots, id_))
            process_egs_dict.update({id_: (' '.join(tokens), sub_tokens, intent, sub_slots, id_)})

    return process_egs, process_egs_dict, intent_set, slot_set


def _parse_jarvis(data_path, tokenizer,  split, lang, intent_set=[]):
    """
    Processes json formatted text (was used for in-house Adobe data)
    """
    process_egs = []
    process_egs_dict = {}
    with open(data_path) as fp:
        i = 0
        for entry in json.load(fp):
            intent = entry['intent']
            if intent not in intent_set:
                intent_set.append(intent)
            words = entry['text'].lower().strip().split(' ')
            if len(words) >= 3 and words[-2].endswith('?'):
                words[-2] = words[-2][:-1]
            tokenized_words = ['[CLS]'] + tokenizer.tokenize(' '.join(words)) + ['[SEP]']
            process_egs.append((''.join(words), list(tokenized_words),  intent, id_))
            id_ = split + "_" + lang + "_" + str(i)
            process_egs_dict.update({id_: (''.join(words), list(tokenized_words),  intent, id_)})
            i += 1
    return process_egs, process_egs_dict, intent_set


def _parse_mtop_simplified(data_path, intent_set=[], slot_set=["O", "X"]):
    """
    To process the flat representation of MTOP by taking the top level in the hierarchical representation
    """
    process_egs = []
    slot_set_unique = []
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            domain = line[3]
            intent = domain+":"+line[0].split(":")[1]
            slot_splits = re.split(',|，', line[1])
            utterance = line[2]

            if intent not in intent_set:
                intent_set.append(intent)

            locale = line[4]
            decoupled_form = line[5]

            slot_line = []
            if line[1] != '':
                for item in slot_splits:
                    if item != '':
                        item_splits = item.split(":")
                        assert len(item_splits) == 4
                        slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[3]}
                        slot_line.append(slot_item)

            token_part = json.loads(line[6])

            tokens = token_part["tokens"]
            tokenSpans = token_part["tokenSpans"]
            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start = tokenspan["start"]
                    if slot_item["slot"] not in slot_set_unique:
                        slot_set_unique.append(slot_item["slot"])
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

            process_egs.append((' '.join(tokens), intent, slots))

    return process_egs, intent_set, slot_set_unique


def _parse_mtop(data_path, tokenizer, split, lang, intent_set=[], slot_set=["O", "X"]):
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

            intent_str = domain+":"+line[0].split(":")[1]

            intent = intent_set.index(intent_str)

            slot_splits = re.split(',|，', line[1])
            utterance = line[2]

            if intent_str not in intent_set:
                intent_set.append(intent_str)

            if domain not in distinct_domains:
                distinct_domains.append(domain)

            locale = line[4]
            decoupled_form = line[5]

            slot_line = []
            if line[1] != '':
                for item in slot_splits:
                    if item != '':
                        item_splits = item.split(":")
                        assert len(item_splits) == 4
                        slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[3]}
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

                    if slot_item["slot"] not in domain_intent_slot_dict[domain][intent_str]:
                        domain_intent_slot_dict[domain][intent_str].append(slot_item["slot"])

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

            sub_tokens = ['[CLS]']
            sub_slots = ['X']
            for j, token in enumerate(tokens):
                sub_sub_tokens = tokenizer.tokenize(token)
                sub_tokens += sub_sub_tokens
                for k, sub_token in enumerate(sub_sub_tokens):
                    if k == 0:
                        sub_slots.append(slots[j])
                    else:
                        sub_slots.append('X')

            sub_tokens += ['[SEP]']
            sub_slots.append('X')
            assert len(sub_slots) == len(sub_tokens)

            id_ = split+"_"+lang+"_"+str(i)
            process_egs.append([' '.join(tokens), sub_tokens, intent, sub_slots, id_])
            process_egs_dict.update({id_: (' '.join(tokens), sub_tokens, intent, sub_slots, id_)})

    return process_egs, process_egs_dict, intent_set, slot_set

def _parse_multi_atis(data_path, tokenizer, split, lang, intent_set=[], slot_set=["O", "X"]):
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
                slots = slots[:len(tokens)]
            elif len(slots) < len(tokens):
                tokens = tokens[:len(slots)]

            sub_tokens = ['[CLS]']
            sub_slots = ['X']
            for j, token in enumerate(tokens):
                bert_toks = tokenizer.tokenize(token)
                sub_tokens += bert_toks
                tag = slots[j]
                if tag not in slot_set:
                    slot_set.append(tag)

                if tag.startswith('B'):
                    cont_tag = 'I' + tag[1:]
                    if cont_tag not in slot_set:
                        slot_set.append(cont_tag)
                    sub_slots.extend([tag] + [cont_tag] * (len(bert_toks) - 1))
                else:
                    sub_slots.extend([tag] * len(bert_toks))

            sub_tokens += ['[SEP]']
            sub_slots.append('X')
            assert len(sub_slots) == len(sub_tokens)

            id_ = split+"_"+lang+"_"+str(i)
            process_egs.append([' '.join(tokens), sub_tokens, intent, sub_slots, id_])
            process_egs_dict.update({id_: (' '.join(tokens), sub_tokens, intent, sub_slots, id_)})

    return process_egs, process_egs_dict, intent_set, slot_set

class Processor(object):
    """
    Processor of multilingual task-oriented dialog benchmarks including MTOP, MultiATIS
    """

    def __init__(self, args, tokenizer):
        self.data_name = args.data_name
        self.data_format = args.data_format
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.class_types = INTENT_TYPES[self.data_name]
        self.slot_types = SLOT_TYPES[self.data_name]
        self.func_map = {"schuster": _parse_schuster, # tsv
                         "jarvis":_parse_jarvis, # json
                         "mtop": _parse_mtop, # txt
                         "multiatis": _parse_multi_atis} # tsv

    def read_split(self, lang, split_name):
        file_path = os.path.join(os.path.join(self.data_root, lang), split_name)

        process_egs, process_egs_dict, self.class_types, self.slot_types = self.func_map[self.data_name](file_path + "." + self.data_format,
                                                                                                         self.tokenizer,
                                                                                                         split_name,
                                                                                                         lang,
                                                                                                         self.class_types,
                                                                                                         self.slot_types)

        return process_egs, process_egs_dict


    def next_batch(self, batch_size, data_split=None, identifier=None):
        if identifier:
            examples = [self.get_item_by_id(identifier)]
        else:
            examples = data_split.next_items(batch_size)

        max_sent_len = 0

        input_ids, lengths, intent_labels, slot_labels, token_type_ids, input_masks, \
            input_texts, input_identifiers = [], [], [], [], [], [], [], []

        for example in examples:
            input_texts.append(example[0])

            cur_input_ids = self.tokenizer.convert_tokens_to_ids(example[1])
            assert len(cur_input_ids) == len(example[1])
            input_ids.append(cur_input_ids)

            max_sent_len = max(max_sent_len, len(example[1]))

            lengths.append(len(cur_input_ids))

            intent_labels.append(example[2])

            if self.use_slots:
                assert len(cur_input_ids) == len(example[3])
                slot_labels_sub = []
                for slot in example[3]:
                    slot_labels_sub.append(self.slot_types.index(slot))
                slot_labels.append(slot_labels_sub)

            input_identifiers.append(example[4])

        # Padding
        for i in range(batch_size):
            input_masks.append([1] * len(input_ids[i]) + [0] * (max_sent_len - len(input_ids[i])))
            input_ids[i] += [0] * (max_sent_len - len(input_ids[i]))

            token_type_ids.append([1 for _ in input_ids[i]])
            # attention_mask.append([int(x > 0) for x in input_ids[i]])
            if self.use_slots:
                slot_labels[i] += [0] * (max_sent_len - len(slot_labels[i]))

        # Convert to LongTensors
        slot_labels = LongTensor(slot_labels)
        input_ids = LongTensor(input_ids)
        lengths = LongTensor(lengths)
        intent_labels = LongTensor(intent_labels)
        token_type_ids = LongTensor(token_type_ids)
        input_masks = LongTensor(input_masks)
        # attention_mask = LongTensor(attention_mask)

        return (input_ids, lengths, token_type_ids, input_masks, intent_labels,
                slot_labels, input_texts, input_identifiers), examples
