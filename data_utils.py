import csv
import json

from io import open
from torch import LongTensor
import re
import random
import os
import ast
from torch.utils.data import Dataset


# detect pattern
# detect <TIME>
pattern_time1 = re.compile(r"[0-9]+[ap]")
pattern_time2 = re.compile(r"[0-9]+[;.h][0-9]+")
pattern_time3 = re.compile(r"[ap][.][am]")
pattern_time4 = range(2000, 2020)
# pattern_time5: token.isdigit() and len(token) == 3

pattern_time_th1 = re.compile(r"[\u0E00-\u0E7F]+[0-9]+")
pattern_time_th2 = re.compile(r"[0-9]+[.]*[0-9]*[\u0E00-\u0E7F]+")
pattern_time_th3 = re.compile(r"[0-9]+[.][0-9]+")

# detect <LAST>
pattern_last1 = re.compile(r"[0-9]+min")
pattern_last2 = re.compile(r"[0-9]+h")
pattern_last3 = re.compile(r"[0-9]+sec")

# detect <DATE>
pattern_date1 = re.compile(r"[0-9]+st")
pattern_date2 = re.compile(r"[0-9]+nd")
pattern_date3 = re.compile(r"[0-9]+rd")
pattern_date4 = re.compile(r"[0-9]+th")

remove_list = ["'s", "'ll", "'ve", "'d", "'m"]


class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch
        if shuffle_between_epoch:
            print("SHUFFLING DONE HERE")
            random.shuffle(self.items)

    def next_items(self, batch_size):
        if self.cur_idx == 0 and self.shuffle_between_epoch:
            random.shuffle(self.items)
            print("SHUFFLING DONE HERE")
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx: end_idx]
        else:
            first_part = items[start_idx: self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0: remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    @property
    def size(self):
        return len(self.items)


def clean_text(token_list, lang):
    """
    Applying the same pre-processing on NLU as in the latest AAAI 2020 publication
    taken from https://github.com/zliucr/mixed-language-training
    :param token_list:
    :param lang:
    :return:
    """
    token_list_clean = []
    for token in token_list:
        new_token = token
        # detect <TIME>
        if lang != "th" and (bool(re.match(pattern_time1, token))
                             or bool(re.match(pattern_time2, token))
                             or bool(re.match(pattern_time3, token))
                             or token in pattern_time4 or (token.isdigit() and len(token) == 3)):

            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        if lang == "th" and (bool(re.match(pattern_time_th1, token))
                             or bool(re.match(pattern_time_th2, token))
                             or bool(re.match(pattern_time_th3, token))):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        # detect <LAST>
        if lang == "en" and (bool(re.match(pattern_last1, token))
                             or bool(re.match(pattern_last2, token))
                             or bool(re.match(pattern_last3, token))):
            new_token = "<LAST>"
            token_list_clean.append(new_token)
            continue
        # detect <DATE>
        if lang == "en" and (bool(re.match(pattern_date1, token))
                             or bool(re.match(pattern_date2, token))
                             or bool(re.match(pattern_date3, token)) or bool(re.match(pattern_date4, token))):
            new_token = "<DATE>"
            token_list_clean.append(new_token)
            continue
        # detect <LOCATION>
        if lang != "th" and (token.isdigit() and len(token) == 5):
            new_token = "<LOCATION>"
            token_list_clean.append(new_token)
            continue
        # detect <NUMBER>
        if token.isdigit():
            new_token = "<NUMBER>"
            token_list_clean.append(new_token)
            continue
        if lang == "en" and ("n't" in token):
            new_token = "not"
            token_list_clean.append(new_token)
            continue
        if lang == "en":
            for item in remove_list:
                if item in token:
                    new_token = token.replace(item, "")
                    break

        token_list_clean.append(new_token)

    assert len(token_list_clean) == len(token_list)

    return token_list_clean


def _parse_tsv(data_path, tokenizer, split, lang, intent_set=[], slot_set=["O", "X"], seen_examples=[]):
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
            #tokens = token_part["tokenizations"][0]["tokens"]
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

    return process_egs, intent_set, slot_set, process_egs_dict


def _parse_json(data_path, tokenizer,  split, lang, intent_set=[]):
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
    return process_egs, intent_set, process_egs_dict


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

            #if i <= 0:
            #    print("utterance:", utterance, " slots:", slots, " tokens: ", tokens, " intent:", intent)

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
            # if i == 16:
            #     break
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

    return process_egs, process_egs_dict


class NLUDataset(Dataset):
    """  """
    def __init__(self,
                 data_path,
                 setup_option,
                 setup_cillia,
                 multi_head_out,
                 use_mono,
                 tokenizer,
                 data_format,
                 use_slots,
                 seed,
                 languages,
                 order_class,
                 order_lang,
                 order_lst,
                 num_intent_tasks,
                 num_lang_tasks,
                 memory_size=0,
                 intent_types=[],
                 slot_types=["O", "X"]):

        self.tokenizer = tokenizer
        self.use_slots = use_slots
        self.data_format = data_format

        self.intent_types = intent_types

        self.slot_types = slot_types
        self.data_path = data_path

        self.seed = seed

        self.order_class = order_class
        self.order_lang = order_lang
        self.order_lst = order_lst.split("_")

        self.languages = languages
        self.setup_option = setup_option
        self.setup_cillia = setup_cillia
        self.multi_head_out = multi_head_out
        self.use_mono = use_mono

        self.memory_size = memory_size

        random.seed(self.seed)

        self.train_set = {lang: [] for lang in languages}
        self.dev_set = {lang: [] for lang in languages}
        self.test_set = {lang: [] for lang in languages}

        self.process_egs_dict_global = {}

        for lang in self.languages:
            print("----------lang:", lang)
            print("Reading train split ... ")
            self.train_set[lang] = self.read_split(lang, "train")
            print("Reading dev split ... ")
            self.dev_set[lang] = self.read_split(lang, "eval")
            print("Reading test split ... ")
            self.test_set[lang] = self.read_split(lang, "test")

        if self.setup_option == "cil":
            """
            Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL"
            - Stream consisting of different combinations of classes from single language each time.
            - We then average over all languages.
            => Each stream consists of one language each time
            """
            self.train_stream = {lang: [] for lang in languages}
            self.dev_stream = {lang: [] for lang in languages}
            self.test_stream = {lang: [] for lang in languages}

            for lang in self.languages:
                ordered_train, ordered_intents = self.partition_per_intent(self.train_set[lang])
                ordered_dev, _ = self.partition_per_intent(self.dev_set[lang],
                                                           keys=ordered_intents) # using the same intent types order as the train

                ordered_test, _ = self.partition_per_intent(self.test_set[lang],
                                                            keys=ordered_intents) # using the same intent types order as the train

                self.test_stream[lang] = {"subtask_"+str(i): {} for i in range(0, len(self.intent_types), num_intent_tasks)}

                for i in range(0, len(self.intent_types), num_intent_tasks):
                    int_task_train = []
                    int_task_dev = []
                    int_task_test = []

                    for j, intent in enumerate(ordered_intents[i:i+num_intent_tasks]):
                        if self.multi_head_out or self.use_mono:
                            # print(i, j, self.intent_types[intent], " ordered_train[intent]:", len(ordered_train[intent]))
                            for eg in ordered_train[intent]:
                                eg[2] = j
                                int_task_train.append(eg)

                            for eg in ordered_dev[intent]:
                                eg[2] = j
                                int_task_dev.append(eg)

                            for eg in ordered_test[intent]:
                                eg[2] = j
                                int_task_test.append(eg)
                        else:
                            int_task_train.extend(ordered_train[intent])
                            int_task_dev.extend(ordered_dev[intent])
                            int_task_test.extend(ordered_test[intent])

                    self.train_stream[lang].append({"intent_list": list(map(lambda x: self.intent_types[x],
                                                                            ordered_intents[i:i+num_intent_tasks])),
                                                    "examples": AugmentedList(int_task_train,
                                                                              shuffle_between_epoch=True),
                                                    "size": len(int_task_train),
                                                    "lang": lang})

                    self.dev_stream[lang].append({"intent_list": list(map(lambda x: self.intent_types[x],
                                                                          ordered_intents[i:i+num_intent_tasks])),
                                                  "examples": AugmentedList(int_task_dev,
                                                                            shuffle_between_epoch=True),
                                                  "size": len(int_task_dev),
                                                  "lang": lang})

                    self.test_stream[lang]["subtask_"+str(i)] = {"intent_list": list(map(lambda x: self.intent_types[x],
                                                                                         ordered_intents[i:i+num_intent_tasks])),
                                                                 "lang": lang,
                                                                 "examples": AugmentedList(int_task_test),
                                                                 "size": len(int_task_test)}

        elif self.setup_option == "cil-other":
            # TODO adjust other too
            """
            Setup 2: CIL with other option:  incremental version of cil where previous intents' subtasks are added
            in addition to other labels for subsequent intents' subtasks

                Subtask 1:
                item | label
                1    | x
                2    | x
                3    | other
                4    | other
                5    | other
                6    | other

                Subtask 2:
                item | label
                1    | x
                2    | x
                3    | y
                4    | y
                5    | other
                6    | other

                Subtask 3:
                item | label
                1    | x
                2    | x
                3    | y
                4    | y
                5    | z
                6    | z

            """

            self.train_stream = {lang: [] for lang in languages}
            self.dev_stream = {lang: [] for lang in languages}
            self.test_stream = {lang: [] for lang in languages}

            for lang in self.languages:
                ordered_train, ordered_intents = self.partition_per_intent(self.train_set[lang])
                ordered_dev, _ = self.partition_per_intent(self.dev_set[lang],
                                                           keys=ordered_intents) # using the same intent types order as the train

                ordered_test, _ = self.partition_per_intent(self.test_set[lang],
                                                            keys=ordered_intents) # using the same intent types order as the train

                int_incremental_task_train = []
                int_incremental_task_dev = []
                int_incremental_task_test = []

                covered_intents = []
                self.test_stream[lang] = {"subtask_"+str(i): {} for i in range(0, len(self.intent_types), num_intent_tasks)}
                for i in range(0, len(self.intent_types), num_intent_tasks):
                    int_other_task_train = []
                    int_other_task_dev = []

                    for intent in ordered_intents[i:i+num_intent_tasks]:
                        covered_intents.append(intent)
                        int_incremental_task_train.extend(ordered_train[intent])
                        int_incremental_task_dev.extend(ordered_dev[intent])
                        int_incremental_task_test.extend(ordered_test[intent])

                    for intent in ordered_intents:
                        if intent not in covered_intents:
                            int_other_task_train.extend(self.set_intent_to_other(ordered_train[intent]))
                            int_other_task_dev.extend(self.set_intent_to_other(ordered_train[intent]))

                    self.train_stream[lang].append({"intent_list": ordered_intents[i:i+num_intent_tasks],
                                                    "examples": AugmentedList(int_incremental_task_train
                                                                              + int_other_task_train,
                                                                              shuffle_between_epoch=True),
                                                    "size": len(int_incremental_task_train),
                                                    "lang": lang})

                    self.dev_stream[lang].append({"intent_list": ordered_intents[i:i+num_intent_tasks],
                                                  "examples": AugmentedList(int_incremental_task_dev
                                                                            + int_other_task_dev,
                                                                            shuffle_between_epoch=True),
                                                  "size": len(int_incremental_task_dev),
                                                  "lang": lang})

                    self.test_stream[lang]["subtask_"+str(i)] = {"intent_list": ordered_intents[i:i+num_intent_tasks],
                                                                 "lang": lang,
                                                                 "examples": AugmentedList(int_incremental_task_test),
                                                                 "size": len(int_incremental_task_test)}

        elif self.setup_option == "cll":
            """
            Setup 3: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning"
            - Stream consisting of different combinations of languages.
            => Each stream sees all intents
            """

            ordered_langs = self.partition_per_lang(self.train_set)

            self.train_stream = [{"lang": lang,
                                  "examples": AugmentedList(self.train_set[lang],
                                                            shuffle_between_epoch=True),
                                  "size": len(self.train_set[lang])}
                                 for lang in ordered_langs]

            self.dev_stream = [{"lang": lang,
                                "examples": AugmentedList(self.dev_set[lang],
                                                          shuffle_between_epoch=True),
                                "size": len(self.dev_set[lang])}
                               for lang in ordered_langs]

            self.test_stream = {lang:
                                    {"examples": AugmentedList(self.test_set[lang]),
                                     "size": len(self.test_set[lang])}
                                for lang in ordered_langs}

        elif self.setup_option == "cil-ll":
            """
            Setup 4: Cross-CIL-LL: "Cross-lingual combinations of languages/intents"
            - Stream consisting of different combinations
            """
            self.train_stream = []
            self.dev_stream = []
            self.test_stream = []

            ordered_langs = self.partition_per_lang(self.train_set)

            ordered_train = {lang: [] for lang in ordered_langs}
            ordered_dev = {lang: [] for lang in ordered_langs}
            ordered_test = {lang: [] for lang in ordered_langs}
            ordered_intents = {lang: [] for lang in ordered_langs}

            for lang in ordered_langs:
                ordered_train[lang], ordered_intents[lang] = self.partition_per_intent(self.train_set[lang])
                ordered_dev[lang], _ = self.partition_per_intent(self.dev_set[lang],
                                                                 keys=ordered_intents[lang]) # using the same intent types order as train

                ordered_test[lang], _ = self.partition_per_intent(self.test_set[lang],
                                                                  keys=ordered_intents[lang]) # using the same intent types order as train

            # CIL/LL Matrix consists of language rows and intent columns
            if self.setup_cillia == "intents": # Horizontally goes linearly over all intents of each languages batch before moving to the next languages batch
                for j in range(0, len(ordered_langs), num_lang_tasks):
                    lang_batch = ordered_langs[j:j+num_lang_tasks]
                    for i in range(0, len(self.intent_types), num_intent_tasks):
                        int_lang_task_train = []
                        int_lang_task_dev = []
                        int_lang_task_test = {lang: [] for lang in self.languages}

                        intent_batches = []
                        for lang in lang_batch:
                            intent_batch = ordered_intents[lang][i:i+num_intent_tasks]
                            intent_batches.extend(intent_batch)
                            for intent in intent_batch:
                                int_lang_task_train += ordered_train[lang][intent]
                                int_lang_task_dev += ordered_dev[lang][intent]
                                int_lang_task_test[lang].extend(ordered_test[lang][intent])

                        self.train_stream.append({"lang": lang_batch,
                                                  "intent": intent_batches,
                                                  "examples": AugmentedList(int_lang_task_train,
                                                                            shuffle_between_epoch=True),
                                                  "size": len(int_lang_task_train)})

                        self.dev_stream.append({"lang": lang_batch,
                                                "intent": intent_batches,
                                                "examples": AugmentedList(int_lang_task_dev,
                                                                          shuffle_between_epoch=True),
                                                "size": len(int_lang_task_dev)})

                        self.test_stream.append({lang: {
                                                  "intent": intent_batches,
                                                  "examples": AugmentedList(int_lang_task_test[lang]),
                                                  "size": len(int_lang_task_test[lang])}
                                                 for lang in int_lang_task_test})

            else: # Vertically goes linearly over all languages of each intent batch before moving to the next intents batch
                for i in range(0, len(self.intent_types), num_intent_tasks):
                    for j in range(0, len(ordered_langs), num_lang_tasks):
                        int_lang_task_train = []
                        int_lang_task_dev = []
                        int_lang_task_test = {lang: [] for lang in self.languages}
                        lang_batch = ordered_langs[j:j+num_lang_tasks]
                        intent_batches = []
                        for lang in lang_batch:
                            intent_batch = ordered_intents[lang][i:i+num_intent_tasks]
                            intent_batches.extend(intent_batch)
                            for intent in intent_batch:
                                int_lang_task_train += ordered_train[lang][intent]
                                int_lang_task_dev += ordered_dev[lang][intent]
                                int_lang_task_test[lang].extend(ordered_test[lang][intent])

                        self.train_stream.append({"lang": lang_batch,
                                                  "intent": intent_batches,
                                                  "examples": AugmentedList(int_lang_task_train,
                                                                            shuffle_between_epoch=True),
                                                  "size": len(int_lang_task_train)})

                        self.dev_stream.append({"lang": lang_batch,
                                                "intent": intent_batches,
                                                "examples": AugmentedList(int_lang_task_dev,
                                                                          shuffle_between_epoch=True),
                                                "size": len(int_lang_task_dev)})

                        self.test_stream.append({lang: {
                                                  "intent": intent_batches,
                                                  "examples": AugmentedList(int_lang_task_test[lang]),
                                                  "size": len(int_lang_task_test[lang])}
                                                 for lang in int_lang_task_test})

        elif self.setup_option == "multi-incr-cil":
            """
            Setup 5: A weaker version of Multi-task/Joint Learning where we gradually fine-tune on the incremental
            of multi-task at each class subtask independently (when testing on subtask list L, we incrementally
            train up to that language)
            """

            self.train_stream = {lang: [] for lang in languages}
            self.dev_stream = {lang: [] for lang in languages}
            self.test_stream = {lang: {} for lang in languages}

            for lang in self.languages:
                ordered_train, ordered_intents = self.partition_per_intent(self.train_set[lang])

                ordered_dev, _ = self.partition_per_intent(self.dev_set[lang],
                                                           keys=ordered_intents) # using the same intent types order as the train

                ordered_test, _ = self.partition_per_intent(self.test_set[lang],
                                                            keys=ordered_intents) # using the same intent types order as the train

                ## Train
                inc_intents_set = []
                for i in range(num_intent_tasks, len(self.intent_types), num_intent_tasks):
                    inc_intents_set.append(ordered_intents[0:i])

                if i < len(self.intent_types):
                    inc_intents_set.append(ordered_intents[0:len(self.intent_types)])

                print("inc_intents_set:", len(inc_intents_set))
                inc_train_set = {str(intents_l): [] for intents_l in inc_intents_set}
                for joined_intents_l in inc_train_set.keys():
                    for j, intent in enumerate(ast.literal_eval(joined_intents_l)):
                        if self.multi_head_out:
                            for eg in ordered_train[intent]:
                                eg[2] = j
                                inc_train_set[joined_intents_l].append(eg)
                        else:
                            inc_train_set[joined_intents_l].extend(ordered_train[intent])

                self.train_stream[lang] = [{"intent_list": list(map(lambda x: self.intent_types[x],
                                                                    ast.literal_eval(joined_intents_l))),
                                            "lang": lang,
                                            "examples": AugmentedList(inc_train_set[joined_intents_l],
                                                                      shuffle_between_epoch=True),
                                            "size": len(inc_train_set[joined_intents_l])}
                                           for joined_intents_l in inc_train_set]

                ## Dev
                inc_dev_set = {str(intents_l): [] for intents_l in inc_intents_set}
                for joined_intents_l in inc_dev_set.keys():
                    for j, intent in enumerate(ast.literal_eval(joined_intents_l)):
                        if self.multi_head_out:
                            for eg in ordered_dev[intent]:
                                eg[2] = j
                                inc_dev_set[joined_intents_l].append(eg)
                        else:
                            inc_dev_set[joined_intents_l].extend(ordered_dev[intent])

                self.dev_stream[lang] = [{"intent_list": list(map(lambda x: self.intent_types[x],
                                                                  ast.literal_eval(joined_intents_l))),
                                          "lang": lang,
                                          "examples": AugmentedList(inc_dev_set[joined_intents_l]),
                                          "size": len(inc_dev_set[joined_intents_l])}
                                         for joined_intents_l in inc_dev_set]

                ## Test
                self.test_stream[lang] = {"subtask_"+str(i): {} for i in range(len(inc_intents_set))}
                for i, subtask in enumerate(self.test_stream[lang].keys()):
                    int_task_test = []
                    for j, intent in enumerate(inc_intents_set[i]):
                        if self.multi_head_out:
                            for eg in ordered_test[intent]:
                                eg[2] = j
                                int_task_test.append(eg)
                        else:
                            int_task_test.extend(ordered_test[intent])

                # for i in range(0, len(self.intent_types), num_intent_tasks):
                #     int_task_test = []
                #     for j, intent in enumerate(ordered_intents[i:i+num_intent_tasks]):
                #         if self.multi_head_out:
                #             for eg in ordered_test[intent]:
                #                 eg[2] = j
                #                 int_task_test.append()
                #         else:
                #             int_task_test.extend(ordered_test[intent])

                    self.test_stream[lang][subtask] = {"intent_list": list(map(lambda x: self.intent_types[x],
                                                                               inc_intents_set[i])),
                                                       "lang": lang,
                                                       "examples": AugmentedList(int_task_test),
                                                       "size": len(int_task_test)}

        elif self.setup_option == "multi-incr-cll":
            """
            Setup 6: A weaker version of Multi-task/Joint Learning where we gradually fine-tune on the incremental
            of multi-task at each language independently (when testing on language L we incrementally train up to that
            language)
            """
            ordered_langs = self.partition_per_lang(self.train_set)

            ## Train
            inc_langs_set = []
            for i, lang in enumerate(ordered_langs):
                inc_langs_set.append(ordered_langs[0:i+1])

            inc_train_set = {"-".join(lang_l): [] for lang_l in inc_langs_set}
            for lang_l in inc_train_set.keys():
                for lang in lang_l.split("-"):
                    inc_train_set[lang_l].extend(self.train_set[lang])

            # print("inc_train_set.keys():", inc_train_set.keys())
            #
            # print("inc_train_set[lang_l]:", inc_train_set["en"])

            # for lang_l in inc_train_set:
            #     print("lang_l:", lang_l)

            self.train_stream = [{"lang": lang_l,
                                  "examples": AugmentedList(inc_train_set[lang_l],
                                                            shuffle_between_epoch=True),
                                  "size": len(inc_train_set[lang_l])}
                                 for lang_l in inc_train_set]

            # debug_stuffy = AugmentedList(inc_train_set["en"],
            #                              shuffle_between_epoch=True)
            #
            # batch, _ = self.next_batch(16, debug_stuffy)
            # input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, input_identifiers = batch
            # print("input_identifiers:", input_identifiers)
            # exit(0)

            ## Dev
            inc_dev_set = {"-".join(lang_l): [] for lang_l in inc_langs_set}
            for lang_l in inc_dev_set.keys():
                for lang in lang_l.split("-"):
                    inc_dev_set[lang_l].extend(self.dev_set[lang])

            # print("inc_dev_set.keys():", inc_dev_set.keys())
            #
            # print("inc_dev_set[lang_l]:", inc_dev_set["en"][:10])

            self.dev_stream = [{"lang": lang_l,
                                "examples": AugmentedList(inc_dev_set[lang_l],
                                                          shuffle_between_epoch=True),
                                "size": len(inc_dev_set[lang_l])}
                               for lang_l in inc_dev_set]

            ## Test
            self.test_stream = {}
            for lang in self.languages:
                self.test_stream.update({lang: {"examples": AugmentedList(self.test_set[lang]),
                                                "size": len(self.test_set[lang])}})

        elif self.setup_option == "cll-er_kd":
            """
            Sanity Check for memory issues in ER/KD
            """
            ordered_langs = self.partition_per_lang(self.train_set)

            ## Train
            mem_langs_set = []
            for i, lang in enumerate(ordered_langs):
                mem_langs_set.append(ordered_langs[0:i+1])

            mem_train_set = {"-".join(lang_l): [] for lang_l in mem_langs_set}
            for i, lang_l in enumerate(mem_train_set.keys()):
                mem_langs = lang_l.split("-")[:i]
                for lang in mem_langs:
                    mem_train_set[lang_l].append(self.train_set[lang][:self.memory_size//len(self.languages)])  # TODO Add portion of memory here => added each task alone

            # print("self.train_set[en]:", self.train_set["en"])

            self.train_stream = [{"lang": lang_l[i],
                                  "examples": AugmentedList(self.train_set[lang_l[i]],
                                                            shuffle_between_epoch=True),
                                  "size": len(self.train_set[lang_l[i]]),  # main size
                                  "memory": [AugmentedList(mem) for mem in mem_train_set["-".join(lang_l)]],
                                  "size_memory": [len(mem) for mem in mem_train_set["-".join(lang_l)]]# each task with its memory size but you don't need that anyways
                                  }
                                 for i, lang_l in enumerate(mem_langs_set)]

            # print("self.dev_set[en]:", self.dev_set["en"][:10])

            # debug_stuffy = AugmentedList(self.train_set["en"],
            #                              shuffle_between_epoch=True)

            # print("AFTER SHUFFLING self.train_set[en]:", self.train_set["en"][:10])

            # batch, _ = self.next_batch(16, debug_stuffy)
            # input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, input_identifiers = batch
            # print("input_identifiers:", input_identifiers)
            # exit(0)

            ## Dev
            self.dev_stream = [{"lang": lang,
                                "examples": AugmentedList(self.dev_set[lang],
                                                          shuffle_between_epoch=True),
                                "size": len(self.dev_set[lang])}
                               for lang in ordered_langs]

            ## Test
            self.test_stream = {}
            for lang in self.languages:
                self.test_stream.update({lang: {"examples": AugmentedList(self.test_set[lang]),
                                                "size": len(self.test_set[lang])}})

        elif self.setup_option == "multi":
            """
            Setup 5: Multi-task/Joint Learning: train on all languages and intent classes at the same time
            """
            train_set_all = []
            for lang in self.train_set:
                train_set_all += self.train_set[lang]

            self.train_stream = {"examples": AugmentedList(train_set_all,
                                                           shuffle_between_epoch=True),
                                 "size": len(train_set_all)}

            dev_set_all = []
            for lang in self.languages:
                dev_set_all += self.dev_set[lang]

            # dev_set_all = [self.dev_set[lang] for lang in self.languages]
            self.dev_stream = {"examples": AugmentedList(dev_set_all,
                                                         shuffle_between_epoch=True),
                               "size": len(dev_set_all)}

            self.test_stream = {}
            for lang in self.languages:
                self.test_stream.update({lang: {"examples": AugmentedList(self.test_set[lang]),
                                                "size": len(self.test_set[lang])}})

    def get_item_by_id(self, id_):
        return self.process_egs_dict_global[id_]

    def save_global_process_egs_dict(self, process_egs_dict):
        self.process_egs_dict_global.update(process_egs_dict)

    def read_split(self, lang, split):
        """

        :param fpaths:
        :return:
        """

        intent_set = self.intent_types
        slot_set = self.slot_types
        file_path = os.path.join(os.path.join(self.data_path, lang),
                                 split)

        if self.data_format == "tsv":
            process_egs, intent_set, slot_set, process_egs_dict = _parse_tsv(file_path + "-" + lang + ".tsv",
                                                                             self.tokenizer,
                                                                             split,
                                                                             lang,
                                                                             intent_set,
                                                                             slot_set)

        elif self.data_format == "json":
            process_egs, intent_set, process_egs_dict = _parse_json(file_path + ".json",
                                                                    self.tokenizer,
                                                                    split,
                                                                    lang,
                                                                    intent_set)
        else:
            process_egs, process_egs_dict = _parse_mtop(file_path + ".txt",
                                                        self.tokenizer,
                                                        split,
                                                        lang,
                                                        intent_set,
                                                        slot_set)

        process_egs_shuffled = random.sample(process_egs,
                                             k=len(process_egs))

        self.save_global_process_egs_dict(process_egs_dict)

        return process_egs_shuffled

    def partition_per_intent(self, processed_egs, keys=None):
        intent_dict = {intent: [] for intent in range(len(self.intent_types))}
        for eg in processed_egs:
            intent_dict[eg[2]].append(eg)

        if keys:
            return {k: intent_dict[k] for k in keys}, keys

        if self.order_class == 2:
            keys = list(intent_dict.keys())
            print(keys)
            random.shuffle(keys)
        else:
            # if len(self.order_lst) == 0 or "en" in self.order_lst:
            reverse_flag = False
            if self.order_class == 0:
                reverse_flag = True

            keys = sorted(intent_dict,
                          key=lambda k: len(intent_dict[k]),
                          reverse=reverse_flag)

        ordered_dict = {k: intent_dict[k] for k in keys}
        return ordered_dict, keys

    def set_intent_to_other(self, processed_egs):
        other_intents = []
        for eg in processed_egs:
            other_eg = (eg[0], eg[1], "OTHER", eg[3], eg[4])
            other_intents.append(other_eg)

        return other_intents

    def partition_per_lang(self, train_set):
        if self.order_lang == 2:
            ordered_langs = train_set.keys()
            random.shuffle(ordered_langs)
        else:
            if len("".join(self.order_lst)) == 0:
                reverse_flag = False
                if self.order_lang == 0:# decreasing frequency
                    reverse_flag = True

                ordered_langs = sorted(train_set,
                                       key=lambda lang: len(train_set[lang]),
                                       reverse=reverse_flag)
            else:
                ordered_langs = self.order_lst

            print("ordered_langs:", ordered_langs)
        return ordered_langs

    def get_batch_one(self, identifier):
        example = self.get_item_by_id(identifier)
        max_sent_len = 0
        input_ids, lengths, intent_labels, slot_labels, token_type_ids, input_masks, \
        input_texts, input_identifiers = [], [], [], [], [], [], [], []

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
        for i in range(1):
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
                slot_labels, input_texts, input_identifiers), example

    def next_batch(self, batch_size, data_split):
        """
        Usual next batch mechanism for pre-training base model
        :param batch_size:
        :param data_split: train or test
        :return:
        """
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
            input_masks.append([1]*len(input_ids[i]) + [0]*(max_sent_len - len(input_ids[i])))
            input_ids[i] += [0] * (max_sent_len - len(input_ids[i]))

            token_type_ids.append([1 for _ in input_ids[i]])
            #attention_mask.append([int(x > 0) for x in input_ids[i]])
            if self.use_slots:
                slot_labels[i] += [0] * (max_sent_len - len(slot_labels[i]))

        # Convert to LongTensors
        slot_labels = LongTensor(slot_labels)
        input_ids = LongTensor(input_ids)
        lengths = LongTensor(lengths)
        intent_labels = LongTensor(intent_labels)
        token_type_ids = LongTensor(token_type_ids)
        input_masks = LongTensor(input_masks)
        #attention_mask = LongTensor(attention_mask)

        return (input_ids, lengths, token_type_ids, input_masks, intent_labels,
                slot_labels, input_texts, input_identifiers), examples
