import csv
import json

from io import open
from torch import LongTensor
import re
import random
import os

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
            random.shuffle(self.items)

    def next_items(self, batch_size):
        if self.cur_idx == 0 and self.shuffle_between_epoch:
            random.shuffle(self.items)
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
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
        if lang != "th" and ( bool(re.match(pattern_time1, token)) or bool(re.match(pattern_time2, token))
                              or bool(re.match(pattern_time3, token)) or token in pattern_time4 or (token.isdigit()
                                                                                                    and len(token)==3)):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        if lang == "th" and ( bool(re.match(pattern_time_th1, token)) or bool(re.match(pattern_time_th2, token))
                              or bool(re.match(pattern_time_th3, token)) ):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        # detect <LAST>
        if lang == "en" and ( bool(re.match(pattern_last1, token)) or bool(re.match(pattern_last2, token))
                              or bool(re.match(pattern_last3, token)) ):
            new_token = "<LAST>"
            token_list_clean.append(new_token)
            continue
        # detect <DATE>
        if lang == "en" and ( bool(re.match(pattern_date1, token)) or bool(re.match(pattern_date2, token))
                              or bool(re.match(pattern_date3, token)) or bool(re.match(pattern_date4, token))):
            new_token = "<DATE>"
            token_list_clean.append(new_token)
            continue
        # detect <LOCATION>
        if lang != "th" and ( token.isdigit() and len(token)==5 ):
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


def _parse_tsv(data_path, tokenizer, lang, intent_set=[], slot_set=["O", "X"], seen_examples=[]):
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
            tokens = token_part["tokenizations"][0]["tokens"]
            #tokens = clean_text(token_part["tokenizations"][0]["tokens"], lang)
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

            sub_tokens += ['[SEP']
            sub_slots.append('X')
            assert len(sub_slots) == len(sub_tokens)

            process_egs.append((' '.join(tokens), sub_tokens, intent, sub_slots, i))

    return process_egs, intent_set, slot_set


def _parse_json(data_path, tokenizer, intent_set=[]):
    """
    Processes json formatted text (was used for in-house Adobe data)
    """
    process_egs = []
    with open(data_path) as fp:
        for entry in json.load(fp):
            intent = entry['intent']
            if intent not in intent_set:
                intent_set.append(intent)
            words = entry['text'].lower().strip().split(' ')
            if len(words) >= 3 and words[-2].endswith('?'):
                words[-2] = words[-2][:-1]
            tokenized_words = ['[CLS]'] + tokenizer.tokenize(' '.join(words)) + ['[SEP]']
            process_egs.append((''.join(words), list(tokenized_words),  intent))
    return process_egs, intent_set


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


def _parse_mtop(data_path, tokenizer, intent_set=[], slot_set=["O", "X"]):
    """
    To process the flat representation of MTOP by taking the top level in the hierarchical representation
    """
    process_egs = []
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            domain = line[3]
            intent = domain+":"+line[0].split(":")[1]
            slot_splits = line[1].split(",")
            utterance = line[2]

            if intent not in intent_set:
                intent_set.append(intent)

            locale = line[4]
            decoupled_form = line[5]

            slot_line = []
            if line[1] != '':
                for item in slot_splits:
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

            if i <= 2:
                print("utterance:", utterance, " slots:", slots, " tokens: ", tokens, " intent:", intent)

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

            sub_tokens += ['[SEP']
            sub_slots.append('X')
            assert len(sub_slots) == len(sub_tokens)

            process_egs.append((' '.join(tokens), sub_tokens, intent, sub_slots))

    return process_egs


class Dataset:
    """  """
    def __init__(self, data_path, setup_option, tokenizer, data_format, use_slots, seed, languages, order_class,
                 order_lang, num_intent_tasks, num_lang_tasks, intent_types=[], slot_types=["O", "X"]):
        self.tokenizer = tokenizer
        self.use_slots = use_slots
        self.data_format = data_format

        self.intent_types = intent_types

        self.slot_types = slot_types
        self.data_path = data_path

        self.seed = seed
        self.order_class = order_class
        self.order_lang = order_lang
        self.languages = languages
        self.setup_option = setup_option

        random.seed(self.seed)

        self.train_set = {lang: [] for lang in languages}
        self.dev_set = {lang: [] for lang in languages}
        self.test_set = {lang: [] for lang in languages}

        for lang in self.languages:
            self.train_set[lang] = self.read_split(lang, "train")
            self.dev_set[lang] = self.read_split(lang, "eval")
            self.test_set[lang] = self.read_split(lang, "test")

        if self.setup_option == 1:
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
                ordered_dev, _ = self.partition_per_intent(self.dev_set[lang], ordered_keys=ordered_intents) # using the same intent types order as the train
                ordered_test, _ = self.partition_per_intent(self.test_set[lang], ordered_keys=ordered_intents) # using the same intent types order as the train

                for i in range(0, len(self.intent_types), num_intent_tasks):
                    int_task_train = []
                    int_task_dev = []
                    int_task_test = []

                    for intent in ordered_intents[i:i+num_intent_tasks]:
                        int_task_train.extend(ordered_train[intent])
                        int_task_dev.extend(ordered_dev[intent])
                        int_task_test.extend(ordered_test[intent])

                    self.train_stream[lang].append({"intent_list": ordered_intents[i:i+num_intent_tasks],
                                                    "stream": AugmentedList(int_task_train, shuffle_between_epoch=True)})
                    self.dev_stream[lang].append({"intent_list": ordered_intents[i:i+num_intent_tasks],
                                                  "stream": AugmentedList(int_task_dev, shuffle_between_epoch=True)})
                    self.test_stream[lang].append({"intent_list": ordered_intents[i:i+num_intent_tasks],
                                                   "stream": AugmentedList(int_task_test)})

        elif self.setup_option == 2:
            """
            Setup 2: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning" 
            - Stream consisting of different combinations of languages.
            => Each stream sees all intents
            """
            self.train_stream = {"all_intents": []}
            self.dev_stream = {"all_intents": []}
            self.test_stream = {"all_intents": []}

            ordered_langs = self.partition_per_lang(self.train_set)

            self.train_stream["all_intents"] = [{"lang": lang,
                                                 "stream": AugmentedList(self.train_set[lang], shuffle_between_epoch=True)}
                                                for lang in ordered_langs]
            self.dev_stream["all_intents"] = [{"lang": lang,
                                               "stream": AugmentedList(self.dev_set[lang], shuffle_between_epoch=True)}
                                              for lang in ordered_langs]
            self.test_stream["all_intents"] = [{"lang": lang, "stream": AugmentedList(self.test_set[lang])}
                                               for lang in ordered_langs]

        elif self.setup_option == 3:
            """
            Setup 3: Cross-CIL-LL: "Cross-lingual combinations of languages/intents"
            - Stream consisting of different combinations 
            """
            self.train_stream = {"all_paths": []}
            self.dev_stream = {"all_paths": []}
            self.test_stream = {"all_paths": []}

            ordered_langs = self.partition_per_lang(self.train_set)

            ordered_train = {lang: [] for lang in ordered_langs}
            ordered_dev = {lang: [] for lang in ordered_langs}
            ordered_test = {lang: [] for lang in ordered_langs}
            ordered_intents = {lang: [] for lang in ordered_langs}

            for lang in ordered_langs:
                ordered_train[lang], ordered_intents[lang] = self.partition_per_intent(self.train_set[lang])
                ordered_dev[lang], _ = self.partition_per_intent(self.dev_set[lang], ordered_keys=ordered_intents[lang]) # using the same intent types order as the train
                ordered_test[lang], _ = self.partition_per_intent(self.test_set[lang], ordered_keys=ordered_intents[lang]) # using the same intent types order as the train

            # CIL/LL Matrix consists of language rows and intent columns
            if self.setup_3 == "intents": # Horizontally goes linearly over all intents of each languages batch before moving to the next languages batch
                for j in range(0, len(ordered_langs), num_lang_tasks):
                    lang_batch = ordered_langs[j:j+num_lang_tasks]
                    for i in range(0, len(self.intent_types), num_intent_tasks):
                        int_lang_task_train = []
                        int_lang_task_dev = []
                        int_lang_task_test = []

                        for lang in lang_batch:
                            intent_batch = ordered_intents[lang][i:i+num_intent_tasks]
                            for intent in intent_batch:
                                int_lang_task_train += ordered_train[lang][intent]
                                int_lang_task_dev += ordered_dev[lang][intent]
                                int_lang_task_test += ordered_test[lang][intent]

                        self.train_stream["all_paths"].append({"lang": lang_batch, "intent": intent_batch,
                                                               "stream": AugmentedList(int_lang_task_train, shuffle_between_epoch=True)})
                        self.dev_stream["all_paths"].append({"lang": lang_batch, "intent": intent_batch,
                                                             "stream": AugmentedList(int_lang_task_dev, shuffle_between_epoch=True)})
                        self.test_stream["all_paths"].append({"lang": lang_batch, "intent": intent_batch,
                                                              "stream": AugmentedList(int_lang_task_test)})

            else: # Vertically goes linearly over all languages of each intent batch before moving to the next intents batch
                for i in range(0, len(self.intent_types), num_intent_tasks):
                    for j in range(0, len(ordered_langs), num_lang_tasks):
                        int_lang_task_train = []
                        int_lang_task_dev = []
                        int_lang_task_test = []
                        lang_batch = ordered_langs[j:j+num_lang_tasks]
                        for lang in lang_batch:
                            intent_batch = ordered_intents[lang][i:i+num_intent_tasks]
                            for intent in intent_batch:
                                int_lang_task_train += ordered_train[lang][intent]
                                int_lang_task_dev += ordered_dev[lang][intent]
                                int_lang_task_test += ordered_test[lang][intent]

                        self.train_stream["all_paths"].append({"lang": lang_batch, "intent": intent_batch,
                                                               "stream": AugmentedList(int_lang_task_train, shuffle_between_epoch=True)})
                        self.dev_stream["all_paths"].append({"lang": lang_batch, "intent": intent_batch,
                                                             "stream":AugmentedList(int_lang_task_dev, shuffle_between_epoch=True)})
                        self.test_stream["all_paths"].append({"lang": lang_batch, "intent": intent_batch,
                                                              "stream":AugmentedList(int_lang_task_test)})

        elif self.setup_option == 4:
            """
            Setup 4: Multi-task: train on all languages and intent classes at the same time 
            
            """
            self.train_stream = AugmentedList([self.train_set[lang] for lang in self.languages], shuffle_between_epoch=True)
            self.dev_stream = AugmentedList([self.dev_set[lang] for lang in self.languages], shuffle_between_epoch=True)
            self.test_stream = {}
            for lang in self.languages:
                self.test_stream.update({lang: AugmentedList(self.test_set[lang])})

    def read_split(self, lang, split):
        """

        :param fpaths:
        :return:
        """

        intent_set = self.intent_types
        slot_set = self.slot_types
        file_path = os.path.join(os.path.join(self.data_path, lang), split)
        if self.data_format == "tsv":
            process_egs = _parse_tsv(file_path + "-" + lang + ".tsv", self.tokenizer, lang, intent_set, slot_set)
        elif self.data_format == "json":
            process_egs = _parse_json(file_path + ".json", self.tokenizer, intent_set)
        else:
            process_egs = _parse_mtop(file_path + ".txt", self.tokenizer, intent_set, slot_set)

        process_egs_shuffled = random.sample(process_egs, k=len(process_egs))

        return process_egs_shuffled

    def partition_per_intent(self, processed_egs, ordered_keys=None):
        intent_dict = {intent: [] for intent in self.intent_types}
        for eg in processed_egs:
            intent_dict[eg[2]].append(eg)

        if ordered_keys:
            return [intent_dict[key] for key in ordered_keys], ordered_keys

        if self.order_class == 0: # decreasing frequency
            new_dict = sorted(intent_dict, key=lambda k: len(intent_dict[k]), reverse=True)
            return new_dict, ordered_keys

        elif self.order_class == 1: # increasing frequency
            new_dict = sorted(intent_dict, key=lambda k: len(intent_dict[k]))
            ordered_keys = new_dict.keys()
            return new_dict, ordered_keys
        else: # random frequency
            keys = intent_dict.keys()
            random.shuffle(keys)
            new_dict = {key: intent_dict[key] for key in keys}
            return new_dict, ordered_keys

    def partition_per_lang(self, train_set):
        if self.order_class == 0: # decreasing frequency
            sorted_langs = sorted(train_set, key=lambda lang: len(train_set[lang]), reverse=True)
            return sorted_langs.keys()

        elif self.order_class == 1: # increasing frequency
            sorted_langs = sorted(train_set, key=lambda lang: len(train_set[lang]))
            return sorted_langs.keys()

        else: # random frequency
            keys = train_set.keys()
            random.shuffle(keys)
            return keys

    def next_batch(self, batch_size, data_split, dev_langs):
        """
        Usual next batch mechanism for pre-training base model
        :param batch_size:
        :param data_split: train or test
        :return:
        """
        examples = []
        if len(dev_langs) != 0:
            for lang in dev_langs:
                examples.extend(data_split[lang].next_items(batch_size))
        else:
            examples = data_split.next_items(batch_size)

        max_sent_len = 0
        input_ids, lengths, intent_labels, slot_labels, token_type_ids, attention_mask, input_texts, input_identifiers \
            = [], [], [], [], [], [], [], []

        for example in examples:
            input_texts.append(example[0])

            cur_input_ids = self.tokenizer.convert_tokens_to_ids(example[1])
            assert len(cur_input_ids) == len(example[1])
            input_ids.append(cur_input_ids)

            max_sent_len = max(max_sent_len, len(example[1]))

            lengths.append(len(cur_input_ids))

            intent_labels.append(self.intent_types.index(example[2]))

            if self.use_slots:
                assert len(cur_input_ids) == len(example[3])
                slot_labels_sub = []
                for slot in example[3]:
                    slot_labels_sub.append(self.slot_types.index(slot))
                slot_labels.append(slot_labels_sub)

            input_identifiers.append(example[4])

        # Padding
        for i in range(batch_size):
            input_ids[i] += [0] * (max_sent_len - len(input_ids[i]))

            token_type_ids.append([1 for x in input_ids[i]])
            attention_mask.append([int(x > 0) for x in input_ids[i]])
            if self.use_slots:
                slot_labels[i] += [0] * (max_sent_len - len(slot_labels[i]))

        # Convert to LongTensors
        slot_labels = LongTensor(slot_labels)
        input_ids = LongTensor(input_ids)
        lengths = LongTensor(lengths)
        intent_labels = LongTensor(intent_labels)
        token_type_ids = LongTensor(token_type_ids)
        attention_mask = LongTensor(attention_mask)

        return (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), examples
