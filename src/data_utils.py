import ast
import random
import importlib
import sys
from torch.utils.data import Dataset
import os
import pickle

sys.path.append(os.getcwd())

# sys.path.append("/home1/mmhamdi/x-continuous-learning/src")
from src.data_processors.utils import *


class AugmentedList:
    """
    A class used to iterate over instances.

    ...

    Attributes
    ----------
    items : list
        List of items
    shuffle_between_epoch : bool
        whether to shuffle between epochs or not:
           - True in the case of training and validation streams
           - False in the case of testing streams

    Methods
    -------
    next_items(batch_size)
        Samples the next items in the batch

    size()
        Computes the number of items in the stream
    """

    def __init__(self, items, shuffle_between_epoch=False):
        """Constructor method
        :param items : list
            List of items
        :param shuffle_between_epoch : bool
            whether to shuffle between epochs or not:
            - True in the case of training and validation streams
            - False in the case of testing streams

        """
        self.items = items
        self.shuffle_between_epoch = shuffle_between_epoch

        if shuffle_between_epoch:
            random.shuffle(self.items)

        # Counter used to keep track of how many items have been sampled whether we are at the beginning of the epoch
        #         and whether we should reset the counter revisit from the beginning
        self.cur_idx = 0

    def next_items(self, batch_size):
        """Samples the next items in the batch

        :param batch_size : int
            The size of the batch to be retrieved/sampled

        :return: list
            A batch of items
        """
        if self.cur_idx == 0 and self.shuffle_between_epoch:
            random.shuffle(self.items)

        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx:end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0:remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    @property
    def size(self):
        """
        Returns
        -------
        :return: int
            number of items in the stream
        """
        return len(self.items)


class MultiPurposeDataset(Dataset):
    """
    A multi-purpose class for datasets from any downstream task among task-oriented dialog (MTOP, ATIS++), natural language inference (XNLI), and span-based question answering (TyDiQA, XQUAD)
    supporting different continual learning setups for designing different data streams and sequences following some language order

    ...

    Attributes
    ----------
    args : dictionary
        Dictionary of parsed arguments from argparse
    tokenizer : instance of transformers tokenizer (e.g. BertTokenizer, XLNetTokenizer, XLMRobertaTokenizer, etc)

    Methods
    -------
    next_items(batch_size)
        Samples the next items in the batch
    """

    def __init__(self, args, tokenizer):
        """Constructor method
        :param args : dictionary
            Dictionary of parsed arguments from argparse
        :param tokenizer : instance of transformers tokenizer (e.g. BertTokenizer, XLNetTokenizer, XLMRobertaTokenizer, etc)
        """

        self.args = args
        self.tokenizer = tokenizer
        self.processor = importlib.import_module(
            "src.data_processors." + args.task_name
        ).Processor(args, tokenizer)

        self.order_lst = args.order_lst.split("_")

        self.class_types = self.processor.class_types
        if args.task_name == "tod":
            self.slot_types = self.processor.slot_types
        else:
            self.slot_types = []

        random.seed(self.args.seed)

        self.process_egs_dict_global = {}

        self.train_set = {}
        self.dev_set = {}  # {lang: [] for lang in self.args.languages}
        self.test_set = {}

        for lang in self.args.languages:
            print("----------lang:", lang)
            print("Reading train split ... ")
            self.train_set.update({lang: self.read_split(lang, "train")})
            print("Reading dev split ... ")
            self.dev_set.update({lang: self.read_split(lang, "eval")})
            print("Reading test split ... ")
            self.test_set.update({lang: self.read_split(lang, "test")})

        if self.args.setup_opt == "cil":
            print("********** CIL")
            self.set_cil_streams()
        elif self.args.setup_opt == "cil-other":
            print("********** CIL-OTHER")
            self.set_cil_other_streams()
        elif self.args.setup_opt == "cll":
            print("********** CLL")
            self.set_cll_streams()
        elif self.args.setup_opt == "cil-ll":
            print("********* CIL-LL")
            self.set_cil_ll_streams()
        elif self.args.setup_opt == "multi-incr-cil":
            print("********* MULTI-INCR-CIL")
            self.set_multi_incr_cil_streams()
        elif self.args.setup_opt == "multi-incr-cll":
            print("********* MULTI-INCR-CLL")
            self.set_multi_incr_cll_streams()
        elif self.args.setup_opt == "cll-er_kd":
            print("********* CLL-ER-KD")
            self.set_cll_er_kd_streams()
        else:  # multi
            print("********* MULTI")
            self.set_multi_streams()

        self.streams = {
            "train": self.train_stream,
            "valid": self.dev_stream,
            "test": self.test_stream,
        }

    def save_global_process_egs_dict(self, process_egs_dict):
        self.process_egs_dict_global.update(process_egs_dict)

    def read_split(self, lang, split):
        """

        :param lang: the language of the data to be pre-processed
        :param split: train, test, or dev
        :return:
        """
        process_egs, process_egs_dict = self.processor.read_split(lang, split)

        process_egs_shuffled = random.sample(process_egs, k=len(process_egs))

        self.save_global_process_egs_dict(process_egs_dict)

        return process_egs_shuffled

    def set_cil_streams(self):  # TODO extend this to other classes than intents
        """
        Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL"
        - Stream consisting of different combinations of classes from single language each time.
        - We then average over all languages.
        => Each stream consists of one language each time
        """
        self.train_stream = {lang: [] for lang in self.args.languages}
        self.dev_stream = {lang: [] for lang in self.args.languages}
        self.test_stream = {lang: [] for lang in self.args.languages}

        for lang in self.args.languages:
            ordered_train, ordered_classes = self.partition_per_class(
                self.train_set[lang]
            )
            ordered_dev, _ = self.partition_per_class(
                self.dev_set[lang], keys=ordered_classes
            )  # using the same intent types order as the train

            ordered_test, _ = self.partition_per_class(
                self.test_set[lang], keys=ordered_classes
            )  # using the same intent types order as the train

            self.test_stream[lang] = {
                "subtask_" + str(i): {}
                for i in range(0, len(self.class_types), self.args.num_class_tasks)
            }

            for i in range(0, len(self.class_types), self.args.num_class_tasks):
                int_task_train = []
                int_task_dev = []
                int_task_test = []

                for j, intent in enumerate(
                    ordered_classes[i : i + self.args.num_class_tasks]
                ):
                    if self.args.multi_head_out or self.args.use_mono:
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

                self.train_stream[lang].append(
                    {
                        "class_list": list(
                            map(
                                lambda x: self.class_types[x],
                                ordered_classes[i : i + self.args.num_class_tasks],
                            )
                        ),
                        "examples": AugmentedList(
                            int_task_train, shuffle_between_epoch=False
                        ),  # True
                        "size": len(int_task_train),
                        "lang": lang,
                    }
                )

                self.dev_stream[lang].append(
                    {
                        "class_list": list(
                            map(
                                lambda x: self.class_types[x],
                                ordered_classes[i : i + self.args.num_class_tasks],
                            )
                        ),
                        "examples": AugmentedList(
                            int_task_dev, shuffle_between_epoch=False
                        ),  # True
                        "size": len(int_task_dev),
                        "lang": lang,
                    }
                )

                self.test_stream[lang]["subtask_" + str(i)] = {
                    "class_list": list(
                        map(
                            lambda x: self.class_types[x],
                            ordered_classes[i : i + self.args.num_class_tasks],
                        )
                    ),
                    "lang": lang,
                    "examples": AugmentedList(int_task_test),
                    "size": len(int_task_test),
                }

    def set_cil_other_streams(self):
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

        self.train_stream = {lang: [] for lang in self.args.languages}
        self.dev_stream = {lang: [] for lang in self.args.languages}
        self.test_stream = {lang: [] for lang in self.args.languages}

        for lang in self.args.languages:
            ordered_train, ordered_classes = self.partition_per_class(
                self.train_set[lang]
            )
            ordered_dev, _ = self.partition_per_class(
                self.dev_set[lang], keys=ordered_classes
            )  # using the same intent types order as the train

            ordered_test, _ = self.partition_per_class(
                self.test_set[lang], keys=ordered_classes
            )  # using the same intent types order as the train

            int_incremental_task_train = []
            int_incremental_task_dev = []
            int_incremental_task_test = []

            covered_intents = []
            self.test_stream[lang] = {
                "subtask_" + str(i): {}
                for i in range(0, len(self.class_types), self.args.num_class_tasks)
            }
            for i in range(0, len(self.class_types), self.args.num_class_tasks):
                int_other_task_train = []
                int_other_task_dev = []

                for intent in ordered_classes[i : i + self.args.num_class_tasks]:
                    covered_intents.append(intent)
                    int_incremental_task_train.extend(ordered_train[intent])
                    int_incremental_task_dev.extend(ordered_dev[intent])
                    int_incremental_task_test.extend(ordered_test[intent])

                for intent in ordered_classes:
                    if intent not in covered_intents:
                        int_other_task_train.extend(
                            self.set_intent_to_other(ordered_train[intent])
                        )
                        int_other_task_dev.extend(
                            self.set_intent_to_other(ordered_train[intent])
                        )

                self.train_stream[lang].append(
                    {
                        "intent_list": ordered_classes[
                            i : i + self.args.num_class_tasks
                        ],
                        "examples": AugmentedList(
                            int_incremental_task_train + int_other_task_train,
                            shuffle_between_epoch=False,
                        ),  # True
                        "size": len(int_incremental_task_train),
                        "lang": lang,
                    }
                )

                self.dev_stream[lang].append(
                    {
                        "intent_list": ordered_classes[
                            i : i + self.args.num_class_tasks
                        ],
                        "examples": AugmentedList(
                            int_incremental_task_dev + int_other_task_dev,
                            shuffle_between_epoch=False,
                        ),  # True
                        "size": len(int_incremental_task_dev),
                        "lang": lang,
                    }
                )

                self.test_stream[lang]["subtask_" + str(i)] = {
                    "intent_list": ordered_classes[i : i + self.args.num_class_tasks],
                    "lang": lang,
                    "examples": AugmentedList(int_incremental_task_test),
                    "size": len(int_incremental_task_test),
                }

    def set_cll_streams(self):
        """
        Setup 3: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning"
        - Stream consisting of different combinations of languages.
        => Each stream sees all intents
        """

        ordered_langs = self.partition_per_lang(self.train_set)

        self.train_stream = [
            {
                "lang": lang,
                "examples": AugmentedList(
                    self.train_set[lang], shuffle_between_epoch=False
                ),  # True
                "size": len(self.train_set[lang]),
            }
            for lang in ordered_langs
        ]

        self.dev_stream = [
            {
                "lang": lang,
                "examples": AugmentedList(
                    self.dev_set[lang], shuffle_between_epoch=False
                ),  # True
                "size": len(self.dev_set[lang]),
            }
            for lang in ordered_langs
        ]

        self.test_stream = {
            lang: {
                "examples": AugmentedList(self.test_set[lang]),
                "size": len(self.test_set[lang]),
            }
            for lang in ordered_langs
        }

    def set_cil_ll_streams(self):
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
            ordered_train[lang], ordered_intents[lang] = self.partition_per_class(
                self.train_set[lang]
            )
            ordered_dev[lang], _ = self.partition_per_class(
                self.dev_set[lang], keys=ordered_intents[lang]
            )  # using the same intent types order as train

            ordered_test[lang], _ = self.partition_per_class(
                self.test_set[lang], keys=ordered_intents[lang]
            )  # using the same intent types order as train

        # CIL/LL Matrix consists of language rows and intent columns
        if (
            self.setup_cillia == "intents"
        ):  # Horizontally goes linearly over all intents of each languages batch before moving to the next languages batch
            for j in range(0, len(ordered_langs), self.args.num_lang_tasks):
                lang_batch = ordered_langs[j : j + self.args.num_lang_tasks]
                for i in range(0, len(self.class_types), self.args.num_class_tasks):
                    int_lang_task_train = []
                    int_lang_task_dev = []
                    int_lang_task_test = {lang: [] for lang in self.args.languages}

                    intent_batches = []
                    for lang in lang_batch:
                        intent_batch = ordered_intents[lang][
                            i : i + self.args.num_class_tasks
                        ]
                        intent_batches.extend(intent_batch)
                        for intent in intent_batch:
                            int_lang_task_train += ordered_train[lang][intent]
                            int_lang_task_dev += ordered_dev[lang][intent]
                            int_lang_task_test[lang].extend(ordered_test[lang][intent])

                    self.train_stream.append(
                        {
                            "lang": lang_batch,
                            "intent": intent_batches,
                            "examples": AugmentedList(
                                int_lang_task_train, shuffle_between_epoch=False
                            ),  # True
                            "size": len(int_lang_task_train),
                        }
                    )

                    self.dev_stream.append(
                        {
                            "lang": lang_batch,
                            "intent": intent_batches,
                            "examples": AugmentedList(
                                int_lang_task_dev, shuffle_between_epoch=False
                            ),
                            "size": len(int_lang_task_dev),
                        }
                    )

                    self.test_stream.append(
                        {
                            lang: {
                                "intent": intent_batches,
                                "examples": AugmentedList(int_lang_task_test[lang]),
                                "size": len(int_lang_task_test[lang]),
                            }
                            for lang in int_lang_task_test
                        }
                    )

        else:  # Vertically goes linearly over all languages of each intent batch before moving to the next intents batch
            for i in range(0, len(self.class_types), self.args.num_class_tasks):
                for j in range(0, len(ordered_langs), self.args.num_lang_tasks):
                    int_lang_task_train = []
                    int_lang_task_dev = []
                    int_lang_task_test = {lang: [] for lang in self.args.languages}
                    lang_batch = ordered_langs[j : j + self.args.num_lang_tasks]
                    intent_batches = []
                    for lang in lang_batch:
                        intent_batch = ordered_intents[lang][
                            i : i + self.args.num_class_tasks
                        ]
                        intent_batches.extend(intent_batch)
                        for intent in intent_batch:
                            int_lang_task_train += ordered_train[lang][intent]
                            int_lang_task_dev += ordered_dev[lang][intent]
                            int_lang_task_test[lang].extend(ordered_test[lang][intent])

                    self.train_stream.append(
                        {
                            "lang": lang_batch,
                            "intent": intent_batches,
                            "examples": AugmentedList(
                                int_lang_task_train, shuffle_between_epoch=False
                            ),  # True
                            "size": len(int_lang_task_train),
                        }
                    )

                    self.dev_stream.append(
                        {
                            "lang": lang_batch,
                            "intent": intent_batches,
                            "examples": AugmentedList(
                                int_lang_task_dev, shuffle_between_epoch=False
                            ),  # True
                            "size": len(int_lang_task_dev),
                        }
                    )

                    self.test_stream.append(
                        {
                            lang: {
                                "intent": intent_batches,
                                "examples": AugmentedList(int_lang_task_test[lang]),
                                "size": len(int_lang_task_test[lang]),
                            }
                            for lang in int_lang_task_test
                        }
                    )

    def set_multi_incr_cil_streams(self):
        """
        Setup 5: A weaker version of Multi-task/Joint Learning where we gradually fine-tune on the incremental
        of multi-task at each class subtask independently (when testing on subtask list L, we incrementally
        train up to that language)
        """

        self.train_stream = {lang: [] for lang in self.args.languages}
        self.dev_stream = {lang: [] for lang in self.args.languages}
        self.test_stream = {lang: {} for lang in self.args.languages}

        for lang in self.args.languages:
            ordered_train, ordered_intents = self.partition_per_class(
                self.train_set[lang]
            )

            ordered_dev, _ = self.partition_per_class(
                self.dev_set[lang], keys=ordered_intents
            )  # using the same intent types order as the train

            ordered_test, _ = self.partition_per_class(
                self.test_set[lang], keys=ordered_intents
            )  # using the same intent types order as the train

            ## Train
            inc_intents_set = []
            for i in range(
                self.args.num_class_tasks,
                len(self.class_types),
                self.args.num_class_tasks,
            ):
                inc_intents_set.append(ordered_intents[0:i])

            if i < len(self.class_types):
                inc_intents_set.append(ordered_intents[0 : len(self.class_types)])

            print("inc_intents_set:", len(inc_intents_set))
            inc_train_set = {str(intents_l): [] for intents_l in inc_intents_set}
            for joined_intents_l in inc_train_set.keys():
                for j, intent in enumerate(ast.literal_eval(joined_intents_l)):
                    if self.args.multi_head_out:
                        for eg in ordered_train[intent]:
                            eg[2] = j
                            inc_train_set[joined_intents_l].append(eg)
                    else:
                        inc_train_set[joined_intents_l].extend(ordered_train[intent])

            self.train_stream[lang] = [
                {
                    "class_list": list(
                        map(
                            lambda x: self.class_types[x],
                            ast.literal_eval(joined_intents_l),
                        )
                    ),
                    "lang": lang,
                    "examples": AugmentedList(
                        inc_train_set[joined_intents_l], shuffle_between_epoch=False
                    ),  # True
                    "size": len(inc_train_set[joined_intents_l]),
                }
                for joined_intents_l in inc_train_set
            ]

            ## Dev
            inc_dev_set = {str(intents_l): [] for intents_l in inc_intents_set}
            for joined_intents_l in inc_dev_set.keys():
                for j, intent in enumerate(ast.literal_eval(joined_intents_l)):
                    if self.args.multi_head_out:
                        for eg in ordered_dev[intent]:
                            eg[2] = j
                            inc_dev_set[joined_intents_l].append(eg)
                    else:
                        inc_dev_set[joined_intents_l].extend(ordered_dev[intent])

            self.dev_stream[lang] = [
                {
                    "class_list": list(
                        map(
                            lambda x: self.class_types[x],
                            ast.literal_eval(joined_intents_l),
                        )
                    ),
                    "lang": lang,
                    "examples": AugmentedList(inc_dev_set[joined_intents_l]),
                    "size": len(inc_dev_set[joined_intents_l]),
                }
                for joined_intents_l in inc_dev_set
            ]

            ## Test
            self.test_stream[lang] = {
                "subtask_" + str(i): {} for i in range(len(inc_intents_set))
            }
            for i, subtask in enumerate(self.test_stream[lang].keys()):
                int_task_test = []
                for j, intent in enumerate(inc_intents_set[i]):
                    if self.multi_head_out:
                        for eg in ordered_test[intent]:
                            eg[2] = j
                            int_task_test.append(eg)
                    else:
                        int_task_test.extend(ordered_test[intent])

                self.test_stream[lang][subtask] = {
                    "class_list": list(
                        map(lambda x: self.class_types[x], inc_intents_set[i])
                    ),
                    "lang": lang,
                    "examples": AugmentedList(int_task_test),
                    "size": len(int_task_test),
                }

    def set_multi_incr_cll_streams(self):
        """
        Setup 6: A weaker version of Multi-task/Joint Learning where we gradually fine-tune on the incremental
        of multi-task at each language independently (when testing on language L we incrementally train up to that
        language)
        """
        ordered_langs = self.partition_per_lang(self.train_set)

        ## Train
        inc_langs_set = []
        for i, lang in enumerate(ordered_langs):
            inc_langs_set.append(ordered_langs[0 : i + 1])

        inc_train_set = {"-".join(lang_l): [] for lang_l in inc_langs_set}
        for lang_l in inc_train_set.keys():
            for lang in lang_l.split("-"):
                inc_train_set[lang_l].extend(self.train_set[lang])

        self.train_stream = [
            {
                "lang": lang_l,
                "examples": AugmentedList(
                    inc_train_set[lang_l], shuffle_between_epoch=False
                ),  # True
                "size": len(inc_train_set[lang_l]),
            }
            for lang_l in inc_train_set
        ]

        ## Dev
        inc_dev_set = {"-".join(lang_l): [] for lang_l in inc_langs_set}
        for lang_l in inc_dev_set.keys():
            for lang in lang_l.split("-"):
                inc_dev_set[lang_l].extend(self.dev_set[lang])

        self.dev_stream = [
            {
                "lang": lang_l,
                "examples": AugmentedList(
                    inc_dev_set[lang_l], shuffle_between_epoch=False
                ),  # True
                "size": len(inc_dev_set[lang_l]),
            }
            for lang_l in inc_dev_set
        ]

        ## Test
        self.test_stream = {}
        for lang in self.args.languages:
            self.test_stream.update(
                {
                    lang: {
                        "examples": AugmentedList(self.test_set[lang]),
                        "size": len(self.test_set[lang]),
                    }
                }
            )

    def set_cll_er_kd_streams_BACKUP(self):
        """
        BACKUP Sanity Check for memory issues in ER/KD
        """
        ordered_langs = self.partition_per_lang(self.train_set)
        ## Train
        mem_langs_set = []
        for i, lang in enumerate(ordered_langs):
            mem_langs_set.append(ordered_langs[0 : i + 1])

        print("self.args.max_mem_sz:", self.args.max_mem_sz)

        mem_train_set = {"-".join(lang_l): [] for lang_l in mem_langs_set}
        for i, lang_l in enumerate(mem_train_set.keys()):
            mem_langs = lang_l.split("-")[:i]
            print("--i => ", i, " mem_langs:", mem_langs)
            for lang in mem_langs:
                mem_train_set[lang_l].append(
                    self.train_set[lang][
                        : self.args.max_mem_sz // len(self.args.languages)
                    ]
                )  # TODO Add portion of memory here => added each task alone

        self.train_stream = [
            {
                "lang": lang_l[i],
                "examples": AugmentedList(
                    self.train_set[lang_l[i]], shuffle_between_epoch=False
                ),  # True
                "size": len(self.train_set[lang_l[i]]),  # main size
                "memory": [
                    AugmentedList(mem) for mem in mem_train_set["-".join(lang_l)]
                ],
                "size_memory": [
                    len(mem) for mem in mem_train_set["-".join(lang_l)]
                ],  # each task with its memory size but you don't need that anyways
            }
            for i, lang_l in enumerate(mem_langs_set)
        ]

        self.dev_stream = [
            {
                "lang": lang,
                "examples": AugmentedList(
                    self.dev_set[lang], shuffle_between_epoch=False
                ),  # True
                "size": len(self.dev_set[lang]),
            }
            for lang in ordered_langs
        ]

        ## Test
        self.test_stream = {}
        for lang in self.args.languages:
            self.test_stream.update(
                {
                    lang: {
                        "examples": AugmentedList(self.test_set[lang]),
                        "size": len(self.test_set[lang]),
                    }
                }
            )

    def set_cll_er_kd_streams(self):
        """
        BACKUP Sanity Check for memory issues in ER/KD
        """
        ordered_langs = self.partition_per_lang(self.train_set)
        clusters_save_dir = "/home1/mmhamdi/x-continuous-learning_new/outputs/k-means/"
        # TODO get saved idx for a specific language per k-means clustering
        if self.args.use_k_means:
            clusters = {}
            for lang in ordered_langs:
                with open(
                    os.path.join(
                        clusters_save_dir, "memory_language_" + lang + ".pickle"
                    ),
                    "rb",
                ) as file:
                    mem_rep = pickle.load(file)
                clusters.update({lang: [self.get_item_by_id(id_) for id_ in mem_rep]})

        ## Train
        mem_langs_set = []
        for i, lang in enumerate(ordered_langs):
            mem_langs_set.append(ordered_langs[0 : i + 1])

        mem_train_set = {"-".join(lang_l): [] for lang_l in mem_langs_set}

        for i, lang_l in enumerate(mem_train_set.keys()):
            mem_langs = lang_l.split("-")[:i]
            if self.args.use_k_means:
                for lang in mem_langs:
                    mem_train_set[lang_l].append(clusters[lang])
            else:
                if self.args.er_strategy == "equal-lang":  # XER(Rand) or mer-rand
                    for lang in mem_langs:
                        mem_train_set[lang_l].append(
                            self.train_set[lang][
                                : self.args.max_mem_sz
                                // len(
                                    mem_langs
                                )  # TODO could change that to // len(self.args.languages) to be K -> 2K -> 3K instead
                            ]
                        )
                else:  # MER(Rand) or er-rand
                    data_to_sample = []
                    for lang in mem_langs:
                        data_to_sample += self.train_set[lang]

                    if len(mem_langs) > 0:
                        random_idx = random.choices(
                            range(0, len(data_to_sample)), k=self.args.max_mem_sz
                        )  # TODO could change that to * len(data_to_sample) // len(self.args.languages) to be K -> 2K -> 3K instead

                        mem_train_set[lang_l] = [data_to_sample[k] for k in random_idx]
                    else:
                        mem_train_set[lang_l] = []

        self.train_stream = [
            {
                "lang": lang_l[i],
                "examples": AugmentedList(
                    self.train_set[lang_l[i]], shuffle_between_epoch=False
                ),  # True
                "size": len(self.train_set[lang_l[i]]),  # main size
            }
            for i, lang_l in enumerate(mem_langs_set)
        ]

        for i, lang_l in enumerate(mem_langs_set):
            if self.args.er_strategy == "equal-lang":  # MER(Rand)
                self.train_stream[i].update(
                    {
                        "memory": [
                            AugmentedList(mem)
                            for mem in mem_train_set["-".join(lang_l)]
                        ],
                        "size_memory": [
                            len(mem) for mem in mem_train_set["-".join(lang_l)]
                        ],  # each task with its memory size but you don't need that anyways
                    }
                )
            else:  # ER(Rand)
                self.train_stream[i].update(
                    {
                        "memory": [AugmentedList(mem_train_set["-".join(lang_l)])],
                        "size_memory": len(
                            mem_train_set["-".join(lang_l)]
                        ),  # each task with its memory size but you don't need that anyways
                    }
                )

        self.dev_stream = [
            {
                "lang": lang,
                "examples": AugmentedList(
                    self.dev_set[lang], shuffle_between_epoch=False
                ),  # True
                "size": len(self.dev_set[lang]),
            }
            for lang in ordered_langs
        ]

        ## Test
        self.test_stream = {}
        for lang in self.args.languages:
            self.test_stream.update(
                {
                    lang: {
                        "examples": AugmentedList(self.test_set[lang]),
                        "size": len(self.test_set[lang]),
                    }
                }
            )

    def set_multi_streams(self):
        """
        Setup 5: Multi-task/Joint Learning: train on all languages and intent classes at the same time
        """
        print("Creating multilingual stream")
        train_set_all = []
        for lang in self.train_set:
            train_set_all += self.train_set[lang]

        print("len(train_set_all):", len(train_set_all))
        self.train_stream = {
            "examples": AugmentedList(
                train_set_all, shuffle_between_epoch=False
            ),  # True
            "size": len(train_set_all),
        }

        dev_set_all = []
        for lang in self.args.languages:
            dev_set_all += self.dev_set[lang]

        print("len(dev_set_all):", len(dev_set_all))

        self.dev_stream = {
            "examples": AugmentedList(dev_set_all, shuffle_between_epoch=False),  # True
            "size": len(dev_set_all),
        }

        self.test_stream = {}
        for lang in self.args.languages:
            self.test_stream.update(
                {
                    lang: {
                        "examples": AugmentedList(self.test_set[lang]),
                        "size": len(self.test_set[lang]),
                    }
                }
            )

        print("test_stream:", self.test_stream)

    def get_item_by_id(self, id_):
        return self.process_egs_dict_global[id_]

    def partition_per_class(self, processed_egs, keys=None):
        class_dict = {_class: [] for _class in range(len(self.class_types))}
        for eg in processed_egs:
            class_dict[eg[2]].append(eg)

        if keys:
            return {k: class_dict[k] for k in keys}, keys

        if self.args.order_class == 2:
            keys = list(class_dict.keys())
            print(keys)
            random.shuffle(keys)
        else:
            # if len(self.order_lst) == 0 or "en" in self.order_lst:
            reverse_flag = False
            if self.args.order_class == 0:
                reverse_flag = True

            keys = sorted(
                class_dict, key=lambda k: len(class_dict[k]), reverse=reverse_flag
            )

        ordered_dict = {k: class_dict[k] for k in keys}
        return ordered_dict, keys

    def set_intent_to_other(self, processed_egs):
        other_intents = []
        for eg in processed_egs:
            other_eg = (eg[0], eg[1], "OTHER", eg[3], eg[4])
            other_intents.append(other_eg)

        return other_intents

    def partition_per_lang(self, train_set):
        if (
            len("".join(self.order_lst)) == 0
        ):  # only if order_lst is not specified then use some order category: in args.order_lang 0:high2low, 1:low2high, 2:random
            if self.args.order_lang == 2:  # random order
                ordered_langs = train_set.keys()
                random.shuffle(ordered_langs)
            else:
                reverse_flag = False
                if self.args.order_lang == 0:  # decreasing frequency
                    reverse_flag = True

                ordered_langs = sorted(
                    train_set,
                    key=lambda lang: len(train_set[lang]),
                    reverse=reverse_flag,
                )
        else:
            ordered_langs = self.order_lst

        print("ordered_langs:", ordered_langs)
        return ordered_langs

    def get_batch_one(self, dataset, identifier, identifiers):
        return self.processor.next_batch(
            dataset=dataset,
            batch_size=1,
            data_split=None,
            identifier=identifier,
            identifiers=identifiers,
        )

    def get_batch_for_examples(self, examples):
        return self.processor.get_batch_for_examples(examples)

    def next_batch(
        self, dataset, batch_size, data_split=None, identifier=None, identifiers=None
    ):
        return self.processor.next_batch(
            dataset=dataset,
            batch_size=batch_size,
            data_split=data_split,
            identifier=identifier,
            identifiers=identifiers,
        )
