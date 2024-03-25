import json
import pickle
import os
from collections import Counter

import sys
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
from src.data_utils import MultiPurposeDataset, AugmentedList
from src.transformers_config import MODELS_dict
from src.consts import DOMAIN_TYPES, INTENT_TYPES, SLOT_TYPES

er_strategy = "easy"

# /project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/BertBaseMultilingualCased/
# MTOP/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/
# NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-random/_WIPEStrategy-random/ER_MaxSize-10105/
# ERSched-er-only/Mode-cont-mono/en_de_hi_th/lt_scheduler-idmovements/lt_scheduler_idmovements_trainon-0_epoch-9.json


class DataArgs:
    def __init__(self):
        self.data_root = "/project/jonmay_231/meryem/Datasets/"
        self.task_name = "tod"
        self.data_name = "mtop"
        self.data_format = "txt"
        self.setup_opt = "cll-er_kd"
        self.order_lst = "en_de_hi_th"
        self.languages = ["en", "de", "hi", "th"]
        self.order_class = 0
        self.order_lang = 0
        self.trans_model = "BertBaseMultilingualCased"
        self.use_slots = True
        self.seed = 42
        self.num_class_tasks = 10
        self.num_lang_tasks = 2
        self.multi_head_out = False
        self.max_mem_sz = 100
        self.er_strategy = er_strategy
        self.order_class = 0
        self.use_k_means = False

        self.data_root = os.path.join(
            self.data_root,
            self.task_name.upper(),
            self.data_name.upper(),
        )


args = DataArgs()
print(args.task_name)
print(args.setup_opt)

tokenizer = MODELS_dict[args.trans_model][1].from_pretrained(
    MODELS_dict[args.trans_model][0], do_lower_case=True, do_basic_tokenize=False
)

dataset = MultiPurposeDataset(args, tokenizer)

# Load the results
LANG_ORDER = "en_de_hi_th"  # "de_en_th_hi"  # "hi_th_en_de"  # "th_hi_de_en"  #
MEM_SIZE = "10105"

ROOT_DATA_PATH = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/"
    + "BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/DemotePrevious/"
    + "LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
    + "NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
)


def print_id(id_):
    print(
        "id: ",
        dataset.get_item_by_id(id_).unique_id,
        " text: ",
        dataset.get_item_by_id(id_).text,
        " label: ",
        INTENT_TYPES["mtop"][dataset.get_item_by_id(id_).label],
        " slot labels: ",
        [
            SLOT_TYPES["mtop"][slot_id]
            for slot_id in dataset.get_item_by_id(id_).slot_labels
        ],
    )


def check_if_stay_at_4(value):
    flag = False
    for val in value:
        if val == 4:
            flag = True
        if flag:
            if val != 4:
                flag = False
                return flag
    return flag


def analyze_id_movements(k, ep):
    # k = 0, ep = 9
    print("Training on " + LANG_ORDER.split("_")[k])
    root_results_dir = (
        ROOT_DATA_PATH
        + er_strategy
        + "/_WIPEStrategy-random/ER_MaxSize-"
        + MEM_SIZE
        + "/ERSched-er-only"
        + "/Mode-cont-mono/"
        + LANG_ORDER
        + "/lt_scheduler-idmovements/"
        + "lt_scheduler_idmovements_trainon-"
        # + "er_lt_scheduler_0idmovements_trainon-"
        + str(k)
        + "_epoch-"
        + str(ep)
        + ".json"
    )

    # print("root_results_dir: ", root_results_dir)
    with open(root_results_dir) as file:
        lt_data = json.load(file)

    ids_keys = list(lt_data.keys())
    values = list(lt_data.values())

    freq_0 = [Counter(value)[0] / len(value) for value in values]
    freq_1 = [Counter(value)[1] / len(value) for value in values]
    freq_2 = [Counter(value)[2] / len(value) for value in values]
    freq_3 = [Counter(value)[3] / len(value) for value in values]
    freq_4 = [Counter(value)[4] / len(value) for value in values]

    max_0 = 0
    max_1 = 0
    max_2 = 0
    max_3 = 0
    max_4 = 0

    stayed_at_0 = []
    stayed_at_4 = []
    for i in range(len(freq_0)):
        if freq_0[i] > max_0:
            max_0 = freq_0[i]
        if freq_1[i] > max_1:
            max_1 = freq_1[i]
        if freq_2[i] > max_2:
            max_2 = freq_2[i]
        if freq_3[i] > max_3:
            max_3 = freq_3[i]
        if freq_4[i] > max_4:
            max_4 = freq_4[i]

        if freq_0[i] == 1.0:
            # print(
            #     " Hard Example: ({}, {}, {}, {}, {}, {}) ".format(
            #         ids_keys[i], freq_0[i], freq_1[i], freq_2[i], freq_3[i], freq_4[i]
            #     )
            # )
            stayed_at_0.append(ids_keys[i])
        if check_if_stay_at_4(values[i]):
            stayed_at_4.append(ids_keys[i])

    # print(
    #     "max_0: ",
    #     max_0,
    #     " max_1:",
    #     max_1,
    #     " max_2: ",
    #     max_2,
    #     " max_3:",
    #     max_3,
    #     " max_4:",
    #     max_4,
    # )

    # for p, id_ in enumerate(stayed_at_0):
    #     print(
    #         [
    #             SLOT_TYPES["mtop"][slot_id]
    #             for slot_id in dataset.get_item_by_id(id_).slot_labels
    #             if SLOT_TYPES["mtop"][slot_id] not in ["X"]
    #         ]
    #     )
    # Return in_between
    in_between = [
        id_ for id_ in ids_keys if id_ not in stayed_at_0 and id_ not in stayed_at_4
    ]
    print(
        "len(stayed_at_0):",
        len(stayed_at_0),
        " len(stayed_at_4):",
        len(stayed_at_4),
        " len(in_between):",
        len(in_between),
    )
    return in_between  # in_between


def analyze_bins(k, ep):
    # k = 0, ep = 9
    print("Training on " + LANG_ORDER.split("_")[k])
    root_results_dir = (
        ROOT_DATA_PATH
        + er_strategy
        + "/_WIPEStrategy-random/ER_MaxSize-"
        + MEM_SIZE
        + "/ERSched-er-only"
        + "/Mode-cont-mono/"
        + LANG_ORDER
        + "/lt_scheduler-decks/"
        # + "er_lt_scheduler_0decks_trainon-"
        + "lt_scheduler_decks_trainon-"
        + str(k)
        + "_epoch-"
        + str(ep)
        + ".json"
    )

    with open(root_results_dir) as file:
        lt_data = json.load(file)

    print(
        "k: ",
        k,
        " ep: ",
        ep,
        " lt_data: 0: ",
        len(lt_data[0]),
        " 1: ",
        len(lt_data[1]),
        " 2: ",
        len(lt_data[2]),
        " 3:",
        len(lt_data[3]),
        " 4:",
        len(lt_data[4]),
    )

    print("lt_data[0][:10]: ", lt_data[0][:10])
    print("lt_data[4][:10]: ", lt_data[4][:10])

    # print("----Easy examples >>> ")
    # for p, id_ in enumerate(lt_data[4][:20]):
    #     # print(p)
    #     # print(id_)
    #     # print(dataset.get_item_by_id(id_).text)
    #     # print(INTENT_TYPES["mtop"][dataset.get_item_by_id(id_).label])
    #     print(
    #         [
    #             SLOT_TYPES["mtop"][slot_id]
    #             for slot_id in dataset.get_item_by_id(id_).slot_labels
    #             if SLOT_TYPES["mtop"][slot_id] not in ["X"]
    #         ]
    #     )
    #     # print_id(id_)

    # print("----Hard examples >>> ")
    # for id_ in lt_data[0][:20]:
    #     # print(id_)
    #     # print(dataset.get_item_by_id(id_).text)
    #     # print(INTENT_TYPES["mtop"][dataset.get_item_by_id(id_).label])
    #     print(
    #         [
    #             SLOT_TYPES["mtop"][slot_id]
    #             for slot_id in dataset.get_item_by_id(id_).slot_labels
    #             if SLOT_TYPES["mtop"][slot_id] not in ["X"]
    #         ]
    #     )
    #     # print_id(id_)

    easy_labels = list(
        set(
            [
                INTENT_TYPES["mtop"][dataset.get_item_by_id(id_).label]
                for id_ in lt_data[4]
            ]
        )
    )
    hard_labels = list(
        set(
            [
                INTENT_TYPES["mtop"][dataset.get_item_by_id(id_).label]
                for id_ in lt_data[0]
            ]
        )
    )
    # for label in hard_labels:
    #     print(label)

    for label in easy_labels:
        if label not in hard_labels:
            print(label)
    # print(
    #     "--Easy intent labels: ",
    #     easy_labels,
    #     "--Hard intent labels: ",
    #     hard_labels,
    # )


def analyze_train_predictions(k, ep):
    LANG_ORDER = "en_de_hi_th"

    root_results_dir = (
        ROOT_DATA_PATH
        + er_strategy
        + "/_WIPEStrategy-random/ER_MaxSize-"
        + MEM_SIZE
        + "/ERSched-er-only"
        + "/Mode-cont-mono/"
        + LANG_ORDER
        + "/predictions/"
        + "train_on-"
        + str(k)
        + "_aftertrainon-"
        + str(k)
        + "_epoch-"
        + str(ep)
    )
    print(
        "Training on " + LANG_ORDER.split("_")[k],
        " root_results_dir:",
        root_results_dir,
    )

    with open(root_results_dir, "r") as file:
        predictions = file.read().splitlines()

    pred_dict = {}
    true_dict = {}
    pred_slot_dict = {}
    true_slot_dict = {}
    in_between = analyze_id_movements(0, 9)
    # with open(
    #     "/home1/mmhamdi/x-continuous-learning_new/humanlearn/data_analysis/stayed_at_0_ids.txt",
    #     "r",
    # ) as file:
    #     stayed_at_0 = file.read().splitlines()

    print("analyze_train_predictions: ========>, ", in_between[:10])
    pred_true_classes = {}
    count_wrong = 0
    count_wrong_all = 0
    count_all = 0
    for pred in predictions:
        id_, text, true_class, pred_class, true_slots, pred_slots = pred.split("\t")
        if id_ in in_between:
            if pred_class != true_class:
                if true_class not in pred_true_classes:
                    pred_true_classes.update({true_class: [pred_class]})
                else:
                    pred_true_classes[true_class].append(pred_class)
                count_wrong += 1
            if pred_class != true_class or pred_slots != true_slots:
                print(
                    id_,
                    text,
                    true_class,
                    " || ",
                    pred_class,
                    true_slots,
                    " || ",
                    pred_slots,
                )
                count_wrong_all += 1
            count_all += 1
            # print(id_, true_class, " || ", pred_class, true_slots, " || ", pred_slots)
            # print(true_slots, " || ", pred_slots)

    print("Count of wrongly labelled items per intent:", count_wrong / count_all)
    print(
        "Count of wrongly labelled items per intent and slot:",
        count_wrong_all / count_all,
    )
    # for id_ in stayed_at_0_ids:
    #     print(id_, pred_dict[id_])

    # for true_class in pred_true_classes:
    #     print(true_class, pred_true_classes[true_class])


def analyze_languages(k, ep):
    print(
        "Training on " + LANG_ORDER.split("_")[k],
        # " root_results_dir:",
        # root_results_dir,
    )
    if er_strategy == "easy":
        wipe_strategy = "hard"
    elif er_strategy == "hard":
        wipe_strategy = "easy"
    else:
        wipe_strategy = "random"
    root_results_dir = (
        ROOT_DATA_PATH
        + er_strategy
        + "/_WIPEStrategy-"
        + wipe_strategy
        + "/ER_MaxSize-"
        + MEM_SIZE
        + "/WIPE_NEW/ERSched-er-only"
        + "/Mode-cont-mono/"
        + LANG_ORDER
        + "/lt_scheduler-decks/"
        # + "er_lt_scheduler_0decks_trainon-"
        + "lt_scheduler_decks_trainon-"
        + str(k)
        + "_epoch-"
        + str(ep)
        + ".json"
    )

    with open(root_results_dir) as file:
        lt_data = json.load(file)

    total = 0
    for o in range(5):
        lang_dist = {lang: 0 for lang in args.languages}
        for id_ in lt_data[o]:
            split, lang, i = id_.split("_")
            lang_dist[lang] += 1
            total += 1

        print("Deck # ", o, " => lang_dist:", lang_dist)

    # total = 0
    # lang_dist = {lang: 0 for lang in args.languages}
    # for o in range(5):
    #     for id_ in lt_data[o]:
    #         split, lang, i = id_.split("_")
    #         lang_dist[lang] += 1
    #         total += 1

    print("Deck # ", o, " => lang_dist:", lang_dist, " TOTAL: ", total)


# analyze_bins(0, 9)
print("LANG_ORDER: ", LANG_ORDER)
analyze_train_predictions(0, 9)
exit(0)
analyze_id_movements(0, 9)
analyze_id_movements(1, 9)
analyze_id_movements(2, 9)
analyze_id_movements(3, 9)

# After one epoch of training for each phase
print("er_strategy: ", er_strategy)
# print("After one epoch of training for each phase ")
# analyze_languages(1, 1)
# analyze_languages(2, 1)
# analyze_languages(3, 1)
# At the end of training for each phase
print("At the end of training for each phase ")
analyze_languages(0, 9)
analyze_languages(1, 9)
analyze_languages(2, 9)
analyze_languages(3, 9)
