import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sys import platform
import sys
from summarize_metrics import (
    acc_avg,
    fwt_avg,
    fwt_avg_mono,
    bwt_avg,
    forget_avg,
    final_perf,
)

if platform == "darwin":
    sys.path.append(
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new"
    )
    location = "LOCAL"
else:
    sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
    location = "CARC"
from src.consts import LANGUAGE_ORDERS, LANGUAGES, METRICS_NAMES

# LANGUAGE_ORDERS = {
#     "MTOP": [
#         "en_de_hi_th",
#         "th_hi_de_en",
#         "hi_th_en_de",
#         "de_en_th_hi",
#     ],
#     "PANX": [
#         "ru_id_te_sw",
#         "sw_te_id_ru",
#         "te_sw_ru_id",
#         "id_ru_sw_te",
#     ],
# }
# LANGUAGES = {"MTOP": ["en", "de", "hi", "th"], "PANX": ["ru", "id", "te", "sw"]}
EVAL_TYPE = "test"
TASK_NAME = "MTOP"
ABLATION_MODE = "wiped"  # "convergence" "er-strategies" "er-techniques"

if TASK_NAME == "MTOP":
    MEM_SIZE = "10105"
elif TASK_NAME in ["TYDIQA", "MULTIATIS"]:
    MEM_SIZE = "500"
else:  # PANX
    MEM_SIZE = "1000"  # "10105"  # "500"

ROOT_DIR = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"
    + TASK_NAME
    + "/HyperparamSearch/BertBaseMultilingualCased/"
)

ROOT_DIR_NEW = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/BertBaseMultilingualCased/"
    + TASK_NAME
)


# METRICS_NAMES = [
#     "Forgetting",
#     "Final Performance",
#     "Accuracy",
#     "Transfer",
#     "Zero-Shot Transfer",
# ]


def multiply_two_lists(test_list1, test_list2):
    return list(map(lambda x, y: x * y, test_list1, test_list2))


def compute_mono_perf():
    mono_perf = {TASK_NAME: {}}
    if TASK_NAME != "MTOP":
        # TODO Change this run mono experiments
        mono_perf[TASK_NAME] = {lang: (0, 0) for lang in LANGUAGES[TASK_NAME]}
    else:
        for lang in LANGUAGES[TASK_NAME]:
            with open(
                "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/BertBaseMultilingualCased/"
                + TASK_NAME
                + "/Baseline/Mode-cont-mono/"
                + lang
                + "/metrics.json"
            ) as file:
                data = json.load(file)

            best_valid_perf = multiply_two_lists(
                data[0]["valid"]["class"]["0"],
                data[0]["valid"]["tags"]["0"],
            )
            ep_idx_best_val = best_valid_perf.index(max(best_valid_perf))
            test_acc = data[0]["test"]["class"][lang][ep_idx_best_val] * 100
            test_tags = data[0]["test"]["tags"][lang][ep_idx_best_val] * 100

            mono_perf[TASK_NAME].update({lang: (test_acc, test_tags)})

    return mono_perf


mono_perf = compute_mono_perf()
rand_perf = {
    "MTOP": {
        "en": (0.1823985408116735, 0.1719692790338232),
        "de": (1.4933784164553396, 0.15755581155353093),
        "hi": (0.17927572606669057, 0.16128618931729896),
        "th": (0.6871609403254972, 0.13090873843779943),
    },
    "PANX": {
        "id": (0, 0),
        "ru": (0, 0),
        "sw": (0, 0),
        "te": (0, 0),
    },
    "TYDIQA": {
        "id": (0, 0),
        "ru": (0, 0),
        "sw": (0, 0),
        "te": (0, 0),
    },
    "MULTIATIS": {
        "en": (0, 0),
        "fr": (0, 0),
        "tr": (0, 0),
        "zh": (0, 0),
    },
}

DIR_LIT = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/x-continuous-learn/BertBaseMultilingualCased/"
    + TASK_NAME
)


def get_paths(model_name, LANG_ORDER):
    if model_name in [
        "ltn-er-only",
        "ltn-er-both",
        "ltn-er-main",
    ]:  # New Results
        er_technique = model_name.split("-")[-1]
        root_results_dir = (
            ROOT_DIR_NEW
            + "/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-fifo/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-easy"
            + "/ERSched-er-"
            + er_technique
            + "/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name == "er-equal":
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/MTOP/Baseline/ER_prop-0.0_TrainERMain/"
            + "_ERStrategy-equal-lang/ER_MaxSize-"
            + MEM_SIZE
            + "/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"
    elif model_name == "er-rand":
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/MTOP/Baseline/ER_prop-0.0_TrainERMain/"
            + "_ERStrategy-random/ER_MaxSize-"
            + MEM_SIZE
            + "/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name in [
        "ltn-er-easy",
        "ltn-er-hard",
        "ltn-er-random",
    ]:
        er_strategy = model_name.split("-")[2]
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-fifo/FreqSample-epoch/FreqUpdate-epoch-everything/"
            + "NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/ER_MaxSize-"
            + MEM_SIZE
            + "/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name in [
        "ltn-er-easy-rand",
        "ltn-er-hard-rand",
    ]:
        er_strategy = model_name.split("-")[2]
        # root_results_dir = (
        #     "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
        #     + "x-continuous-learn/BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/"
        #     + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
        #     + "NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
        #     + er_strategy
        #     + "/ER_MaxSize-"
        #     + MEM_SIZE
        #     + "/ERSched-er-only/Mode-cont-mono/"
        #     + LANG_ORDER
        #     + "/metrics.json"
        # )
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/_WIPEStrategy-random/ER_MaxSize-"
            + MEM_SIZE
            + "/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name in [
        "ltn-er-rand",
    ]:
        er_strategy = "random"
        # root_results_dir = (
        #     "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
        #     + "x-continuous-learn/BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/"
        #     + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
        #     + "NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
        #     + er_strategy
        #     + "/ER_MaxSize-"
        #     + MEM_SIZE
        #     + "/ERSched-er-only/Mode-cont-mono/"
        #     + LANG_ORDER
        #     + "/metrics.json"
        # )
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/_WIPEStrategy-random/ER_MaxSize-"
            + MEM_SIZE
            + "/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"
    elif model_name in [
        "ltn-er-rand-prop",
    ]:
        er_strategy = "random"
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-rand-prop/FreqSample-epoch/FreqUpdate-epoch-everything/"
            + "NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/ER_MaxSize-"
            + MEM_SIZE
            + "/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name in [
        "ltn-er-easy-wiped",
        "ltn-er-hard-wiped",
        "ltn-er-random-wiped",
    ]:
        er_strategy = model_name.split("-")[2]
        if er_strategy == "easy":
            wipe_strategy = "hard"
            # wipe_strategy = "random"
        elif er_strategy == "hard":
            wipe_strategy = "easy"
            # wipe_strategy = "random"
        else:
            wipe_strategy = "random"

        # root_results_dir = (
        #     DIR_LIT
        #     + "/Ltn/LtnModel-ltn/"
        #     + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
        #     + "NumDecks-5"
        #     + "/ER_prop-0.0_TrainERMain/_ERStrategy-"
        #     + er_strategy
        #     + "/ER_MaxSize-"
        #     + MEM_SIZE
        #     + "/WIPE/ERSched-er-only/Mode-cont-mono/"
        #     + LANG_ORDER
        #     + "/metrics.json"
        # )

        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/_WIPEStrategy-"
            + wipe_strategy
            + "/ER_MaxSize-"
            + MEM_SIZE
            + "/WIPE_NEW/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name == "vanilla":  # Old Results "vanilla"
        if TASK_NAME == "TYDIQA":
            root_results_dir = (
                "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/x-continuous-learn/"
                + "BertBaseMultilingualCased/"
                + TASK_NAME
                + "/Baseline/Mode-cont-mono/"
                + LANG_ORDER
                + "/"
                + "metrics.json"
            )
            acc_alias = "class"
            slots_alias = "tags"
        else:
            root_results_dir = (
                ROOT_DIR
                + "Baseline/ltmode-cont-mono/"
                + LANG_ORDER
                + "/W_SCHEDULER_0_WARMUP/"
                + "metrics.json"
            )
            acc_alias = "acc"
            slots_alias = "slots"

    return root_results_dir, acc_alias, slots_alias


def get_mean_tuple(dict_, LANGS, list=True):
    if list:
        per_lang = {}
        for lang in LANGS:
            per_lang.update(
                {
                    lang: (
                        list(np.mean([dict_[k][0][lang] for k in dict_], axis=0)),
                        list(np.mean([dict_[k][1][lang] for k in dict_], axis=0)),
                    )
                }
            )

    else:
        per_lang = {}
        for lang in LANGS:
            per_lang.update(
                {
                    lang: (
                        np.mean([dict_[k][0][lang] for k in dict_], axis=0),
                        np.mean([dict_[k][1][lang] for k in dict_], axis=0),
                    )
                }
            )

    return per_lang


def compute_metrics(model_name):
    avg_forgetting_epochs = {
        LANG_ORDER: ([], []) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_final_perf_epochs = {
        LANG_ORDER: ([], []) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_acc_perf_epochs = {
        LANG_ORDER: ([], []) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_fwt_perf_epochs = {
        LANG_ORDER: ([], []) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_fwt_mono_perf_epochs = {
        LANG_ORDER: ([], []) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }

    for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]:
        root_results_dir, acc_alias, slots_alias = get_paths(model_name, LANG_ORDER)
        with open(root_results_dir) as file:
            results_json = json.load(file)
        LANGS = LANG_ORDER.split("_")
        for epoch in range(10):
            metrics = [[0.0 for _ in range(14)] for _ in range(7)]

            for i, _ in enumerate(LANGS):  # training
                for j, test_lang in enumerate(LANGS):  # testing
                    metrics[i][2 * j] = (
                        results_json[i]["test"][acc_alias][test_lang][epoch] * 100
                    )
                    metrics[i][2 * j + 1] = (
                        results_json[i]["test"][slots_alias][test_lang][epoch] * 100
                    )

            metrics_np = np.asarray(metrics)

            avg_forget = forget_avg(metrics_np, LANGS, LANGS)
            avg_forgetting_epochs[LANG_ORDER][0].append(avg_forget[4])
            avg_forgetting_epochs[LANG_ORDER][1].append(avg_forget[5])
            #
            avg_fp = final_perf(metrics_np, LANGS)
            avg_final_perf_epochs[LANG_ORDER][0].append(avg_fp[0])
            avg_final_perf_epochs[LANG_ORDER][1].append(avg_fp[1])
            #
            avg_acc = acc_avg(metrics_np, LANGS, LANGS)
            avg_acc_perf_epochs[LANG_ORDER][0].append(avg_acc[2])
            avg_acc_perf_epochs[LANG_ORDER][1].append(avg_acc[3])

            #
            if "Transfer" in METRICS_NAMES:
                avg_fwt_mono = fwt_avg_mono(metrics_np, LANGS, mono_perf[TASK_NAME])
                avg_fwt_mono_perf_epochs[LANG_ORDER][0].append(avg_fwt_mono[2])
                avg_fwt_mono_perf_epochs[LANG_ORDER][1].append(avg_fwt_mono[3])

            #
            if "Zero-Shot Transfer" in METRICS_NAMES:
                avg_fwt = fwt_avg(metrics_np, LANGS, rand_perf[TASK_NAME])
                avg_fwt_perf_epochs[LANG_ORDER][0].append(avg_fwt[2])
                avg_fwt_perf_epochs[LANG_ORDER][1].append(avg_fwt[3])

    return (
        get_mean_tuple(avg_forgetting_epochs, LANGS),
        get_mean_tuple(avg_final_perf_epochs, LANGS),
        get_mean_tuple(avg_acc_perf_epochs, LANGS),
        get_mean_tuple(avg_fwt_mono_perf_epochs, LANGS),
        get_mean_tuple(avg_fwt_perf_epochs, LANGS),
    )


def compute_metrics_best_dev(model_name):
    print("model_name:", model_name)
    avg_forgetting = {
        LANG_ORDER: (0.0, 0.0) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_final_perf = {
        LANG_ORDER: (0.0, 0.0) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_acc_perf = {LANG_ORDER: (0.0, 0.0) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]}
    avg_fwt_mono_perf = {
        LANG_ORDER: (0.0, 0.0) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]
    }
    avg_fwt_perf = {LANG_ORDER: (0.0, 0.0) for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]}

    if model_name == "vanilla":  # MTOP stuff
        if TASK_NAME == "MTOP":
            alias = "val"
        else:
            alias = "valid"
    else:
        alias = "valid"
    for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]:
        metrics = [[0.0 for _ in range(14)] for _ in range(7)]

        root_results_dir, acc_alias, slots_alias = get_paths(model_name, LANG_ORDER)
        with open(root_results_dir) as file:
            results_json = json.load(file)
        LANGS = LANG_ORDER.split("_")

        for i, _ in enumerate(LANGS):  # training
            # print(
            #     "results_json[i][val][acc_alias]: ",
            #     results_json[i]["val"][acc_alias],
            #     " results_json[i][val][slots_alias]: ",
            #     results_json[i]["val"][slots_alias],
            # )
            if model_name == "vanilla" and TASK_NAME == "MTOP":
                best_valid_perf = multiply_two_lists(
                    results_json[i][alias][acc_alias],
                    results_json[i][alias][slots_alias],
                )
            else:
                best_valid_perf = multiply_two_lists(
                    results_json[i][alias][acc_alias][str(i)],
                    results_json[i][alias][slots_alias][str(i)],
                )
            ep_idx_best_val = best_valid_perf.index(max(best_valid_perf))
            for j, test_lang in enumerate(LANGS):  # testing
                metrics[i][2 * j] = (
                    results_json[i]["test"][acc_alias][test_lang][ep_idx_best_val] * 100
                )
                metrics[i][2 * j + 1] = (
                    results_json[i]["test"][slots_alias][test_lang][ep_idx_best_val]
                    * 100
                )

        metrics_np = np.asarray(metrics)

        avg_forget = forget_avg(metrics_np, LANGS, LANGS)
        avg_forgetting[LANG_ORDER] = (avg_forget[0], avg_forget[1])
        # avg_forgetting[LANG_ORDER] = (avg_forget[2], avg_forget[3])
        #
        avg_fp = final_perf(metrics_np, LANGS)
        avg_final_perf[LANG_ORDER] = (avg_fp[4], avg_fp[5])
        #
        avg_acc = acc_avg(metrics_np, LANGS, LANGS)
        avg_acc_perf[LANG_ORDER] = (avg_acc[0], avg_acc[1])

        #
        if "Transfer" in METRICS_NAMES:
            avg_fwt_mono = fwt_avg_mono(metrics_np, LANGS, mono_perf[TASK_NAME])
            avg_fwt_mono_perf[LANG_ORDER] = (avg_fwt_mono[4], avg_fwt_mono[5])
        else:
            avg_fwt_mono_perf[LANG_ORDER] = (0.0, 0.0)

        #
        if "Zero-Shot Transfer" in METRICS_NAMES:
            avg_fwt = fwt_avg(metrics_np, LANGS, rand_perf[TASK_NAME])
            avg_fwt_perf[LANG_ORDER] = (avg_fwt[4], avg_fwt[5])
        else:
            avg_fwt_perf[LANG_ORDER] = (0.0, 0.0)

    return (
        get_mean_tuple(avg_forgetting, LANGS, list=False),
        get_mean_tuple(avg_final_perf, LANGS, list=False),
        get_mean_tuple(avg_acc_perf, LANGS, list=False),
        get_mean_tuple(avg_fwt_mono_perf, LANGS, list=False),
        get_mean_tuple(avg_fwt_perf, LANGS, list=False),
    )


if ABLATION_MODE == "convergence":
    SELECTED_MODELS = [
        "vanilla",
        # "er-equal",
        # "er-rand",
        # "ltn-er-easy",
        "ltn-er-easy-rand",
        "ltn-er-hard-rand",
        # "ltn-er-random",
        "ltn-er-rand",
        # "ltn-er-rand-prop",
    ]
    EVAL_FUNC = compute_metrics_best_dev
elif ABLATION_MODE == "er-strategies":
    SELECTED_MODELS = [
        "ltn-er-easy",
        "ltn-er-hard",
        "ltn-er-extreme",
    ]
    EVAL_FUNC = compute_metrics_best_dev
elif ABLATION_MODE == "er-techniques":
    SELECTED_MODELS = [
        "ltn-er-only",
        "ltn-er-main",
        "ltn-er-both",
    ]
    EVAL_FUNC = compute_metrics_best_dev

elif ABLATION_MODE == "wiped":
    SELECTED_MODELS = [
        "ltn-er-easy-wiped",
        # "ltn-er-easy",
        # "ltn-er-hard",
        "ltn-er-hard-wiped",
        "ltn-er-random-wiped",
    ]
    EVAL_FUNC = compute_metrics_best_dev

plot_save_dir = (
    "/home1/mmhamdi/x-continuous-learning_new/outputs/Plots/spacedrepetition/Test/"
    + TASK_NAME
    + "/"
    + ABLATION_MODE
    + "/ALL_LANG_ORDERS/"
)
if not os.path.isdir(plot_save_dir):
    os.makedirs(plot_save_dir)

if EVAL_FUNC == compute_metrics:
    metrics_epochs_models = {metric_name: ({}, {}) for metric_name in METRICS_NAMES}
    all_metrics_epochs_models = ({}, {})

    for model_name in SELECTED_MODELS:
        outputs = EVAL_FUNC(model_name)
        for j, metric_name in enumerate(METRICS_NAMES):
            for i in range(2):
                metrics_epochs_models[metric_name][i].update(
                    {model_name: outputs[j][i]}
                )

    for i in range(2):
        for j, metric_name in enumerate(METRICS_NAMES):
            for model_name in SELECTED_MODELS:
                all_metrics_epochs_models[i].update(
                    {
                        metric_name
                        + "_"
                        + model_name: metrics_epochs_models[metric_name][i][model_name]
                    }
                )

    # print("INTENTS:", all_metrics_epochs_models[0])
    print("*********************************************")
    print("SLOTS:", all_metrics_epochs_models[1])
    if TASK_NAME == "MTOP":
        df_intents = pd.DataFrame(all_metrics_epochs_models[0])
        df_slots = pd.DataFrame(all_metrics_epochs_models[1])

        df_intents.to_csv(plot_save_dir + "Intents.csv")
        df_slots.to_csv(plot_save_dir + "Slots.csv")
        i_range = [0, 1]
    else:  # PANX
        df_tags = pd.DataFrame(all_metrics_epochs_models[1])
        df_tags.to_csv("Tags_average_lang_orders.csv")
        i_range = [1]
else:
    metrics_models = {metric_name: ({}, {}) for metric_name in METRICS_NAMES}

    intents_df = {"metric": METRICS_NAMES}
    intents_df.update({model_name: [] for model_name in SELECTED_MODELS})
    slots_df = {"metric": METRICS_NAMES}
    slots_df.update({model_name: [] for model_name in SELECTED_MODELS})
    for model_name in SELECTED_MODELS:
        outputs = EVAL_FUNC(model_name)
        print(model_name, outputs)
        # for j, metric_name in enumerate(METRICS_NAMES):
        #     intents_df[model_name].append(outputs[j][0])

        #     slots_df[model_name].append(outputs[j][1])
        #     for i in range(2):
        #         metrics_models[metric_name][i].update({model_name: outputs[j][i]})

    exit(0)

    for j, metric_name in enumerate(METRICS_NAMES):
        for i in range(2):
            if i == 0:
                task = "intents"
                y_label = "Accuracy"
            else:
                task = "slots"
                y_label = "F1 Score"

            df = pd.DataFrame()
            x_list = []
            y_list = []
            cat_list = []
            print(">>> metric_name: ", metric_name, " task: ", task)
            for model_name in metrics_models[metric_name][i]:
                x_list.append(metric_name)
                y_list.append(metrics_models[metric_name][i][model_name])
                cat_list.append(model_name)
                print(model_name, metrics_models[metric_name][i][model_name])
            df["x"] = x_list
            df["y"] = y_list
            df["category"] = cat_list

            # Bar Plots
            ax = sns.barplot(x="x", y="y", hue="category", data=df)
            # ax.bar_label(ax.containers[0], fmt="%.1f")
            plt.show()
            plt.savefig(plot_save_dir + y_label + "_" + metric_name + ".png")

    # print("intents_df:", intents_df)
    # print("slots_df:", slots_df)
    # sns.barplot(x="x", y="y", hue="category", data=intents_df)
    # plt.show()
    # plt.savefig(plot_save_dir + "intents.png")

    # #
    # sns.barplot(x="x", y="y", hue="category", data=slots_df)
    # plt.show()
    # plt.savefig(plot_save_dir + "slots.png")

    # Saving this into an excel sheet for each subtask separately
    if TASK_NAME == "MTOP":
        pd.DataFrame(intents_df).to_csv(plot_save_dir + "Intents" + ".csv")
        pd.DataFrame(slots_df).to_csv(plot_save_dir + "Slots.csv")
    else:  # PANX
        pd.DataFrame(slots_df).to_csv(plot_save_dir + "Tags.csv")

exit(0)

for i in i_range:
    output = ""
    for metric_name in METRICS_NAMES:
        for model_name in SELECTED_MODELS:
            output += (
                metric_name
                + " "
                + model_name
                + ":"
                + metrics_epochs_models[metric_name][i][model_name]
            )
    print(output)
    print("**********************************************************")

exit(0)

plot_save_dir = (
    "/home1/mmhamdi/x-continuous-learning_new/outputs/Plots/spacedrepetition/Test/"
    + TASK_NAME
    + "/cont_new/"
)

for metric_name in metrics_epochs_models:
    for i in range(2):
        df = pd.DataFrame(metrics_epochs_models[metric_name][i])

        plt.clf()
        for model_name in metrics_epochs_models[metric_name][i]:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        if i == 0:
            plt.ylabel("Intent Accuracy")
            task = "intents"
        else:
            plt.ylabel("Slot Filling")
            task = "slots"
        plt.title("Average " + metric_name + " over " + EVAL_TYPE + " languages")
        plt.legend(loc="lower right")
        plt.savefig(
            plot_save_dir + LANG_ORDER + "/" + task + "/" + metric_name + ".png"
        )
