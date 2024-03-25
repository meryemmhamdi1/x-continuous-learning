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
from src.utils import multiply_two_lists


EVAL_TYPE = "test"
TASK_NAME = "MTOP"
# ABLATION_MODE = "more-decks"
ABLATION_MODE = "wiped-new"
# "wiped"  # "convergence" "er-strategies" "er-techniques" "num-decks" "wiped"
LANG_ORDER = LANGUAGE_ORDERS[TASK_NAME][0]
# LANG_ORDER = LANGUAGE_ORDERS[TASK_NAME][4]
# LANG_ORDER = LANGUAGE_ORDERS[TASK_NAME][5]
# LANG_ORDER = LANGUAGE_ORDERS[TASK_NAME][6]
# LANG_ORDER = LANGUAGE_ORDERS[TASK_NAME][7]
# LANG_ORDER = LANGUAGE_ORDERS[TASK_NAME][8]
print("LANG_ORDER: ", LANG_ORDER)
LANGS = LANG_ORDER.split("_")
if TASK_NAME == "MTOP":
    MEM_SIZE = "7500"  # "5000" "2500" "10105"
elif TASK_NAME in ["TYDIQA", "MULTIATIS"]:
    MEM_SIZE = "500"
else:  # PANX
    MEM_SIZE = "1000"  # "10105"  # "500"

# 13h:25 -> 1:15
# 22/11 03:24 -> 22/11 (17:01:08)
ROOT_DIR = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"
    + TASK_NAME
    + "/HyperparamSearch/BertBaseMultilingualCased/"
)

ROOT_DIR_NEW = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/BertBaseMultilingualCased/"
    + TASK_NAME
)


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

        mono_perf[TASK_NAME].update({"fr": (0.0, 0.0)})
        mono_perf[TASK_NAME].update({"es": (0.0, 0.0)})

    return mono_perf


mono_perf = compute_mono_perf()
rand_perf = {
    "MTOP": {
        "en": (0.1823985408116735, 0.1719692790338232),
        "de": (1.4933784164553396, 0.15755581155353093),
        "hi": (0.17927572606669057, 0.16128618931729896),
        "th": (0.6871609403254972, 0.13090873843779943),
        "fr": (0, 0),
        "es": (0, 0),
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


def get_paths(model_name):
    print("model_name: ", model_name)
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
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Baseline/ER_prop-0.0_TrainERMain/"
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
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Baseline/ER_prop-0.0_TrainERMain/"
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
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/"
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
        "ltn-er-rand",
    ]:
        er_strategy = "random"
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
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
        if er_strategy == "easy":
            wipe_strategy = "hard"
        elif er_strategy == "hard":
            wipe_strategy = "easy"
        else:
            wipe_strategy = "random"

        ## TODO for MTOP
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
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
        "ltn-er-rand-prop",
    ]:
        er_strategy = "random"
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/"
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
        "ltn-er-easy-3-decks",
        "ltn-er-easy-5-decks",
        "ltn-er-easy-7-decks",
    ]:
        num_decks = model_name.split("-")[3]
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-fifo/FreqSample-epoch/FreqUpdate-epoch-everything/"
            + "NumDecks-"
            + num_decks
            + "/ER_prop-0.0_TrainERMain/_ERStrategy-easy/ER_MaxSize-"
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
        elif er_strategy == "hard":
            wipe_strategy = "easy"
            # wipe_strategy = "random"
        else:
            wipe_strategy = "random"

        ## For MTOP
        # root_results_dir = (
        #     "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
        #     + "x-continuous-learn/BertBaseMultilingualCased/"
        #     + TASK_NAME
        #     + "/Ltn/LtnModel-ltn/"
        #     + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
        #     + "NumDecks-5"
        #     + "/ER_prop-0.0_TrainERMain/_ERStrategy-"
        #     + er_strategy
        #     + "/ER_MaxSize-"
        #     + MEM_SIZE
        #     + "/WIPE_NEW/ERSched-er-only/Mode-cont-mono/"
        #     + LANG_ORDER
        #     + "/metrics.json"
        # )

        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/"
            + "x-continuous-learn/BertBaseMultilingualCased/"
            + TASK_NAME
            + "/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
            + "NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
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

    elif model_name in [
        "easy-new",
        "hard-new",
        "random-new",
        "extreme-new",
    ]:
        er_strategy = model_name.split("-")[0]
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
        "easy-wiped-new",
        "hard-wiped-new",
        "random-wiped-new",
        "extreme-wiped-new",
    ]:
        er_strategy = model_name.split("-")[0]
        if er_strategy == "easy":
            wipe_strategy = "hard"
            # wipe_strategy = "random"
        elif er_strategy == "hard":
            wipe_strategy = "easy"
            # wipe_strategy = "random"
        else:
            wipe_strategy = "random"
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
        if TASK_NAME == "MTOP":
            root_results_dir = (
                ROOT_DIR
                + "Baseline/ltmode-cont-mono/"
                + LANG_ORDER
                + "/W_SCHEDULER_0_WARMUP/"
                + "metrics.json"
            )

            acc_alias = "acc"
            slots_alias = "slots"

        else:
            if TASK_NAME == "MULTIATIS":
                alias = "ResultsSpacedRepetitionALL"
            else:
                alias = "ResultsSpacedRepetitionEREasiness"

            root_results_dir = (
                "/project/jonmay_231/meryem/"
                + alias
                + "/x-continuous-learn/"
                + "BertBaseMultilingualCased/"
                + TASK_NAME
                + "/Baseline/Mode-cont-mono/"
                + LANG_ORDER
                + "/"
                + "metrics.json"
            )

            acc_alias = "class"
            slots_alias = "tags"

    elif model_name in [
        "easy-3-decks",
        "hard-3-decks",
        "random-3-decks",
        "easy-10-decks",
        "hard-10-decks",
        "random-10-decks",
    ]:
        strateg, deck_numss, _ = model_name.split("-")
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/BertBaseMultilingualCased/MTOP/"
            + "Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-"
            + deck_numss
            + "/"
            + "ER_prop-0.0_TrainERMain/_ERStrategy-"
            + strateg
            + "/_WIPEStrategy-random/ER_MaxSize-10105/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )

        acc_alias = "class"
        slots_alias = "tags"

    return root_results_dir, acc_alias, slots_alias


def compute_metrics(model_name):
    root_results_dir, acc_alias, slots_alias = get_paths(model_name)
    avg_forgetting_epochs = ([], [])
    avg_final_perf_epochs = ([], [])
    avg_acc_perf_epochs = ([], [])
    avg_fwt_perf_epochs = ([], [])
    avg_fwt_mono_perf_epochs = ([], [])

    with open(root_results_dir) as file:
        results_json = json.load(file)

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
        avg_forgetting_epochs[0].append(avg_forget[4])
        avg_forgetting_epochs[1].append(avg_forget[5])
        #
        avg_fp = final_perf(metrics_np, LANGS)
        avg_final_perf_epochs[0].append(avg_fp[0])
        avg_final_perf_epochs[1].append(avg_fp[1])
        #
        avg_acc = acc_avg(metrics_np, LANGS, LANGS)
        avg_acc_perf_epochs[0].append(avg_acc[2])
        avg_acc_perf_epochs[1].append(avg_acc[3])

        #
        if "Transfer" in METRICS_NAMES:
            avg_fwt_mono = fwt_avg_mono(metrics_np, LANGS, mono_perf[TASK_NAME])
            avg_fwt_mono_perf_epochs[0].append(avg_fwt_mono[2])
            avg_fwt_mono_perf_epochs[1].append(avg_fwt_mono[3])

        #
        if "Zero-Shot Transfer" in METRICS_NAMES:
            avg_fwt = fwt_avg(metrics_np, LANGS, rand_perf[TASK_NAME])
            avg_fwt_perf_epochs[0].append(avg_fwt[2])
            avg_fwt_perf_epochs[1].append(avg_fwt[3])

    return (
        avg_forgetting_epochs,
        avg_final_perf_epochs,
        avg_acc_perf_epochs,
        avg_fwt_mono_perf_epochs,
        avg_fwt_perf_epochs,
    )


def compute_metrics_best_dev(model_name):
    root_results_dir, acc_alias, slots_alias = get_paths(model_name)

    with open(root_results_dir) as file:
        results_json = json.load(file)

    metrics = [[0.0 for _ in range(14)] for _ in range(7)]

    if model_name == "vanilla":  # MTOP stuff
        if TASK_NAME == "MTOP":
            alias = "val"
        else:
            alias = "valid"
    else:
        alias = "valid"

    for i, _ in enumerate(LANGS):  # training
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
        print("ep_idx_best_val:", ep_idx_best_val)
        for j, test_lang in enumerate(LANGS):  # testing
            metrics[i][2 * j] = (
                results_json[i]["test"][acc_alias][test_lang][ep_idx_best_val] * 100
            )
            metrics[i][2 * j + 1] = (
                results_json[i]["test"][slots_alias][test_lang][ep_idx_best_val] * 100
            )

    metrics_np = np.asarray(metrics)

    avg_forget = forget_avg(metrics_np, LANGS, LANGS)
    avg_forgetting = (avg_forget[4], avg_forget[5])
    #
    avg_fp = final_perf(metrics_np, LANGS)
    avg_final_perf = (avg_fp[0], avg_fp[1])
    #
    avg_acc = acc_avg(metrics_np, LANGS, LANGS)
    avg_acc_perf = (avg_acc[2], avg_acc[3])

    #
    if "Transfer" in METRICS_NAMES:
        avg_fwt_mono = fwt_avg_mono(metrics_np, LANGS, mono_perf[TASK_NAME])
        avg_fwt_mono_perf = (avg_fwt_mono[2], avg_fwt_mono[3])
    else:
        avg_fwt_mono_perf = (0.0, 0.0)

    #
    if "Zero-Shot Transfer" in METRICS_NAMES:
        avg_fwt = fwt_avg(metrics_np, LANGS, rand_perf[TASK_NAME])
        avg_fwt_perf = (avg_fwt[2], avg_fwt[3])
    else:
        avg_fwt_perf = (0.0, 0.0)

    return (
        avg_forgetting,
        avg_final_perf,
        avg_acc_perf,
        avg_fwt_mono_perf,
        avg_fwt_perf,
    )


if ABLATION_MODE == "convergence":
    SELECTED_MODELS = [
        # "vanilla",
        # "er-equal",
        # "er-rand",
        # "ltn-er-easy",
        # "ltn-er-hard",
        # "ltn-er-random",
        "vanilla",
        "ltn-er-easy-rand",
        "ltn-er-rand",
        "ltn-er-hard-rand",
        # "ltn-er-rand-prop",
    ]
    EVAL_FUNC = compute_metrics_best_dev  # compute_metrics
elif ABLATION_MODE == "convergence-new":
    SELECTED_MODELS = [
        "vanilla",
        "easy-new",
        # "extreme-new",
        "random-new",
        "hard-new",
    ]
    EVAL_FUNC = compute_metrics_best_dev  # compute_metrics
elif ABLATION_MODE == "wiped":
    SELECTED_MODELS = [
        "ltn-er-easy-wiped",
        # "ltn-er-easy",
        # "ltn-er-hard",
        # "ltn-er-hard-wiped",
        "ltn-er-random-wiped",
    ]
    EVAL_FUNC = compute_metrics_best_dev  # compute_metrics
elif ABLATION_MODE == "wiped-new":
    SELECTED_MODELS = [
        "easy-wiped-new",
        "random-wiped-new",
        "hard-wiped-new",
        # "extreme-wiped-new",
    ]
    EVAL_FUNC = compute_metrics_best_dev
elif ABLATION_MODE == "er-strategies":
    SELECTED_MODELS = [
        "ltn-er-easy",
        "ltn-er-hard",
        "ltn-er-extreme",
    ]
    EVAL_FUNC = compute_metrics_best_dev
elif ABLATION_MODE == "num-decks":
    SELECTED_MODELS = [
        "ltn-er-easy-3-decks",
        "ltn-er-easy-5-decks",
        "ltn-er-easy-7-decks",
    ]
    EVAL_FUNC = compute_metrics
elif ABLATION_MODE == "more-decks":
    SELECTED_MODELS = [
        "easy-3-decks",
        "hard-3-decks",
        "random-3-decks",
        "easy-10-decks",
        "hard-10-decks",
        "random-10-decks",
    ]
    EVAL_FUNC = compute_metrics  # compute_metrics_best_dev
else:  # "er-techniques"
    SELECTED_MODELS = [
        "ltn-er-only",
        "ltn-er-main",
        "ltn-er-both",
    ]
    EVAL_FUNC = compute_metrics_best_dev

plot_save_dir = (
    "/home1/mmhamdi/x-continuous-learning_new/outputs/plots/spacedrepetition/Test/"
    + TASK_NAME
    + "/"
    + ABLATION_MODE
    + "/"
    + LANG_ORDER
    + "/"
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
    # print("*********************************************")
    # print("SLOTS:", all_metrics_epochs_models[1])
    for method in all_metrics_epochs_models[1]:
        # if "Transfer" in method and "Zero" not in method:
        # if "Final Performance" in method:
        if "Forgetting" in method:
            print("method: ", method)
            for value in all_metrics_epochs_models[1][method]:
                print(value)

            print("*************************************************")

    if TASK_NAME in ["MTOP", "TYDIQA"]:
        df_intents = pd.DataFrame(all_metrics_epochs_models[0])
        df_slots = pd.DataFrame(all_metrics_epochs_models[1])

        df_intents.to_csv(plot_save_dir + "Intents.csv")
        df_slots.to_csv(plot_save_dir + "Slots.csv")
        i_range = [0, 1]
    else:  # PANX
        df_tags = pd.DataFrame(all_metrics_epochs_models[1])
        df_tags.to_csv("Tags_" + LANG_ORDER.upper() + ".csv")
        i_range = [1]
else:
    metrics_models = {metric_name: ({}, {}) for metric_name in METRICS_NAMES}

    intents_df = {"metric": METRICS_NAMES}
    intents_df.update({model_name: [] for model_name in SELECTED_MODELS})
    slots_df = {"metric": METRICS_NAMES}
    slots_df.update({model_name: [] for model_name in SELECTED_MODELS})
    for model_name in SELECTED_MODELS:
        outputs = EVAL_FUNC(model_name)
        for j, metric_name in enumerate(METRICS_NAMES):
            intents_df[model_name].append(outputs[j][0])

            slots_df[model_name].append(outputs[j][1])
            for i in range(2):
                metrics_models[metric_name][i].update({model_name: outputs[j][i]})

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
                print(model_name, round(metrics_models[metric_name][i][model_name], 2))
            # df["x"] = x_list
            # df["y"] = y_list
            # df["category"] = cat_list

            # # Bar Plots
            # ax = sns.barplot(x="x", y="y", hue="category", data=df)
            # # ax.bar_label(ax.containers[0], fmt="%.1f")
            # plt.show()
            # plt.savefig(plot_save_dir + y_label + "_" + metric_name + ".png")

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
    if TASK_NAME in ["MTOP", "TYDIQA"]:
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
