import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import os
from sys import platform
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score
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
from src.results_analysis.bootstrap_sampling_2 import compute_statistical_average
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

n = 100
n_iters = 100
t = 1.9849  # for 100
# 1.9639 for 600 # 95%


def std_dev(x, mean):
    sum = 0.0
    for i in range(n_iters):
        sum += (x[i] - mean) ** 2

    return sum / (n - 1)


def compute_statistical_average(l):
    mean_l = np.mean(l)
    s_l = std_dev(l, mean_l)
    d_l = t * (s_l / math.sqrt(n))
    return mean_l, d_l


EVAL_TYPE = "test"
TASK_NAME = "MTOP"
ABLATION_MODE = "convergence-new"  # "convergence" "er-strategies" "er-techniques" "num-decks" "wiped"
MEM_SIZE = "10105"

ROOT_DIR = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"
    + TASK_NAME
    + "/HyperparamSearch/BertBaseMultilingualCased/"
)


def compute_mono_perf():
    mono_perf = {TASK_NAME: {}}
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
}

DIR_LIT = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetitionEREasiness/x-continuous-learn/BertBaseMultilingualCased/"
    + TASK_NAME
)


def get_paths(model_name):
    if model_name == "vanilla":
        root_results_dir = ROOT_DIR + "Baseline/ltmode-cont-mono/"

    elif model_name in [
        "ltn-er-easy-fifo-5",
        "ltn-er-hard-fifo-5",
        "ltn-er-random-fifo-5",
        "ltn-er-easy-rand-5",
        "ltn-er-hard-rand-5",
        "ltn-er-random-rand-5",
        "ltn-er-random-randprop-5",
        "ltn-er-easy-fifo-3",
        "ltn-er-easy-fifo-7",
    ]:
        _, _, er_strat, rep_strat, num_decks = model_name.split("-")
        if rep_strat == "randprop":
            rep_strat = "rand-prop"
        root_results_dir = (
            DIR_LIT
            + "/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-"
            + rep_strat
            + "/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-"
            + num_decks
            + "/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strat
            + "/ER_MaxSize-"
            + MEM_SIZE
            + "/ERSched-er-only/Mode-cont-mono/"
        )

    elif model_name in [
        "ltn-er-easy-wiped",
        "ltn-er-hard-wiped",
        "ltn-er-random-wiped",
    ]:
        er_strategy = model_name.split("-")[2]
        root_results_dir = (
            DIR_LIT
            + "/Ltn/LtnModel-ltn/"
            + "DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/"
            + "NumDecks-5"
            + "/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/ER_MaxSize-"
            + MEM_SIZE
            + "/WIPE/ERSched-er-only/Mode-cont-mono/"
        )
        acc_alias = "class"
        slots_alias = "tags"

    elif model_name in ["er-equallang", "er-random"]:
        er_strat = model_name.split("-")[1]
        if er_strat == "equallang":
            er_strat = "equal-lang"
        root_results_dir = (
            DIR_LIT
            + "/Baseline/ER_prop-0.0_TrainERMain/"
            + "_ERStrategy-"
            + er_strat
            + "/ER_MaxSize-"
            + MEM_SIZE
            + "/Mode-cont-mono/"
        )

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
        )
        acc_alias = "class"
        slots_alias = "tags"

    acc_alias = "acc"
    slots_alias = "slots"
    return root_results_dir, acc_alias, slots_alias


def bootstrap(root_dir, lang_order, epoch, model_name):
    all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]
    lang_list = lang_order.split("_")

    # Approach 1 : Compute the metrics for all
    acc_lang = []
    for i_train, train_lang in enumerate(lang_list):
        acc_lang.append(train_lang)
        for j_test, test_lang in enumerate(lang_list):
            if model_name == "vanilla":
                with open(
                    os.path.join(
                        root_dir,
                        "Test_on-"
                        + test_lang
                        + "_aftertrainon-"
                        + str(i_train)
                        + "_epoch-"
                        + str(epoch),
                    ),
                    "r",
                ) as file:
                    results = file.read().splitlines()
            else:
                with open(
                    os.path.join(
                        root_dir,
                        "test_on-"
                        + test_lang
                        + "_aftertrainon-"
                        + str(i_train)
                        + "_epoch-"
                        + str(epoch),
                    ),
                    "r",
                ) as file:
                    results = file.read().splitlines()

            sampled_results = random.choices(results, k=n)

            intent_corrects = 0

            intents_true = []
            intents_pred = []

            slots_true = []
            slots_pred = []

            for _, sent in enumerate(sampled_results):
                parts = sent.split("\t")
                if len(parts) == 5:
                    sent_text, intent_true, intent_pred, slot_true, slot_pred = parts
                else:
                    assert len(parts) == 6
                    (
                        sent_id,
                        sent_text,
                        intent_true,
                        intent_pred,
                        slot_true,
                        slot_pred,
                    ) = parts

                intents_true.append(intent_true)
                intents_pred.append(intent_pred)

                #####
                slot_true_list = slot_true.split(" ")
                slot_pred_list = slot_pred.split(" ")

                if len(slot_true_list) != len(slot_pred_list):
                    print(
                        "len(slot_true): ",
                        len(slot_true),
                        "len(slot_pred): ",
                        len(slot_pred),
                        " slot_true:",
                        slot_true,
                        " slot_pred:",
                        slot_pred,
                    )

                assert len(slot_true_list) == len(slot_pred_list)

                slots_true.extend(slot_true_list)
                slots_pred.extend(slot_pred_list)

                intent_corrects += int(intent_pred == intent_true)

            intent_acc = float(intent_corrects) * 100 / n
            slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

            all_metrics[i_train][2 * j_test] = intent_acc
            all_metrics[i_train][2 * j_test + 1] = slot_f1

    all_metrics_np = np.asarray(all_metrics)
    (
        acc_avg_all,
        bwt_avg_all,
        fwt_avg_all,
        fwt_avg_mono_all,
        forget_perf_all,
        final_perf_all,
    ) = (
        acc_avg(all_metrics_np, lang_list, lang_list),
        bwt_avg(all_metrics_np, lang_list, lang_list),
        fwt_avg(all_metrics_np, lang_list, rand_perf[TASK_NAME]),
        fwt_avg_mono(all_metrics_np, lang_list, mono_perf[TASK_NAME]),
        forget_avg(all_metrics_np, lang_list, lang_list),
        final_perf(all_metrics_np, lang_list),
    )
    return (
        acc_avg_all,
        bwt_avg_all,
        fwt_avg_all,
        fwt_avg_mono_all,
        forget_perf_all,
        final_perf_all,
    )


def compute_bootstrap(root_dir, lang_order, epoch, model_name):
    acc_array = ([], [])
    forget_array = ([], [])
    fwt_array = ([], [])
    fwt_mono_array = ([], [])
    final_perf_array = ([], [])
    for _ in range(n_iters):
        (
            acc_avg_all,
            _,
            fwt_avg_all,
            fwt_avg_mono_all,
            forget_perf_all,
            final_perf_all,
        ) = bootstrap(root_dir, lang_order, epoch, model_name)

        acc_array[0].append(acc_avg_all[2])
        acc_array[1].append(acc_avg_all[3])

        forget_array[0].append(forget_perf_all[4])
        forget_array[1].append(forget_perf_all[5])

        fwt_array[0].append(fwt_avg_all[2])
        fwt_array[1].append(fwt_avg_all[3])

        fwt_mono_array[0].append(fwt_avg_mono_all[2])
        fwt_mono_array[1].append(fwt_avg_mono_all[3])

        final_perf_array[0].append(final_perf_all[0])
        final_perf_array[1].append(final_perf_all[1])

    mean_l_a_i, d_l_a_i = compute_statistical_average(acc_array[0])
    mean_l_a_s, d_l_a_s = compute_statistical_average(acc_array[1])

    mean_l_f_i, d_l_f_i = compute_statistical_average(forget_array[0])
    mean_l_f_s, d_l_f_s = compute_statistical_average(forget_array[1])

    mean_l_fwt_i, d_l_fwt_i = compute_statistical_average(fwt_array[0])
    mean_l_fwt_s, d_l_fwt_s = compute_statistical_average(fwt_array[1])

    mean_l_fwt_mono_i, d_l_fwt_mono_i = compute_statistical_average(fwt_mono_array[0])
    mean_l_fwt_mono_s, d_l_fwt_mono_s = compute_statistical_average(fwt_mono_array[1])

    mean_l_fp_i, d_l_fp_i = compute_statistical_average(final_perf_array[0])
    mean_l_fp_s, d_l_fp_s = compute_statistical_average(final_perf_array[1])

    return (
        mean_l_a_i,
        mean_l_a_s,
        mean_l_f_i,
        mean_l_f_s,
        mean_l_fwt_i,
        mean_l_fwt_s,
        mean_l_fwt_mono_i,
        mean_l_fwt_mono_s,
        mean_l_fp_i,
        mean_l_fp_s,
    )


def compute_bootstrap_per_lang(root_dir, lang_order, epoch, model_name):
    acc_array = (
        {lang: [] for lang in lang_order.split("_")},
        {lang: [] for lang in lang_order.split("_")},
    )
    forget_array = (
        {lang: [] for lang in lang_order.split("_")},
        {lang: [] for lang in lang_order.split("_")},
    )
    fwt_array = (
        {lang: [] for lang in lang_order.split("_")},
        {lang: [] for lang in lang_order.split("_")},
    )
    fwt_mono_array = (
        {lang: [] for lang in lang_order.split("_")},
        {lang: [] for lang in lang_order.split("_")},
    )
    final_perf_array = (
        {lang: [] for lang in lang_order.split("_")},
        {lang: [] for lang in lang_order.split("_")},
    )
    for _ in range(n_iters):
        (
            acc_avg_all,
            _,
            fwt_avg_all,
            fwt_avg_mono_all,
            forget_perf_all,
            final_perf_all,
        ) = bootstrap(root_dir, lang_order, epoch, model_name)

        acc_array[0].append(acc_avg_all[2])
        acc_array[1].append(acc_avg_all[3])

        forget_array[0].append(forget_perf_all[4])
        forget_array[1].append(forget_perf_all[5])

        fwt_array[0].append(fwt_avg_all[2])
        fwt_array[1].append(fwt_avg_all[3])

        fwt_mono_array[0].append(fwt_avg_mono_all[2])
        fwt_mono_array[1].append(fwt_avg_mono_all[3])

        final_perf_array[0].append(final_perf_all[0])
        final_perf_array[1].append(final_perf_all[1])

    mean_l_a_i, d_l_a_i = compute_statistical_average(acc_array[0])
    mean_l_a_s, d_l_a_s = compute_statistical_average(acc_array[1])

    mean_l_f_i, d_l_f_i = compute_statistical_average(forget_array[0])
    mean_l_f_s, d_l_f_s = compute_statistical_average(forget_array[1])

    mean_l_fwt_i, d_l_fwt_i = compute_statistical_average(fwt_array[0])
    mean_l_fwt_s, d_l_fwt_s = compute_statistical_average(fwt_array[1])

    mean_l_fwt_mono_i, d_l_fwt_mono_i = compute_statistical_average(fwt_mono_array[0])
    mean_l_fwt_mono_s, d_l_fwt_mono_s = compute_statistical_average(fwt_mono_array[1])

    mean_l_fp_i, d_l_fp_i = compute_statistical_average(final_perf_array[0])
    mean_l_fp_s, d_l_fp_s = compute_statistical_average(final_perf_array[1])

    return (
        mean_l_a_i,
        mean_l_a_s,
        mean_l_f_i,
        mean_l_f_s,
        mean_l_fwt_i,
        mean_l_fwt_s,
        mean_l_fwt_mono_i,
        mean_l_fwt_mono_s,
        mean_l_fp_i,
        mean_l_fp_s,
    )


do_language_order = True
LANG_ORDER = "en_de_hi_th"

# do_language_order = False


def compute_metrics_one_lang_order(model_name):
    root_results_dir, acc_alias, slots_alias = get_paths(model_name)
    avg_forgetting_epochs = ([], [])
    avg_final_perf_epochs = ([], [])
    avg_acc_perf_epochs = ([], [])
    avg_fwt_perf_epochs = ([], [])
    avg_fwt_mono_perf_epochs = ([], [])

    print("For ", model_name, " ...")
    for epoch in tqdm(range(10)):
        if model_name == "vanilla":
            root_dir = os.path.join(
                root_results_dir,
                LANG_ORDER,
                "W_SCHEDULER_0_WARMUP",
                "predictions",
            )
        else:
            root_dir = os.path.join(root_results_dir, LANG_ORDER, "predictions")

        # TODO Compute boostrap here
        (
            a_i,
            a_s,
            f_i,
            f_s,
            fwt_i,
            fwt_s,
            fwt_mono_i,
            fwt_mono_s,
            fp_i,
            fp_s,
        ) = compute_bootstrap(root_dir, LANG_ORDER, epoch, model_name)

        # AVERAGE OVER DIFFERENT LANGUAGE ORDERS
        avg_acc_perf_epochs[0].append(a_i)
        avg_acc_perf_epochs[1].append(a_s)

        #
        avg_forgetting_epochs[0].append(f_i)
        avg_forgetting_epochs[1].append(f_s)
        #
        avg_final_perf_epochs[0].append(fp_i)
        avg_final_perf_epochs[1].append(fp_s)

        #
        if "Zero-Shot Transfer" in METRICS_NAMES:
            avg_fwt_perf_epochs[0].append(fwt_i)
            avg_fwt_perf_epochs[1].append(fwt_s)

        #
        if "Transfer" in METRICS_NAMES:
            avg_fwt_mono_perf_epochs[0].append(fwt_mono_i)
            avg_fwt_mono_perf_epochs[1].append(fwt_mono_s)

    return (
        avg_forgetting_epochs,
        avg_final_perf_epochs,
        avg_acc_perf_epochs,
        avg_fwt_mono_perf_epochs,
        avg_fwt_perf_epochs,
    )


def compute_metrics(model_name):
    root_results_dir, acc_alias, slots_alias = get_paths(model_name)
    avg_forgetting_epochs = ([], [])
    avg_final_perf_epochs = ([], [])
    avg_acc_perf_epochs = ([], [])
    avg_fwt_perf_epochs = ([], [])
    avg_fwt_mono_perf_epochs = ([], [])

    print("For ", model_name, " ...")
    for epoch in tqdm(range(10)):
        (
            a_i_l,
            a_s_l,
            f_i_l,
            f_s_l,
            fp_i_l,
            fp_s_l,
            fwt_mono_i_l,
            fwt_mono_s_l,
            fwt_i_l,
            fwt_s_l,
        ) = ([], [], [], [], [], [], [], [], [], [])
        for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]:
            if model_name == "vanilla":
                root_dir = os.path.join(
                    root_results_dir,
                    LANG_ORDER,
                    "W_SCHEDULER_0_WARMUP",
                    "predictions",
                )
            else:
                root_dir = os.path.join(root_results_dir, LANG_ORDER, "predictions")

            # TODO Compute boostrap here
            (
                a_i,
                a_s,
                f_i,
                f_s,
                fwt_i,
                fwt_s,
                fwt_mono_i,
                fwt_mono_s,
                fp_i,
                fp_s,
            ) = compute_bootstrap(root_dir, LANG_ORDER, epoch, model_name)

            a_i_l.append(a_i)
            a_s_l.append(a_s)

            f_i_l.append(f_i)
            f_s_l.append(f_s)

            fp_i_l.append(fp_i)
            fp_s_l.append(fp_s)

            fwt_i_l.append(fwt_i)
            fwt_s_l.append(fwt_s)

            fwt_mono_i_l.append(fwt_mono_i)
            fwt_mono_s_l.append(fwt_mono_s)

        # AVERAGE OVER DIFFERENT LANGUAGE ORDERS
        avg_acc_perf_epochs[0].append(np.mean(a_i_l))
        avg_acc_perf_epochs[1].append(np.mean(a_s_l))

        #
        avg_forgetting_epochs[0].append(np.mean(f_i_l))
        avg_forgetting_epochs[1].append(np.mean(f_s_l))
        #
        avg_final_perf_epochs[0].append(np.mean(fp_i_l))
        avg_final_perf_epochs[1].append(np.mean(fp_s_l))

        #
        if "Zero-Shot Transfer" in METRICS_NAMES:
            avg_fwt_perf_epochs[0].append(np.mean(fwt_i_l))
            avg_fwt_perf_epochs[1].append(np.mean(fwt_s_l))

        #
        if "Transfer" in METRICS_NAMES:
            avg_fwt_mono_perf_epochs[0].append(np.mean(fwt_mono_i_l))
            avg_fwt_mono_perf_epochs[1].append(np.mean(fwt_mono_s_l))

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

    for i, _ in enumerate(LANGS):  # training
        best_valid_perf = multiply_two_lists(
            results_json[i]["valid"][acc_alias][str(i)],
            results_json[i]["valid"][slots_alias][str(i)],
        )
        ep_idx_best_val = best_valid_perf.index(max(best_valid_perf))
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


def compute_metrics_best_dev_bootstrap(model_name):
    root_results_dir, acc_alias, slots_alias = get_paths(model_name)
    avg_forgetting_epochs = ([], [])
    avg_final_perf_epochs = ([], [])
    avg_acc_perf_epochs = ([], [])
    avg_fwt_perf_epochs = ([], [])
    avg_fwt_mono_perf_epochs = ([], [])

    # Find the best idx
    ep_ids = []
    with open(root_results_dir) as file:
        results_json = json.load(file)

    LANGS = LANG_ORDER.split("_")
    for i, _ in enumerate(LANGS):  # training
        best_valid_perf = multiply_two_lists(
            results_json[i]["valid"][acc_alias][str(i)],
            results_json[i]["valid"][slots_alias][str(i)],
        )
        ep_idx_best_val = best_valid_perf.index(max(best_valid_perf))
        ep_ids.append(ep_idx_best_val)

    print("For ", model_name, " ...")
    (
        a_i_l,
        a_s_l,
        f_i_l,
        f_s_l,
        fp_i_l,
        fp_s_l,
        fwt_mono_i_l,
        fwt_mono_s_l,
        fwt_i_l,
        fwt_s_l,
    ) = ([], [], [], [], [], [], [], [], [], [])
    for LANG_ORDER in LANGUAGE_ORDERS[TASK_NAME]:
        if model_name == "vanilla":
            root_dir = os.path.join(
                root_results_dir,
                LANG_ORDER,
                "W_SCHEDULER_0_WARMUP",
                "predictions",
            )
        else:
            root_dir = os.path.join(root_results_dir, LANG_ORDER, "predictions")

        # TODO Compute boostrap here
        (
            a_i,
            a_s,
            f_i,
            f_s,
            fwt_i,
            fwt_s,
            fwt_mono_i,
            fwt_mono_s,
            fp_i,
            fp_s,
        ) = compute_bootstrap(root_dir, LANG_ORDER, ep_ids, model_name)

        a_i_l.append(a_i)
        a_s_l.append(a_s)

        f_i_l.append(f_i)
        f_s_l.append(f_s)

        fp_i_l.append(fp_i)
        fp_s_l.append(fp_s)

        fwt_i_l.append(fwt_i)
        fwt_s_l.append(fwt_s)

        fwt_mono_i_l.append(fwt_mono_i)
        fwt_mono_s_l.append(fwt_mono_s)

    # AVERAGE OVER DIFFERENT LANGUAGE ORDERS
    avg_acc_perf_epochs = (np.mean(a_i_l), np.mean(a_s_l))

    #
    avg_forgetting_epochs = (np.mean(f_i_l), np.mean(f_s_l))
    #
    avg_final_perf_epochs = (np.mean(fp_i_l), np.mean(fp_s_l))

    #
    if "Zero-Shot Transfer" in METRICS_NAMES:
        avg_fwt_perf_epochs = (np.mean(fwt_i_l), np.mean(fwt_s_l))

    #
    if "Transfer" in METRICS_NAMES:
        avg_fwt_mono_perf_epochs = (np.mean(fwt_mono_i_l), np.mean(fwt_mono_s_l))

    return (
        avg_forgetting_epochs,
        avg_final_perf_epochs,
        avg_acc_perf_epochs,
        avg_fwt_mono_perf_epochs,
        avg_fwt_perf_epochs,
    )


if ABLATION_MODE == "convergence":
    SELECTED_MODELS = [
        "vanilla",
        # "er-equallang",
        # "er-random",
        # "ltn-er-easy-fifo-5",
        # "ltn-er-hard-fifo-5",
        # "ltn-er-random-fifo-5",
        "ltn-er-random-rand-5",
        "ltn-er-easy-rand-5",
        "ltn-er-hard-rand-5",
    ]
    if do_language_order:
        EVAL_FUNC = compute_metrics_one_lang_order
    else:
        EVAL_FUNC = compute_metrics

elif ABLATION_MODE == "convergence-new":
    SELECTED_MODELS = [
        "easy-new",
        "hard-new",
        "random-new",
    ]
    if do_language_order:
        EVAL_FUNC = compute_metrics_one_lang_order
    else:
        EVAL_FUNC = compute_metrics


elif ABLATION_MODE == "wiped-new":
    SELECTED_MODELS = [
        "easy-wiped-new",
        "random-wiped-new",
        "hard-wiped-new",
        # "extreme-wiped-new",
    ]
    if do_language_order:
        EVAL_FUNC = compute_metrics_one_lang_order
    else:
        EVAL_FUNC = compute_metrics
elif ABLATION_MODE == "wiped":
    SELECTED_MODELS = [
        "ltn-er-easy-wiped",
        # "ltn-er-easy",
        # "ltn-er-hard",
        "ltn-er-hard-wiped",
        "ltn-er-random-wiped",
    ]
    if do_language_order:
        EVAL_FUNC = compute_metrics_one_lang_order
    else:
        EVAL_FUNC = compute_metrics
elif ABLATION_MODE == "er-strategies":
    SELECTED_MODELS = [
        "ltn-er-easy",
        "ltn-er-hard",
        "ltn-er-extreme",
    ]
    if do_language_order:
        EVAL_FUNC = compute_metrics_one_lang_order
    else:
        EVAL_FUNC = compute_metrics
elif ABLATION_MODE == "num-decks":
    SELECTED_MODELS = [
        "ltn-er-easy-fifo-3",
        "ltn-er-easy-fifo-5",
        "ltn-er-easy-fifo-7",
    ]
    if do_language_order:
        EVAL_FUNC = compute_metrics_one_lang_order
    else:
        EVAL_FUNC = compute_metrics
else:  # "er-techniques"
    SELECTED_MODELS = [
        "ltn-er-only",
        "ltn-er-main",
        "ltn-er-both",
    ]
    EVAL_FUNC = compute_metrics_best_dev

if do_language_order:
    plot_save_dir = (
        "/home1/mmhamdi/x-continuous-learning_new/outputs/plots/spacedrepetition/Test/"
        + TASK_NAME
        + "/"
        + ABLATION_MODE
        + "/"
        + "/AVERAGE_BOOTSTRAP/"
        + LANG_ORDER
        + "/"
    )
else:
    plot_save_dir = (
        "/home1/mmhamdi/x-continuous-learning_new/outputs/plots/spacedrepetition/Test/"
        + TASK_NAME
        + "/"
        + ABLATION_MODE
        + "/AVERAGE_BOOTSTRAP/"
    )
if not os.path.isdir(plot_save_dir):
    os.makedirs(plot_save_dir)

if EVAL_FUNC == compute_metrics or EVAL_FUNC == compute_metrics_one_lang_order:
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
    if TASK_NAME == "MTOP":
        df_intents = pd.DataFrame(all_metrics_epochs_models[0])
        df_slots = pd.DataFrame(all_metrics_epochs_models[1])

        df_intents.to_csv(plot_save_dir + "Intents.csv")
        df_slots.to_csv(plot_save_dir + "Slots.csv")
        i_range = [0, 1]

    print("INTENTS:", all_metrics_epochs_models[0])
    print("SLOTS:", all_metrics_epochs_models[1])

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

    if TASK_NAME == "MTOP":
        pd.DataFrame(intents_df).to_csv(plot_save_dir + "Intents" + ".csv")
        pd.DataFrame(slots_df).to_csv(plot_save_dir + "Slots.csv")
    else:  # PANX
        pd.DataFrame(slots_df).to_csv(plot_save_dir + "Tags.csv")
