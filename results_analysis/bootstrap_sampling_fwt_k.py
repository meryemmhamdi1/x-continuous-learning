from sklearn.metrics import f1_score, precision_score, recall_score
from random import choices
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_k, fwt_avg_mono, bwt_avg, forget_avg, final_perf
from scipy import stats
import scikits.bootstrap as bootstrap
import scipy
import pickle
import os

def compute_statistical_average(l, n_iters):
    mean_l = np.mean(l)
    s_l = std_dev(l, mean_l, n_iters)
    d_l = t * (s_l / math.sqrt(n))
    return mean_l, d_l

def compute_random(root_dir, seed="42"):
    print("Computing Bootstrap for RANDOM ")
    rand_perf = {lang: [0.0, 0.0] for lang in languages}
    rand_conf = {lang: [0.0, 0.0] for lang in languages}
    prediction_dir = root_dir+"/random/SEED_"+seed+"/" # TODO Change
    for i_test, test_lang in enumerate(languages):
        with open(prediction_dir + "initial_perf_"+test_lang + ".txt", "r") as file:
            results = file.read().splitlines()

        accs = []
        f1s = []

        # n = len(results)
        overall_picked_indices = []
        for _ in tqdm(range(n_iters)):
            intent_corrects = 0

            intents_true = []
            intents_pred = []

            slots_true = []
            slots_pred = []

            indices = list(range(len(results)))
            picked_indices = random.choices(indices, k=n)
            # sampled_results = random.choices(results, k=n)
            overall_picked_indices.extend(picked_indices)

            for i in picked_indices:
                parts = results[i].split("\t")
                assert len(parts) == 5
                sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

                intents_true.append(intent_true)
                intents_pred.append(intent_pred)

                #####
                slot_true_list = slot_true.split(" ")
                slot_pred_list = slot_pred.split(" ")

                if len(slot_true_list) != len(slot_pred_list):
                    print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                          " slot_true:", slot_true, " slot_pred:", slot_pred)

                assert len(slot_true_list) == len(slot_pred_list)

                slots_true.extend(slot_true_list)
                slots_pred.extend(slot_pred_list)

                intent_corrects += int(intent_pred == intent_true)

            intent_accuracy = float(intent_corrects)*100 / n
            slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100
            # print("slots_true:", slots_true, " slots_pred:", slots_pred, " slot_f1:", slot_f1)

            accs.append(intent_accuracy)
            f1s.append(slot_f1)

        mean_acc = np.mean(accs)
        s_acc = std_dev(accs, mean_acc, n_iters)
        d_acc = t * (s_acc / math.sqrt(n))

        mean_f1 = np.mean(f1s)
        s_f1 = std_dev(f1s, mean_f1, n_iters)
        d_f1 = t * (s_f1 / math.sqrt(n_iters))

        rand_perf[test_lang][0] = mean_acc
        rand_perf[test_lang][1] = mean_f1

        rand_conf[test_lang][0] = d_acc
        rand_conf[test_lang][1] = d_f1

    return rand_perf, rand_conf


def compute_multi(root_dir, seed="42"):
    print("Computing Bootstrap for MULTI ")
    multi_perf = {lang: [0.0, 0.0] for lang in languages}
    multi_conf = {lang: [0.0, 0.0] for lang in languages}
    prediction_dir = root_dir+"/multi/SEED_"+seed+"/"
    for i_test, test_lang in enumerate(languages):
        with open(prediction_dir + "End_test_perf-train_de-en-es-fr-hi-th-test_"+test_lang, "r") as file:
            results = file.read().splitlines()

        accs = []
        f1s = []

        # n = len(results)
        for _ in range(n_iters):
            intent_corrects = 0

            intents_true = []
            intents_pred = []

            slots_true = []
            slots_pred = []

            sampled_results = random.choices(results, k=n)
            for i, sent in enumerate(sampled_results):
                parts = sent.split("\t")
                assert len(parts) == 5
                sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

                intents_true.append(intent_true)
                intents_pred.append(intent_pred)

                #####
                slot_true_list = slot_true.split(" ")
                slot_pred_list = slot_pred.split(" ")

                if len(slot_true_list) != len(slot_pred_list):
                    print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                          " slot_true:", slot_true, " slot_pred:", slot_pred)

                assert len(slot_true_list) == len(slot_pred_list)

                slots_true.extend(slot_true_list)
                slots_pred.extend(slot_pred_list)

                intent_corrects += int(intent_pred == intent_true)

            intent_accuracy = float(intent_corrects)*100 / n
            slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

            accs.append(intent_accuracy)
            f1s.append(slot_f1)

        mean_acc = np.mean(accs)
        s_acc = std_dev(accs, mean_acc, n_iters)
        d_acc = t * (s_acc / math.sqrt(n))

        mean_f1 = np.mean(f1s)
        s_f1 = std_dev(f1s, mean_f1, n_iters)
        d_f1 = t * (s_f1 / math.sqrt(n_iters))

        multi_perf[test_lang][0] = mean_acc
        multi_perf[test_lang][1] = mean_f1

        multi_conf[test_lang][0] = d_acc
        multi_conf[test_lang][1] = d_f1

    return multi_perf, multi_conf


def compute_mono(root_dir, seed="42", mode="orig"):
    print("Computing MONO ", mode)
    mono_perf = {lang: [0.0, 0.0] for lang in languages}
    mono_conf = {lang: [0.0, 0.0] for lang in languages}
    for i_train, train_lang in enumerate(languages):
        if mode != "orig":
            prediction_dir = root_dir+"/mono/adapters/" + train_lang +  "/" + mode + "/SEED_"+seed+"/"
        else:
            prediction_dir = root_dir+"/mono/vanilla/" + train_lang + "/SEED_"+seed+"/"

        with open(prediction_dir + "End_test_perf-train_"+train_lang+"-test_"+train_lang, "r") as file:
            results = file.read().splitlines()

        accs = []
        f1s = []

        # n = len(results)
        for iter in range(n_iters):
            intent_corrects = 0

            intents_true = []
            intents_pred = []

            slots_true = []
            slots_pred = []

            sampled_results = random.choices(results, k=n)
            for i, sent in enumerate(sampled_results):
                parts = sent.split("\t")
                assert len(parts) == 5
                sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

                intents_true.append(intent_true)
                intents_pred.append(intent_pred)

                #####
                slot_true_list = slot_true.split(" ")
                slot_pred_list = slot_pred.split(" ")

                if len(slot_true_list) != len(slot_pred_list):
                    print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                          " slot_true:", slot_true, " slot_pred:", slot_pred)

                assert len(slot_true_list) == len(slot_pred_list)

                slots_true.extend(slot_true_list)
                slots_pred.extend(slot_pred_list)

                intent_corrects += int(intent_pred == intent_true)

            intent_accuracy = float(intent_corrects)*100 / n
            slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

            accs.append(intent_accuracy)
            f1s.append(slot_f1)

        mean_acc = np.mean(accs)
        s_acc = std_dev(accs, mean_acc, n_iters)
        d_acc = t * (s_acc / math.sqrt(n))

        mean_f1 = np.mean(f1s)
        s_f1 = std_dev(f1s, mean_f1, n_iters)
        d_f1 = t * (s_f1 / math.sqrt(n_iters))

        mono_perf[train_lang][0] = mean_acc
        mono_perf[train_lang][1] = mean_f1

        mono_conf[train_lang][0] = d_acc
        mono_conf[train_lang][1] = d_f1

    return mono_perf, mono_conf


def std_dev(x, mean, n):
    sum = 0.0
    for i in range(n):
        sum += ((x[i]-mean)**2)

    return sum / (n-1)


def compute_bootstrap(root_dir, lang_order, model_name, n, seed):
    if model_name == "adapters/TUNED_BERT":
        mono_perf = mono_ada_tuned_perf
    elif model_name == "adapters/FROZEN_BERT":
        mono_perf = mono_ada_frozen_perf
    else:
        mono_perf = mono_orig_perf

    all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]
    lang_list = lang_order.split("_")

    # Approach 1 : Compute the metrics for all
    acc_lang = []
    for i_train, train_lang in enumerate(lang_list):
        acc_lang.append(train_lang)
        for j_test, test_lang in enumerate(lang_list):
            # print("Computing for train_lang:", train_lang, " test_lang:", test_lang)
            if model_name == "joint_plus":
                file_name = root_dir+"/"+lang_order+"/"+model_name+"/SEED_"+seed+"/End_test_perf-train_"+"-".join(acc_lang)+"-test_"+test_lang
            else:
                file_name = root_dir+"/"+lang_order+"/"+model_name+"/SEED_"+seed+"/End_test_perf-train_"+train_lang+"-test_"+test_lang

            with open(file_name, "r") as file:
                results = file.read().splitlines()

            sampled_results = choices(results, k=n)

            intent_corrects = 0

            intents_true = []
            intents_pred = []

            slots_true = []
            slots_pred = []

            for i, sent in enumerate(sampled_results):
                parts = sent.split("\t")
                assert len(parts) == 5
                sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

                intents_true.append(intent_true)
                intents_pred.append(intent_pred)

                #####
                slot_true_list = slot_true.split(" ")
                slot_pred_list = slot_pred.split(" ")

                if len(slot_true_list) != len(slot_pred_list):
                    print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                          " slot_true:", slot_true, " slot_pred:", slot_pred)

                assert len(slot_true_list) == len(slot_pred_list)

                slots_true.extend(slot_true_list)
                slots_pred.extend(slot_pred_list)

                intent_corrects += int(intent_pred == intent_true)

            intent_acc = float(intent_corrects)*100 / n
            slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

            all_metrics[i_train][2*j_test] = intent_acc
            all_metrics[i_train][2*j_test+1] = slot_f1

    all_metrics_np = np.asarray(all_metrics)
    acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_all_1, fwt_avg_all_2, fwt_avg_all_3, fwt_avg_all_4, fwt_avg_all_5,\
    fwt_avg_mono_all, forget_perf_all, final_perf_all = \
        acc_avg(all_metrics_np, lang_list, lang_list), \
        bwt_avg(all_metrics_np, lang_list, lang_list), \
        fwt_avg(all_metrics_np, lang_list, rand_perf), \
        fwt_avg_k(all_metrics_np, lang_list, rand_perf, k=1), \
        fwt_avg_k(all_metrics_np, lang_list, rand_perf, k=2), \
        fwt_avg_k(all_metrics_np, lang_list, rand_perf, k=3), \
        fwt_avg_k(all_metrics_np, lang_list, rand_perf, k=4), \
        fwt_avg_k(all_metrics_np, lang_list, rand_perf, k=5), \
        fwt_avg_mono(all_metrics_np, lang_list, mono_perf), \
        forget_avg(all_metrics_np, lang_list, lang_list), \
        final_perf(all_metrics_np, lang_list)
    return acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_all_1, fwt_avg_all_2, fwt_avg_all_3, fwt_avg_all_4, fwt_avg_all_5, fwt_avg_mono_all, forget_perf_all, final_perf_all


def method_1(root_dir, n, n_iters, model_name):
    forget_intent_array_all = []  # for all samples x lang_orders
    forget_slot_array_all = []  # for all samples x lang_orders
    fwt_intent_array_all = []  # for all samples x lang_orders
    fwt_slot_array_all = []  # for all samples x lang_orders
    fwt_mono_intent_array_all = []  # for all samples x lang_orders
    fwt_mono_slot_array_all = []  # for all samples x lang_orders
    final_perf_intent_array_all = []  # for all samples x lang_orders
    final_perf_slot_array_all = []  # for all samples x lang_orders
    for iter in range(n_iters):
        forget_intent_array_sub = []
        forget_slot_array_sub = []
        fwt_intent_array_sub = []
        fwt_slot_array_sub = []
        fwt_mono_intent_array_sub = []
        fwt_mono_slot_array_sub = []
        final_perf_intent_array_sub = []
        final_perf_slot_array_sub = []
        for lang_order in lang_orders:
            acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all \
                = compute_bootstrap(root_dir, lang_order, model_name, n)

            results_array = [round(forget_perf_all[2], 2), round(forget_perf_all[3], 2),
                             round(fwt_avg_all[2], 2), round(fwt_avg_all[3], 2),
                             round(fwt_avg_mono_all[2], 2), round(fwt_avg_mono_all[3], 2),
                             round(final_perf_all[0], 2), round(final_perf_all[1], 2)]

            forget_intent_array_sub.append(results_array[0])
            forget_slot_array_sub.append(results_array[1])

            fwt_intent_array_sub.append(results_array[2])
            fwt_slot_array_sub.append(results_array[3])

            fwt_mono_intent_array_sub.append(results_array[4])
            fwt_mono_slot_array_sub.append(results_array[5])

            final_perf_intent_array_sub.append(results_array[6])
            final_perf_slot_array_sub.append(results_array[7])

        forget_intent_array_all.append(forget_intent_array_sub)
        forget_slot_array_all.append(forget_slot_array_sub)

        fwt_intent_array_all.append(fwt_intent_array_sub)
        fwt_slot_array_all.append(fwt_slot_array_sub)

        fwt_mono_intent_array_all.append(fwt_mono_intent_array_sub)
        fwt_mono_slot_array_all.append(fwt_mono_slot_array_sub)

        final_perf_intent_array_all.append(final_perf_intent_array_sub)
        final_perf_slot_array_all.append(final_perf_slot_array_sub)

    return forget_intent_array_all, forget_slot_array_all, fwt_intent_array_all, fwt_slot_array_all, \
           fwt_mono_intent_array_all, fwt_mono_slot_array_all, final_perf_intent_array_all, final_perf_slot_array_all


def method_2(root_dir, n, n_iters, model_name, seed):
    forget_intent_array_all = []  # for all samples x lang_orders
    forget_slot_array_all = []  # for all samples x lang_orders
    fwt_intent_array_all = []  # for all samples x lang_orders
    fwt_slot_array_all = []  # for all samples x lang_orders
    fwt_mono_intent_array_all = []  # for all samples x lang_orders
    fwt_mono_slot_array_all = []  # for all samples x lang_orders

    fwt_1_intent_array_all = []  # for all samples x lang_orders
    fwt_1_slot_array_all = []  # for all samples x lang_orders

    fwt_2_intent_array_all = []  # for all samples x lang_orders
    fwt_2_slot_array_all = []  # for all samples x lang_orders

    fwt_3_intent_array_all = []  # for all samples x lang_orders
    fwt_3_slot_array_all = []  # for all samples x lang_orders

    fwt_4_intent_array_all = []  # for all samples x lang_orders
    fwt_4_slot_array_all = []  # for all samples x lang_orders

    fwt_5_intent_array_all = []  # for all samples x lang_orders
    fwt_5_slot_array_all = []  # for all samples x lang_orders

    final_perf_intent_array_all = []  # for all samples x lang_orders
    final_perf_slot_array_all = []  # for all samples x lang_orders
    forget_intent_array_lang = {l: [] for l in languages}
    forget_slot_array_lang = {l: [] for l in languages}
    fwt_intent_array_lang = {l: [] for l in languages}
    fwt_slot_array_lang = {l: [] for l in languages}
    fwt_mono_intent_array_lang = {l: [] for l in languages}
    fwt_mono_slot_array_lang = {l: [] for l in languages}
    per_order_metrics = []
    for lang_order in lang_orders:
        print("------------lang_order:", lang_order)
        forget_intent_array_sub = []
        forget_slot_array_sub = []
        fwt_intent_array_sub = []
        fwt_slot_array_sub = []

        fwt_1_intent_array_sub = []
        fwt_1_slot_array_sub = []

        fwt_2_intent_array_sub = []
        fwt_2_slot_array_sub = []

        fwt_3_intent_array_sub = []
        fwt_3_slot_array_sub = []

        fwt_4_intent_array_sub = []
        fwt_4_slot_array_sub = []

        fwt_5_intent_array_sub = []
        fwt_5_slot_array_sub = []

        fwt_mono_intent_array_sub = []
        fwt_mono_slot_array_sub = []
        final_perf_intent_array_sub = []
        final_perf_slot_array_sub = []

        for iter in tqdm(range(n_iters)):
            acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_all_1, fwt_avg_all_2, fwt_avg_all_3, fwt_avg_all_4,\
            fwt_avg_all_5, fwt_avg_mono_all, forget_perf_all, final_perf_all \
                = compute_bootstrap(root_dir, lang_order, model_name, n, seed)

            for lang, val in forget_perf_all[2].items():
                forget_intent_array_lang[lang].append(val)

            for lang, val in forget_perf_all[3].items():
                forget_slot_array_lang[lang].append(val)

            for lang, val in fwt_avg_all[0].items():
                fwt_intent_array_lang[lang].append(val)

            for lang, val in fwt_avg_all[1].items():
                fwt_slot_array_lang[lang].append(val)

            for lang, val in fwt_avg_mono_all[0].items():
                fwt_mono_intent_array_lang[lang].append(val)

            for lang, val in fwt_avg_mono_all[1].items():
                fwt_mono_slot_array_lang[lang].append(val)

            results_array = [forget_perf_all[4], forget_perf_all[5],
                             fwt_avg_all[2], fwt_avg_all[3],
                             fwt_avg_all_1[2], fwt_avg_all_1[3],
                             fwt_avg_all_2[2], fwt_avg_all_2[3],
                             fwt_avg_all_3[2], fwt_avg_all_3[3],
                             fwt_avg_all_4[2], fwt_avg_all_4[3],
                             fwt_avg_all_5[2], fwt_avg_all_5[3],
                             fwt_avg_mono_all[2], fwt_avg_mono_all[3],
                             final_perf_all[0], final_perf_all[1]]

            forget_intent_array_sub.append(results_array[0])
            forget_slot_array_sub.append(results_array[1])

            fwt_intent_array_sub.append(results_array[2])
            fwt_slot_array_sub.append(results_array[3])

            fwt_1_intent_array_sub.append(results_array[4])
            fwt_1_slot_array_sub.append(results_array[5])

            fwt_2_intent_array_sub.append(results_array[6])
            fwt_2_slot_array_sub.append(results_array[7])

            fwt_3_intent_array_sub.append(results_array[8])
            fwt_3_slot_array_sub.append(results_array[9])

            fwt_4_intent_array_sub.append(results_array[10])
            fwt_4_slot_array_sub.append(results_array[11])

            fwt_5_intent_array_sub.append(results_array[12])
            fwt_5_slot_array_sub.append(results_array[13])

            fwt_mono_intent_array_sub.append(results_array[14])
            fwt_mono_slot_array_sub.append(results_array[15])

            final_perf_intent_array_sub.append(results_array[16])
            final_perf_slot_array_sub.append(results_array[17])

        # We want to compute the average over each language order
        mean_l_f_i, d_l_f_i = compute_statistical_average(forget_intent_array_sub, n_iters)
        mean_l_f_s, d_l_f_s = compute_statistical_average(forget_slot_array_sub, n_iters)

        mean_l_fwt_i, d_l_fwt_i = compute_statistical_average(fwt_intent_array_sub, n_iters)
        mean_l_fwt_s, d_l_fwt_s = compute_statistical_average(fwt_slot_array_sub, n_iters)

        ###

        mean_l_fwt_1_i, d_l_fwt_1_i = compute_statistical_average(fwt_1_intent_array_sub, n_iters)
        mean_l_fwt_1_s, d_l_fwt_1_s = compute_statistical_average(fwt_1_slot_array_sub, n_iters)

        mean_l_fwt_2_i, d_l_fwt_2_i = compute_statistical_average(fwt_2_intent_array_sub, n_iters)
        mean_l_fwt_2_s, d_l_fwt_2_s = compute_statistical_average(fwt_2_slot_array_sub, n_iters)

        mean_l_fwt_3_i, d_l_fwt_3_i = compute_statistical_average(fwt_3_intent_array_sub, n_iters)
        mean_l_fwt_3_s, d_l_fwt_3_s = compute_statistical_average(fwt_3_slot_array_sub, n_iters)

        mean_l_fwt_4_i, d_l_fwt_4_i = compute_statistical_average(fwt_4_intent_array_sub, n_iters)
        mean_l_fwt_4_s, d_l_fwt_4_s = compute_statistical_average(fwt_4_slot_array_sub, n_iters)

        mean_l_fwt_5_i, d_l_fwt_5_i = compute_statistical_average(fwt_5_intent_array_sub, n_iters)
        mean_l_fwt_5_s, d_l_fwt_5_s = compute_statistical_average(fwt_5_slot_array_sub, n_iters)

        ###

        mean_l_fwt_mono_i, d_l_fwt_mono_i = compute_statistical_average(fwt_mono_intent_array_sub, n_iters)
        mean_l_fwt_mono_s, d_l_fwt_mono_s = compute_statistical_average(fwt_mono_slot_array_sub, n_iters)

        mean_l_fp_i, d_l_fp_i = compute_statistical_average(final_perf_intent_array_sub, n_iters)
        mean_l_fp_s, d_l_fp_s = compute_statistical_average(final_perf_slot_array_sub, n_iters)

        per_order_metrics.append([model_name, lang_order,
                                  round(mean_l_f_i, 2),  round(d_l_f_i, 2),
                                  round(mean_l_f_s, 2),   round(d_l_f_s, 2),

                                  round(mean_l_fwt_i, 2),  round(d_l_fwt_i, 2),
                                  round(mean_l_fwt_s, 2),  round(d_l_fwt_s, 2),

                                  round(mean_l_fwt_1_i, 2),  round(d_l_fwt_1_i, 2),
                                  round(mean_l_fwt_1_s, 2),  round(d_l_fwt_1_s, 2),

                                  round(mean_l_fwt_2_i, 2),  round(d_l_fwt_2_i, 2),
                                  round(mean_l_fwt_2_s, 2),  round(d_l_fwt_2_s, 2),

                                  round(mean_l_fwt_3_i, 2),  round(d_l_fwt_3_i, 2),
                                  round(mean_l_fwt_3_s, 2),  round(d_l_fwt_3_s, 2),

                                  round(mean_l_fwt_4_i, 2),  round(d_l_fwt_4_i, 2),
                                  round(mean_l_fwt_4_s, 2),  round(d_l_fwt_4_s, 2),

                                  round(mean_l_fwt_5_i, 2),  round(d_l_fwt_5_i, 2),
                                  round(mean_l_fwt_5_s, 2),  round(d_l_fwt_5_s, 2),

                                  round(mean_l_fwt_mono_i, 2), round(d_l_fwt_mono_i, 2),
                                  round(mean_l_fwt_mono_s, 2), round(d_l_fwt_mono_s, 2),

                                  round(mean_l_fp_i, 2), round(d_l_fp_i, 2),
                                  round(mean_l_fp_s, 2), round(d_l_fp_s, 2)])

        forget_intent_array_all.append(mean_l_f_i)
        forget_slot_array_all.append(mean_l_f_s)

        fwt_intent_array_all.append(mean_l_fwt_i)
        fwt_slot_array_all.append(mean_l_fwt_s)

        fwt_1_intent_array_all.append(mean_l_fwt_1_i)
        fwt_1_slot_array_all.append(mean_l_fwt_1_s)

        fwt_2_intent_array_all.append(mean_l_fwt_2_i)
        fwt_2_slot_array_all.append(mean_l_fwt_2_s)

        fwt_3_intent_array_all.append(mean_l_fwt_3_i)
        fwt_3_slot_array_all.append(mean_l_fwt_3_s)

        fwt_4_intent_array_all.append(mean_l_fwt_4_i)
        fwt_4_slot_array_all.append(mean_l_fwt_4_s)

        fwt_5_intent_array_all.append(mean_l_fwt_5_i)
        fwt_5_slot_array_all.append(mean_l_fwt_5_s)

        fwt_mono_intent_array_all.append(mean_l_fwt_mono_i)
        fwt_mono_slot_array_all.append(mean_l_fwt_mono_s)

        final_perf_intent_array_all.append(mean_l_fp_i)
        final_perf_slot_array_all.append(mean_l_fp_s)

    avg_metrics = [model_name, "average_lang",
                   round(np.mean(forget_intent_array_all), 2), round(np.std(forget_intent_array_all), 2),
                   round(np.mean(forget_slot_array_all), 2), round(np.std(forget_slot_array_all), 2),

                   round(np.mean(fwt_intent_array_all), 2), round(np.std(fwt_intent_array_all), 2),
                   round(np.mean(fwt_slot_array_all), 2), round(np.std(fwt_slot_array_all), 2),

                   round(np.mean(fwt_1_intent_array_all), 2), round(np.std(fwt_1_intent_array_all), 2),
                   round(np.mean(fwt_1_slot_array_all), 2), round(np.std(fwt_1_slot_array_all), 2),

                   round(np.mean(fwt_2_intent_array_all), 2), round(np.std(fwt_2_intent_array_all), 2),
                   round(np.mean(fwt_2_slot_array_all), 2), round(np.std(fwt_2_slot_array_all), 2),

                   round(np.mean(fwt_3_intent_array_all), 2), round(np.std(fwt_3_intent_array_all), 2),
                   round(np.mean(fwt_3_slot_array_all), 2), round(np.std(fwt_3_slot_array_all), 2),

                   round(np.mean(fwt_4_intent_array_all), 2), round(np.std(fwt_4_intent_array_all), 2),
                   round(np.mean(fwt_4_slot_array_all), 2), round(np.std(fwt_4_slot_array_all), 2),

                   round(np.mean(fwt_5_intent_array_all), 2), round(np.std(fwt_5_intent_array_all), 2),
                   round(np.mean(fwt_5_slot_array_all), 2), round(np.std(fwt_5_slot_array_all), 2),

                   round(np.mean(fwt_mono_intent_array_all), 2), round(np.std(fwt_mono_intent_array_all), 2),
                   round(np.mean(fwt_mono_slot_array_all), 2), round(np.std(fwt_mono_slot_array_all), 2),

                   round(np.mean(final_perf_intent_array_all), 2), round(np.std(final_perf_intent_array_all), 2),
                   round(np.mean(final_perf_slot_array_all), 2), round(np.std(final_perf_slot_array_all), 2)]

    CIs = bootstrap.ci(fwt_intent_array_all, scipy.mean, n_samples=len(fwt_intent_array_all))
    print(CIs[0], CIs[1])

    forget_lang_string_format = []
    for lang in languages:
        # We want to compute the average over each language order
        mean_l_f_i, d_l_f_i = compute_statistical_average(forget_intent_array_lang[lang], len(forget_intent_array_lang[lang]))
        mean_l_f_s, d_l_f_s = compute_statistical_average(forget_slot_array_lang[lang], len(forget_slot_array_lang[lang]))

        forget_lang_string_format.extend([lang, " & ", round(mean_l_f_i, 2),  round(d_l_f_i, 2),  " & ",
                                          round(mean_l_f_s, 2),   round(d_l_f_s, 2), " & "])

    fwt_lang_string_format = []
    for lang in languages:
        mean_l_fwt_i, d_l_fwt_i = compute_statistical_average(fwt_intent_array_lang[lang], len(fwt_intent_array_lang[lang]))
        mean_l_fwt_s, d_l_fwt_s = compute_statistical_average(fwt_slot_array_lang[lang], len(fwt_slot_array_lang[lang]))

        fwt_lang_string_format.extend([lang, " & ",  round(mean_l_fwt_i, 2),  round(d_l_fwt_i, 2), " & ",
                                       round(mean_l_fwt_s, 2),  round(d_l_fwt_s, 2), " & "])

    fwt_mono_string_format = []
    for lang in languages:
        mean_l_fwt_mono_i, d_l_fwt_mono_i = compute_statistical_average(fwt_mono_intent_array_lang[lang], len(fwt_mono_intent_array_lang[lang]))
        mean_l_fwt_mono_s, d_l_fwt_mono_s = compute_statistical_average(fwt_mono_slot_array_lang[lang], len(fwt_mono_slot_array_lang[lang]))

        fwt_mono_string_format.extend([lang, "&",   round(mean_l_fwt_mono_i, 2), round(d_l_fwt_mono_i, 2),
                                       " & ", round(mean_l_fwt_mono_s, 2), round(d_l_fwt_mono_s, 2)])

    return per_order_metrics, avg_metrics, forget_lang_string_format, fwt_lang_string_format, fwt_mono_string_format

def multi_purpose_func(model, root_dir, out_dir, seed, languages):
    if model == "random":
        perf, conf = compute_random(root_dir, seed)
    elif model == "multi":
        perf, conf = compute_multi(root_dir, seed)
    elif model == "mono_orig":
        perf, conf = compute_mono(root_dir, seed, mode="orig")
    elif model == "mono_ada_tuned":
        perf, conf = compute_mono(root_dir, seed, mode="TUNED_BERT")
    else:
        perf, conf = compute_mono(root_dir, seed, mode="FROZEN_BERT")

    perf_summary = ""
    for lang in perf:
        perf_summary += lang + " " + str(round(perf[lang][0], 2)) + " \pm $" + str(round(conf[lang][0], 2)) + "$ &" \
                        + str(round(perf[lang][1], 2)) + " \pm $" + str(round(conf[lang][1], 2)) + "$ &"

    print("MULTI:" + perf_summary)
    print(" average :" + str(round(np.mean([perf[lang][0] for lang in languages]), 2)) + " \pm $"
          + str(round(np.mean([conf[lang][0] for lang in languages]), 2)) + "$ &"
          + str(round(np.mean([perf[lang][1] for lang in languages]), 2)) + " \pm $"
          + str(round(np.mean([conf[lang][1] for lang in languages]), 2)) + "$")

    dict_pickle = {model + "_perf": perf,
                   model + "_conf": conf}

    print("pickle:", dict_pickle)

    with open(out_dir + model+".pickle", "wb") as file:
        pickle.dump(dict_pickle, file)

    return perf, conf

def load_multi_purpose(out_dir, model):
    with open(out_dir+model+".pickle", "rb") as file:
        data_pickle = pickle.load(file)

    return data_pickle[model + "_perf"], data_pickle[model + "_conf"]

if __name__ == "__main__":
    #### CONSTANTS
    root_dir = "new_predictions"
    out_dir = "metrics/bootstrap/"
    n = 600
    n_iters = 600
    t = 1.9639  # 95%
    languages = ["de", "en", "fr", "es", "hi", "th"]

    lang_orders = ["en_de_fr_hi_es_th",
                   "th_es_hi_fr_de_en",
                   "fr_th_de_en_hi_es",
                   "hi_en_es_th_fr_de",
                   "es_hi_en_de_th_fr",
                   "de_fr_th_es_en_hi"]

    model_names = ["vanilla",
                   "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                   "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                   "multi_head_out",
                   "adapters/TUNED_BERT",
                   "adapters/FROZEN_BERT",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]

    #### RANDOM
    #### RANDOM
    # rand_perf, rand_conf = multi_purpose_func("random",
    #                                           root_dir,
    #                                           out_dir,
    #                                           seed,
    #                                           languages)

    rand_perf, rand_conf = load_multi_purpose(out_dir, "random")
    print("--------------------------------------------------------------------------------------------")
    # multi_perf, multi_conf = multi_purpose_func("multi",
    #                                             root_dir,
    #                                             out_dir,
    #                                             seed,
    #                                             languages)

    multi_perf, multi_conf = load_multi_purpose(out_dir, "multi")

    print("--------------------------------------------------------------------------------------------")
    # mono_orig_perf, mono_orig_conf = multi_purpose_func("mono_orig",
    #                                                     root_dir,
    #                                                     out_dir,
    #                                                     seed,
    #                                                     languages)

    mono_orig_perf, mono_orig_conf = load_multi_purpose(out_dir, "mono_orig")

    print("--------------------------------------------------------------------------------------------")
    # mono_ada_tuned_perf, mono_ada_tuned_conf = multi_purpose_func("mono_ada_tuned",
    #                                                               root_dir,
    #                                                               out_dir,
    #                                                               seed,
    #                                                               languages)

    mono_ada_tuned_perf, mono_ada_tuned_conf = load_multi_purpose(out_dir, "mono_ada_tuned")

    print("--------------------------------------------------------------------------------------------")
    # mono_ada_frozen_perf, mono_ada_frozen_conf = multi_purpose_func("mono_ada_frozen",
    #                                                                 root_dir,
    #                                                                 out_dir,
    #                                                                 seed,
    #                                                                 languages)

    mono_ada_frozen_perf, mono_ada_frozen_conf = load_multi_purpose(out_dir, "mono_ada_frozen")

    all_per_order_metrics = []
    all_avg_metrics = []

    for model_name in model_names:
        print("COMPUTING BOOTSTRAP FOR ", model_name)
        per_order_metrics, avg_metrics, forget_lang_string_format, fwt_lang_string_format, fwt_mono_string_format \
            = method_2(root_dir, n, n_iters, model_name, "42")

        all_per_order_metrics.append(per_order_metrics)
        all_avg_metrics.append(avg_metrics)

    print("Print Overleaf Friendly Format Summaries")

    print("AVERAGE OVER ALL ORDERS-------------------------------------------------------------------------------")
    for avg_metrics in all_avg_metrics:
        print(" & ".join(map(str, avg_metrics)))


    alias = "fwt_k"
    res_dir = os.path.join(out_dir, alias)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    with open(os.path.join(res_dir, "all_per_order_metrics_bootstrap.pickle"), "wb") as file:
        pickle.dump(all_per_order_metrics, file)

    with open(os.path.join(res_dir, "all_avg_metrics_bootstrap.pickle"), "wb") as file:
        pickle.dump(all_avg_metrics, file)


