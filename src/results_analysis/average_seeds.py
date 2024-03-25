from sklearn.metrics import f1_score, precision_score, recall_score
import random
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf
from scipy import stats
import scikits.bootstrap as bootstrap
import scipy
import pickle
import os

random.seed(42)
np.random.seed(42)


def compute_statistical_average(l, n_iters):
    mean_l = np.mean(l)
    s_l = std_dev(l, mean_l, n_iters)
    d_l = t * (s_l / math.sqrt(n))
    return mean_l, d_l


def compute_random(seed, root_dir):
    print("Computing Bootstrap for RANDOM SEED_"+seed)
    rand_perf = {lang: [0.0, 0.0] for lang in languages}
    prediction_dir = root_dir+"/random/SEED_"+seed+"/"
    for i_test, test_lang in enumerate(languages):
        with open(prediction_dir + "initial_perf_"+test_lang + ".txt", "r") as file:
            results = file.read().splitlines()

        intent_corrects = 0

        intents_true = []
        intents_pred = []

        slots_true = []
        slots_pred = []

        n = len(results)
        for i in range(n):
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

        rand_perf[test_lang][0] = intent_accuracy
        rand_perf[test_lang][1] = slot_f1

    return rand_perf


def compute_multi(seed, root_dir):
    print("Computing Bootstrap for MULTI SEED_"+ seed)
    multi_perf = {lang: [0.0, 0.0] for lang in languages}
    prediction_dir = root_dir+"/multi/SEED_"+seed+"/"
    for i_test, test_lang in enumerate(languages):
        with open(prediction_dir + "End_test_perf-train_de-en-es-fr-hi-th-test_"+test_lang, "r") as file:
            results = file.read().splitlines()

        intent_corrects = 0

        intents_true = []
        intents_pred = []

        slots_true = []
        slots_pred = []

        n = len(results)
        for i in range(n):
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


        multi_perf[test_lang][0] = intent_accuracy
        multi_perf[test_lang][1] = slot_f1

    return multi_perf


def compute_mono(root_dir, seed, mode="orig"):
    print("Computing Bootstrap for MONO "+mode + " SEED_" + seed)
    mono_perf = {lang: [0.0, 0.0] for lang in languages}
    for i_train, train_lang in enumerate(languages):
        if mode != "orig":
            prediction_dir = root_dir+"/mono/adapters/" + train_lang +  "/" + mode + "/SEED_"+seed+"/"
        else:
            prediction_dir = root_dir+"/mono/vanilla/" + train_lang + "/SEED_"+seed+"/"

        with open(prediction_dir + "End_test_perf-train_"+train_lang+"-test_"+train_lang, "r") as file:
            results = file.read().splitlines()

        intent_corrects = 0

        intents_true = []
        intents_pred = []

        slots_true = []
        slots_pred = []

        n = len(results)
        for i in range(n):
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

        mono_perf[train_lang][0] = intent_accuracy
        mono_perf[train_lang][1] = slot_f1

    return mono_perf


def std_dev(x, mean, n):
    sum = 0.0
    for i in range(n):
        sum += ((x[i]-mean)**2)

    return sum / (n-1)


def compute_bootstrap(root_dir, lang_order, model_name, seed):
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
            if model_name == "joint_plus":
                file_name = root_dir+"/"+lang_order+"/"+model_name+"/SEED_"+str(seed)+"/End_test_perf-train_"+"-".join(acc_lang)+"-test_"+test_lang
            else:
                file_name = root_dir+"/"+lang_order+"/"+model_name+"/SEED_"+str(seed)+"/End_test_perf-train_"+train_lang+"-test_"+test_lang

            with open(file_name, "r") as file:
                results = file.read().splitlines()

            intent_corrects = 0

            intents_true = []
            intents_pred = []

            slots_true = []
            slots_pred = []

            n = len(results)
            for i in range(n):
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

            intent_acc = float(intent_corrects)*100 / n
            slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

            all_metrics[i_train][2*j_test] = intent_acc
            all_metrics[i_train][2*j_test+1] = slot_f1

    all_metrics_np = np.asarray(all_metrics)
    acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all = \
        acc_avg(all_metrics_np, lang_list, lang_list), \
        bwt_avg(all_metrics_np, lang_list, lang_list), \
        fwt_avg(all_metrics_np, lang_list, rand_perf), \
        fwt_avg_mono(all_metrics_np, lang_list, mono_perf), \
        forget_avg(all_metrics_np, lang_list, lang_list), \
        final_perf(all_metrics_np, lang_list)
    return acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all


def method_2(root_dir, n, seed, n_iters, model_name):
    forget_intent_array_all = []  # for all samples x lang_orders
    forget_slot_array_all = []  # for all samples x lang_orders
    fwt_intent_array_all = []  # for all samples x lang_orders
    fwt_slot_array_all = []  # for all samples x lang_orders
    fwt_mono_intent_array_all = []  # for all samples x lang_orders
    fwt_mono_slot_array_all = []  # for all samples x lang_orders
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
        fwt_mono_intent_array_sub = []
        fwt_mono_slot_array_sub = []
        final_perf_intent_array_sub = []
        final_perf_slot_array_sub = []

        for iter in tqdm(range(n_iters)):
            acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all \
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
                             fwt_avg_mono_all[2], fwt_avg_mono_all[3],
                             final_perf_all[0], final_perf_all[1]]

            forget_intent_array_sub.append(results_array[0])
            forget_slot_array_sub.append(results_array[1])

            fwt_intent_array_sub.append(results_array[2])
            fwt_slot_array_sub.append(results_array[3])

            fwt_mono_intent_array_sub.append(results_array[4])
            fwt_mono_slot_array_sub.append(results_array[5])

            final_perf_intent_array_sub.append(results_array[6])
            final_perf_slot_array_sub.append(results_array[7])

        # We want to compute the average over each language order
        mean_l_f_i, d_l_f_i = compute_statistical_average(forget_intent_array_sub, n_iters)
        mean_l_f_s, d_l_f_s = compute_statistical_average(forget_slot_array_sub, n_iters)

        mean_l_fwt_i, d_l_fwt_i = compute_statistical_average(fwt_intent_array_sub, n_iters)
        mean_l_fwt_s, d_l_fwt_s = compute_statistical_average(fwt_slot_array_sub, n_iters)

        mean_l_fwt_mono_i, d_l_fwt_mono_i = compute_statistical_average(fwt_mono_intent_array_sub, n_iters)
        mean_l_fwt_mono_s, d_l_fwt_mono_s = compute_statistical_average(fwt_mono_slot_array_sub, n_iters)

        mean_l_fp_i, d_l_fp_i = compute_statistical_average(final_perf_intent_array_sub, n_iters)
        mean_l_fp_s, d_l_fp_s = compute_statistical_average(final_perf_slot_array_sub, n_iters)

        per_order_metrics.append([model_name, lang_order,
                                  round(mean_l_f_i, 2),  round(d_l_f_i, 2),
                                  round(mean_l_f_s, 2),   round(d_l_f_s, 2),

                                  round(mean_l_fwt_i, 2),  round(d_l_fwt_i, 2),
                                  round(mean_l_fwt_s, 2),  round(d_l_fwt_s, 2),

                                  round(mean_l_fwt_mono_i, 2), round(d_l_fwt_mono_i, 2),
                                  round(mean_l_fwt_mono_s, 2), round(d_l_fwt_mono_s, 2),

                                  round(mean_l_fp_i, 2), round(d_l_fp_i, 2),
                                  round(mean_l_fp_s, 2), round(d_l_fp_s, 2)])

        forget_intent_array_all.append(mean_l_f_i)
        forget_slot_array_all.append(mean_l_f_s)

        fwt_intent_array_all.append(mean_l_fwt_i)
        fwt_slot_array_all.append(mean_l_fwt_s)

        fwt_mono_intent_array_all.append(mean_l_fwt_mono_i)
        fwt_mono_slot_array_all.append(mean_l_fwt_mono_s)

        final_perf_intent_array_all.append(mean_l_fp_i)
        final_perf_slot_array_all.append(mean_l_fp_s)

    avg_metrics = [model_name, "average_lang",
                   round(np.mean(forget_intent_array_all), 2), round(np.std(forget_intent_array_all), 2),
                   round(np.mean(forget_slot_array_all), 2), round(np.std(forget_slot_array_all), 2),

                   round(np.mean(fwt_intent_array_all), 2), round(np.std(fwt_intent_array_all), 2),
                   round(np.mean(fwt_slot_array_all), 2), round(np.std(fwt_slot_array_all), 2),

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


def multi_purpose_func(model, SEEDS, root_dir, out_dir, languages):
    average_intents = []
    average_slots = []
    perfs = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        if model == "multi":
            perfs[seed] = compute_multi(seed, root_dir)
        elif model == "mono_orig":
            perfs[seed] = compute_mono(root_dir, seed, mode="orig")
        elif model == "mono_ada_tuned":
            perfs[seed] = compute_mono(root_dir, seed, mode="TUNED_BERT")
        else:
            perfs[seed] = compute_mono(root_dir, seed, mode="FROZEN_BERT")

        avg_intent = np.mean([perfs[seed][lang][0] for lang in languages])
        avg_slot = np.mean([perfs[seed][lang][1] for lang in languages])

        print("Model:", model, " seed: ", seed, " perfs[seed]:", perfs[seed],
              " avg_intent: ", round(avg_intent, 2), " avg_slot: ", round(avg_slot, 2))

        average_intents.append(avg_intent)
        average_slots.append(avg_slot)

    print(" Average for all seeds :"
          , str(round(np.mean(average_intents), 2)), " \pm $", str(round(np.std(average_intents), 2))
          , " ", str(round(np.mean(average_slots), 2)), " \pm $", str(round(np.std(average_slots), 2)))

    perf = {lang: [np.mean([perfs[seed][lang][0] for seed in SEEDS]),
                   np.mean([perfs[seed][lang][1] for seed in SEEDS])] for lang in languages}

    conf = {lang: [np.std([perfs[seed][lang][0] for seed in SEEDS]),
                   np.std([perfs[seed][lang][1] for seed in SEEDS])] for lang in languages}

    dict_pickle = {model + "_perf": perf,
                   model + "_conf": conf}

    print("pickle:", dict_pickle)

    with open(out_dir + model+".pickle", "wb") as file:
        pickle.dump(dict_pickle, file)

    return perf, conf

def compute_perf_conf_lang_order_4_parts(metrics, SEEDS):
    avg = {lang: np.mean(metrics[seed][0][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][0].keys()}, \
          {lang: np.mean(metrics[seed][1][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][1].keys()}, \
          np.mean([metrics[seed][2] for seed in range(len(SEEDS))]), \
          np.mean([metrics[seed][3] for seed in range(len(SEEDS))])

    conf = {lang: np.std(metrics[seed][0][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][0].keys()}, \
           {lang: np.std(metrics[seed][1][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][1].keys()}, \
           np.std([metrics[seed][2] for seed in range(len(SEEDS))]), \
           np.std([metrics[seed][3] for seed in range(len(SEEDS))])

    return avg, conf

def compute_perf_conf_lang_order_6_parts(metrics, SEEDS):
    avg = {lang: np.mean(metrics[seed][0][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][0].keys()}, \
          {lang: np.mean(metrics[seed][1][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][1].keys()}, \
          {lang: np.mean(metrics[seed][2][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][2].keys()}, \
          {lang: np.mean(metrics[seed][3][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][3].keys()}, \
          np.mean([metrics[seed][4] for seed in range(len(SEEDS))]), \
          np.mean([metrics[seed][5] for seed in range(len(SEEDS))])

    conf = {lang: np.std(metrics[seed][0][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][0].keys()}, \
           {lang: np.std(metrics[seed][1][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][1].keys()}, \
           {lang: np.std(metrics[seed][2][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][2].keys()}, \
           {lang: np.std(metrics[seed][3][lang]) for seed in range(len(SEEDS)) for lang in metrics[seed][3].keys()}, \
           np.std([metrics[seed][4] for seed in range(len(SEEDS))]), \
           np.std([metrics[seed][5] for seed in range(len(SEEDS))])

    return avg, conf

def compute_perf_conf_lang_order_2_parts(metrics, SEEDS):
    avg = np.mean([metrics[seed][0] for seed in range(len(SEEDS))]), \
          np.mean([metrics[seed][1] for seed in range(len(SEEDS))])

    conf = np.std([metrics[seed][0] for seed in range(len(SEEDS))]), \
           np.std([metrics[seed][1] for seed in range(len(SEEDS))])

    return avg, conf


if __name__ == "__main__":
    #### CONSTANTS
    root_dir = "new_predictions"
    out_dir = "metrics/seeds/"
    SEEDS = ["40", "35", "42"]

    languages = ["de", "en", "fr", "es", "hi", "th"]
    lang_orders = ["en_de_fr_hi_es_th",
                   "th_es_hi_fr_de_en",
                   "fr_th_de_en_hi_es",
                   "hi_en_es_th_fr_de",
                   "es_hi_en_de_th_fr",
                   "de_fr_th_es_en_hi"]

    #### RANDOM
    rand_perfs = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        rand_perfs[seed] = compute_random(seed, root_dir)
        print("seed: ", seed, " rand_perfs[seed]:", rand_perfs[seed])

    rand_perf = {lang: [np.mean([rand_perfs[seed][lang][0] for seed in SEEDS]),
                        np.mean([rand_perfs[seed][lang][1] for seed in SEEDS])] for lang in languages}

    rand_conf = {lang: [np.std([rand_perfs[seed][lang][0] for seed in SEEDS]),
                        np.std([rand_perfs[seed][lang][1] for seed in SEEDS])] for lang in languages}

    rand_pickle = {"rand_perf": rand_perf,
                   "rand_conf": rand_conf}

    print("rand_pickle:", rand_pickle)

    with open(out_dir + "rand.pickle", "wb") as file:
        pickle.dump(rand_pickle, file)

    print("----------------------------------------------------------------------------------------------------------")

    multi_perf, multi_conf = multi_purpose_func("multi",
                                                SEEDS,
                                                root_dir,
                                                out_dir,
                                                languages)

    print("----------------------------------------------------------------------------------------------------------")

    mono_orig_perf, mono_orig_conf = multi_purpose_func("mono_orig",
                                                        SEEDS,
                                                        root_dir,
                                                        out_dir,
                                                        languages)

    print("----------------------------------------------------------------------------------------------------------")

    mono_ada_tuned_perf, mono_ada_tuned_conf = multi_purpose_func("mono_ada_tuned",
                                                                  SEEDS,
                                                                  root_dir,
                                                                  out_dir,
                                                                  languages)

    print("----------------------------------------------------------------------------------------------------------")

    mono_ada_frozen_perf, mono_ada_frozen_conf = multi_purpose_func("mono_ada_frozen",
                                                                    SEEDS,
                                                                    root_dir,
                                                                    out_dir,
                                                                    languages)

    print("----------------------------------------------------------------------------------------------------------")

    model_names = ["vanilla",
                   #"adapters/FROZEN_BERT", # TODO NOT FULLY DONE
                   "adapters/TUNED_BERT",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "joint_plus",
                   "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-rep_memsz-6000_type-reservoir_sample-random_k-16",
                   "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                   # "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8", ###### NOT DONE YET
                   "multi_head_out"]#,
                   # ]

    # MISSING th_es_hi_fr_de_en adapters FROZEN SEED 35 TODO NOT DONE


    alias = "all_except_adaptersfrozen-joint-multienc8"

    all_per_order_metrics = []
    all_avg_metrics = []
    for model_name in tqdm(model_names):
        per_order_metrics = []
        avg_seeds_acc_avg_all = []
        avg_seeds_bwt_avg_all = []
        avg_seeds_fwt_avg_all = []
        avg_seeds_fwt_avg_mono_all = []
        avg_seeds_forget_perf_all = []
        avg_seeds_final_perf_all = []
        for lang_order in lang_orders:
            lang_order_avg_seeds_acc_avg_all = []
            lang_order_avg_seeds_bwt_avg_all = []
            lang_order_avg_seeds_fwt_avg_all = []
            lang_order_avg_seeds_fwt_avg_mono_all = []
            lang_order_avg_seeds_forget_perf_all = []
            lang_order_avg_seeds_final_perf_all = []
            for seed in SEEDS:
                acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all = \
                    compute_bootstrap(root_dir, lang_order, model_name, seed)

                lang_order_avg_seeds_acc_avg_all.append(acc_avg_all)
                lang_order_avg_seeds_bwt_avg_all.append(bwt_avg_all)
                lang_order_avg_seeds_fwt_avg_all.append(fwt_avg_all)
                lang_order_avg_seeds_fwt_avg_mono_all.append(fwt_avg_mono_all)
                lang_order_avg_seeds_forget_perf_all.append(forget_perf_all)
                lang_order_avg_seeds_final_perf_all.append(final_perf_all)

            ## Take the average and standard deviation for all seeds for that language order
            lang_order_avg_seeds_acc_avg = compute_perf_conf_lang_order_4_parts(lang_order_avg_seeds_acc_avg_all,
                                                                                SEEDS)
            lang_order_avg_seeds_bwt_avg = compute_perf_conf_lang_order_4_parts(lang_order_avg_seeds_bwt_avg_all,
                                                                                SEEDS)
            lang_order_avg_seeds_fwt_avg = compute_perf_conf_lang_order_4_parts(lang_order_avg_seeds_fwt_avg_all,
                                                                                SEEDS)
            lang_order_avg_seeds_fwt_mono_avg = compute_perf_conf_lang_order_4_parts(lang_order_avg_seeds_fwt_avg_mono_all,
                                                                                     SEEDS)
            lang_order_avg_seeds_forget_perf_avg = compute_perf_conf_lang_order_6_parts(lang_order_avg_seeds_forget_perf_all,
                                                                                        SEEDS)
            lang_order_avg_seeds_final_perf_avg = compute_perf_conf_lang_order_2_parts(lang_order_avg_seeds_final_perf_all,
                                                                                       SEEDS)


            per_order_metrics.append([model_name, lang_order,

                                      round(lang_order_avg_seeds_forget_perf_avg[0][4], 2),
                                      round(lang_order_avg_seeds_forget_perf_avg[1][4], 2),

                                      round(lang_order_avg_seeds_forget_perf_avg[0][5], 2),
                                      round(lang_order_avg_seeds_forget_perf_avg[1][5], 2),

                                      round(lang_order_avg_seeds_fwt_avg[0][2], 2),
                                      round(lang_order_avg_seeds_fwt_avg[1][2], 2),

                                      round(lang_order_avg_seeds_fwt_avg[0][3], 2),
                                      round(lang_order_avg_seeds_fwt_avg[1][3], 2),

                                      round(lang_order_avg_seeds_fwt_mono_avg[0][2], 2),
                                      round(lang_order_avg_seeds_fwt_mono_avg[1][2], 2),

                                      round(lang_order_avg_seeds_fwt_mono_avg[0][3], 2),
                                      round(lang_order_avg_seeds_fwt_mono_avg[1][3], 2),

                                      round(lang_order_avg_seeds_final_perf_avg[0][0], 2),
                                      round(lang_order_avg_seeds_final_perf_avg[1][0], 2),

                                      round(lang_order_avg_seeds_final_perf_avg[0][1], 2),
                                      round(lang_order_avg_seeds_final_perf_avg[1][1], 2),])

            avg_seeds_acc_avg_all.append(lang_order_avg_seeds_acc_avg[0])
            avg_seeds_bwt_avg_all.append(lang_order_avg_seeds_bwt_avg[0])
            avg_seeds_fwt_avg_all.append(lang_order_avg_seeds_fwt_avg[0])
            avg_seeds_fwt_avg_mono_all.append(lang_order_avg_seeds_fwt_mono_avg[0])
            avg_seeds_forget_perf_all.append(lang_order_avg_seeds_forget_perf_avg[0])
            avg_seeds_final_perf_all.append(lang_order_avg_seeds_final_perf_avg[0])

        all_per_order_metrics.append(per_order_metrics)
        avg_seeds_acc_avg = compute_perf_conf_lang_order_4_parts(avg_seeds_acc_avg_all,
                                                                 SEEDS)

        avg_seeds_bwt_avg = compute_perf_conf_lang_order_4_parts(avg_seeds_bwt_avg_all,
                                                                 SEEDS)

        seeds_fwt_avg = compute_perf_conf_lang_order_4_parts(avg_seeds_fwt_avg_all,
                                                             SEEDS)

        seeds_fwt_mono_avg = compute_perf_conf_lang_order_4_parts(avg_seeds_fwt_avg_mono_all,
                                                                  SEEDS)

        seeds_forget_perf_avg = compute_perf_conf_lang_order_6_parts(avg_seeds_forget_perf_all,
                                                                     SEEDS)

        seeds_final_perf_avg = compute_perf_conf_lang_order_2_parts(avg_seeds_final_perf_all,
                                                                    SEEDS)

        avg_metrics = [model_name, "average_lang",
                       round(seeds_forget_perf_avg[0][4], 2), round(seeds_forget_perf_avg[1][4], 2),
                       round(seeds_forget_perf_avg[0][5], 2), round(seeds_forget_perf_avg[1][5], 2),

                       round(seeds_fwt_avg[0][2], 2), round(seeds_fwt_avg[1][2], 2),
                       round(seeds_fwt_avg[0][3], 2), round(seeds_fwt_avg[1][3], 2),

                       round(seeds_fwt_mono_avg[0][2], 2), round(seeds_fwt_mono_avg[1][2], 2),
                       round(seeds_fwt_mono_avg[0][3], 2), round(seeds_fwt_mono_avg[1][3], 2),

                       round(seeds_final_perf_avg[0][0], 2), round(seeds_final_perf_avg[1][0], 2),
                       round(seeds_final_perf_avg[0][1], 2), round(seeds_final_perf_avg[1][1], 2)]

        all_avg_metrics.append(avg_metrics)

    # TODO missing files in vanilla
    res_dir = os.path.join(out_dir, alias)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    with open(os.path.join(res_dir, "all_per_order_metrics_bootstrap.pickle"), "wb") as file:
        pickle.dump(all_per_order_metrics, file)

    with open(os.path.join(res_dir, "all_avg_metrics_bootstrap.pickle"), "wb") as file:
        pickle.dump(all_avg_metrics, file)





