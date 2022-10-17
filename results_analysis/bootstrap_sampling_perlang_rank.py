from sklearn.metrics import f1_score, precision_score, recall_score
from random import choices
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


def compute_statistical_average(l, n_iters):
    mean_l = np.mean(l)
    s_l = std_dev(l, mean_l, n_iters)
    d_l = t * (s_l / math.sqrt(n))
    return mean_l, d_l



def std_dev(x, mean, n):
    sum = 0.0
    for i in range(n):
        sum += ((x[i]-mean)**2)

    return sum / (n-1)


def compute_bootstrap(root_dir, lang_order, model_name, n, n_iters, seed):
    intents_metrics = [[0.0 for _ in range(6)] for _ in range(6)]
    slots_metrics = [[0.0 for _ in range(6)] for _ in range(6)]

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

            accs = []
            f1s = []

            for iter in range(n_iters):
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

                accs.append(intent_acc)
                f1s.append(slot_f1)

            mean_acc = np.mean(accs)
            s_acc = std_dev(accs, mean_acc, n_iters)
            d_acc = t * (s_acc / math.sqrt(n))

            mean_f1 = np.mean(f1s)
            s_f1 = std_dev(f1s, mean_f1, n_iters)
            d_f1 = t * (s_f1 / math.sqrt(n_iters))

            intents_metrics[i_train][j_test] = mean_acc
            slots_metrics[i_train][j_test] = mean_f1

    intents_metrics_np = np.asarray(intents_metrics)
    slots_metrics_np = np.asarray(slots_metrics)
    return intents_metrics_np, slots_metrics_np


def method_2(root_dir, n, n_iters, model_name, seed):
    per_lang_rank_intents = {lang: {rank: [] for rank in range(1, 6)} for lang in languages}
    per_lang_rank_slots = {lang: {rank: [] for rank in range(1, 6)} for lang in languages}

    per_lang_rank_intents_avg = {lang: {rank: 0.0 for rank in range(1, 6)} for lang in languages}
    per_lang_rank_slots_avg = {lang: {rank: 0.0 for rank in range(1, 6)} for lang in languages}
    for lang_order in tqdm(lang_orders):
        print("------------lang_order:", lang_order)
        lang_parts = lang_order.split("_")

        intents_metrics, slots_metrics = compute_bootstrap(root_dir, lang_order, model_name, n, n_iters, seed)

        for lang in languages:
            rank = lang_parts.index(lang)
            for r in range(rank):
                per_lang_rank_intents[lang][rank].append(intents_metrics[r][rank])
                per_lang_rank_slots[lang][rank].append(slots_metrics[r][rank])

    # Take the average per rank for each language
    for lang in per_lang_rank_intents:
        for rank in per_lang_rank_intents[lang]:
            per_lang_rank_intents_avg[lang][rank] = np.mean(per_lang_rank_intents[lang][rank])
            per_lang_rank_slots_avg[lang][rank] = np.mean(per_lang_rank_slots[lang][rank])

    return per_lang_rank_intents_avg, per_lang_rank_slots_avg


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
                   "joint_plus",
                   "adapters/TUNED_BERT",
                   "adapters/FROZEN_BERT",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]

    per_lang_rank_intents_avg_all = []
    per_lang_rank_slots_avg_all = []

    for model_name in model_names:
        print("COMPUTING BOOTSTRAP FOR ", model_name)
        per_lang_rank_intents_avg, per_lang_rank_slots_avg = method_2(root_dir, n, n_iters, model_name, "42")

        print("per_lang_rank_intents_avg:", per_lang_rank_intents_avg)

        per_lang_rank_intents_avg_all.append(per_lang_rank_intents_avg)
        per_lang_rank_slots_avg_all.append(per_lang_rank_slots_avg)

    alias = "per_lang_rank"
    res_dir = os.path.join(out_dir, alias)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    with open(os.path.join(res_dir, "intents_avg_all.pickle"), "wb") as file:
        pickle.dump(per_lang_rank_intents_avg_all, file)

    with open(os.path.join(res_dir, "slots_avg_all.pickle"), "wb") as file:
        pickle.dump(per_lang_rank_slots_avg_all, file)
