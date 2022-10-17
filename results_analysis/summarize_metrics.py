import pickle
import pandas
import csv
import numpy as np


dict_lang = {"en": 0, "de": 1, "fr": 2, "hi": 3, "es": 4, "th": 5}


def swap(metrics, lang):
    li, ls = dict_lang[lang]*2, (dict_lang[lang]*2) + 1
    j = 0
    for i in range(dict_lang[lang]):
        f, l = li, j
        metrics[:, [f, l]] = metrics[:, [l, f]]
        ###
        j += 1
        f, l = ls, j
        metrics[:, [f, l]] = metrics[:, [l, f]]

        j += 1

    return metrics


def flatten(t):
    return [item for sublist in t for item in sublist]


def acc_avg(metrics, order, eff_order):
    sum_intent_acc = {el: 0.0 for el in order}
    sum_slot_f1 = {el: 0.0 for el in order}

    for i_test, el_test in enumerate(order): ## the order in training stream
        for i_train, el_train in enumerate(eff_order): ## the order in testing stream
            sum_intent_acc[el_test] += metrics[i_train][i_test*2]
            sum_slot_f1[el_test] += metrics[i_train][i_test*2+1]

    len_eff_order = len(eff_order)
    if np.mean([metrics[len(eff_order)-1][i_test*2] for i_test, lang_test in enumerate(eff_order)]) == 0:
        len_eff_order -= 1

    # Computing the average over training stream
    avg_intent_acc_el = {el: sum_intent_acc[el]/len_eff_order for el in sum_intent_acc}
    avg_slot_f1_el = {el: sum_slot_f1[el]/len_eff_order for el in sum_slot_f1}

    # Computing the average over all elements (could be languages or subtasks)
    avg_intent_acc = np.mean([avg_intent_acc_el[el] for el in order])
    avg_slot_f1 = np.mean([avg_slot_f1_el[el] for el in order])

    return avg_intent_acc_el, avg_slot_f1_el, avg_intent_acc, avg_slot_f1


def bwt_avg(metrics, order, eff_order):
    back_order = eff_order[1:]

    avg_intent_acc_el = {el: 0.0 for el in order}
    avg_slot_f1_el = {el: 0.0 for el in order}

    for i_train, el_train in enumerate(eff_order):
        if i_train == 0:
            continue
        for i_test, el_test in enumerate(order[:i_train]):
            avg_intent_acc_el[el_train] += (metrics[i_train][i_test*2] - metrics[i_test][i_test*2])
            avg_slot_f1_el[el_train] += (metrics[i_train][i_test*2+1] - metrics[i_test][i_test*2+1])

    # Computing the average
    j = 1
    for el in back_order:
        avg_intent_acc_el[el] = avg_intent_acc_el[el] / j
        avg_slot_f1_el[el] = avg_slot_f1_el[el] / j

        j += 1

    # Computing the average over all elements (could be languages or subtasks)
    avg_intent_acc = np.mean([avg_intent_acc_el[el] for el in back_order])
    avg_slot_f1 = np.mean([avg_slot_f1_el[el] for el in back_order])

    return avg_intent_acc_el, avg_slot_f1_el, avg_intent_acc, avg_slot_f1


def compute_max_forget(diff_list):
    return np.max(diff_list)


def forget_avg(metrics, order, eff_order):
    back_order = eff_order[1:]
    avg_intent_acc_train = {lang: 0.0 for lang in order}
    avg_slot_f1_train = {lang: 0.0 for lang in order}

    avg_intent_acc_test = {lang: 0.0 for lang in order}
    avg_slot_f1_test = {lang: 0.0 for lang in order}

    for i_train, el_train in enumerate(eff_order):
        if i_train == 0:
            continue
        for i_test, lang_test in enumerate(order[:i_train]):
            curr_intent_perf = metrics[i_train][i_test*2]
            curr_slot_perf = metrics[i_train][i_test*2+1]

            max_forget_intent \
                = compute_max_forget([metrics[i_prev][i_test*2]-curr_intent_perf for i_prev in range(i_train)])

            max_forget_slot \
                = compute_max_forget([metrics[i_prev][i_test*2+1]-curr_slot_perf for i_prev in range(i_train)])

            avg_intent_acc_test[lang_test] += max_forget_intent
            avg_slot_f1_test[lang_test] += max_forget_slot

            avg_intent_acc_train[el_train] += max_forget_intent
            avg_slot_f1_train[el_train] += max_forget_slot

    j = 1
    for lang in back_order:
        avg_intent_acc_train[lang] = avg_intent_acc_train[lang] / j
        avg_slot_f1_train[lang] = avg_slot_f1_train[lang] / j
        j += 1

    j = len(order[:-1])
    for lang in order[:-1]:
        avg_intent_acc_test[lang] = avg_intent_acc_test[lang] / j
        avg_slot_f1_test[lang] = avg_slot_f1_test[lang] / j
        j -= 1

    avg_intent_acc = np.mean([avg_intent_acc_train[lang] for lang in back_order])
    avg_slot_f1 = np.mean([avg_slot_f1_train[lang] for lang in back_order])

    return avg_intent_acc_train, avg_slot_f1_train, avg_intent_acc_test, avg_slot_f1_test, avg_intent_acc, avg_slot_f1


def fwt_avg(metrics,
            order,
            random_perf):

    fwt_order = order[1:]  # we don't look at forward transfer effect on the first element
    avg_intent_acc_el = {el: 0.0 for el in fwt_order}
    avg_slot_f1_el = {el: 0.0 for el in fwt_order}

    for i_test, el_test in enumerate(order):  # same as eff_order (but need to keep the original order of indices to match with the metrics results matrix in random_perf)
        if i_test == 0:
            continue
        for i_train, el_train in enumerate(order[:i_test]):
            avg_intent_acc_el[el_test] += (metrics[i_train][i_test*2] - random_perf[el_test][0])
            avg_slot_f1_el[el_test] += (metrics[i_train][i_test*2+1] - random_perf[el_test][1])

    j = 1
    for el in fwt_order:  # 1 2 3 4 5
        avg_intent_acc_el[el] = avg_intent_acc_el[el] / j
        avg_slot_f1_el[el] = avg_slot_f1_el[el] / j

        j += 1

    avg_intent_acc = np.mean([avg_intent_acc_el[el] for el in fwt_order])
    avg_slot_f1 = np.mean([avg_slot_f1_el[el] for el in fwt_order])

    return avg_intent_acc_el, avg_slot_f1_el, avg_intent_acc, avg_slot_f1


def fwt_avg_k(metrics,
              order,
              random_perf,
              k):
    fwt_order = order[1:]  # we don't look at the forward transfer effect on the first element
    avg_intent_acc_test = {lang: 0.0 for lang in fwt_order}
    avg_slot_f1_test = {lang: 0.0 for lang in fwt_order}

    sum_intent_acc = 0.0
    sum_slot_f1 = 0.0

    for i_train, lang_train in enumerate(order):  # same as eff_order (but need to keep the original order of indices to match with the metrics results matrix in random_perf)
        if i_train + k >= len(order):
            continue
        lang_test = order[i_train+k]
        sum_intent_acc += (metrics[i_train][(i_train+k)*2] - random_perf[lang_test][0])
        sum_slot_f1 += (metrics[i_train][(i_train+k)*2+1] - random_perf[lang_test][1])
        avg_intent_acc_test[lang_test] += (metrics[i_train][(i_train+k)*2] - random_perf[lang_test][0])
        avg_slot_f1_test[lang_test] += (metrics[i_train][(i_train+k)*2+1] - random_perf[lang_test][1])

    avg_intent_acc = sum_intent_acc / (len(order) - k)
    avg_slot_f1 = sum_slot_f1 / (len(order) - k)

    return avg_intent_acc_test, avg_slot_f1_test, avg_intent_acc, avg_slot_f1

def fwt_avg_mono(metrics,
                 order,
                 mono_perf):

    fwt_order = order[1:]  # we don't look at forward transfer effect on the first language
    avg_intent_acc_el = {el: 0.0 for el in fwt_order}
    avg_slot_f1_el = {el: 0.0 for el in fwt_order}

    for i_test, el_test in enumerate(order):  # same as eff_order (but need to keep the original order of indices to match with the metrics results matrix)
        if i_test == 0:
            continue
        avg_intent_acc_el[el_test] = (metrics[i_test][i_test*2] - mono_perf[el_test][0])
        avg_slot_f1_el[el_test] = (metrics[i_test][i_test*2+1] - mono_perf[el_test][1])

    avg_intent_acc = np.mean([avg_intent_acc_el[el] for el in fwt_order])
    avg_slot_f1 = np.mean([avg_slot_f1_el[el] for el in fwt_order])

    return avg_intent_acc_el, avg_slot_f1_el, avg_intent_acc, avg_slot_f1


def final_perf(metrics, order_lang):
    intent_acc_lang = []
    slot_f1_lang = []
    end_index = len(order_lang)-1

    if np.mean([metrics[end_index][i_test*2] for i_test, lang_test in enumerate(order_lang)]) == 0:
        end_index = len(order_lang)-2
    for i_test, lang_test in enumerate(order_lang):
        intent_acc_lang.append(metrics[end_index][i_test*2])
        slot_f1_lang.append(metrics[end_index][i_test*2+1])

    avg_intent_acc = np.mean(intent_acc_lang)
    avg_slot_f1 = np.mean(slot_f1_lang)

    return avg_intent_acc, avg_slot_f1
