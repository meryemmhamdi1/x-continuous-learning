from sklearn.metrics import f1_score, precision_score, recall_score
from random import choices
import numpy as np
import math
from tqdm import tqdm

languages = ["en", "de", "fr", "es", "hi", "th"]


def std_dev(x, mean, n):
    sum = 0.0
    for i in range(n):
        sum += ((x[i]-mean)**2)

    return sum / (n-1)


def compute_bootstrap(root_dir, model_dir, n, n_iters, t):
    option = model_dir
    for train_lang in languages:
        for test_lang in languages:
            with open(root_dir+option+"End_test_perf-train_"+train_lang+ \
                      "-test_"+test_lang, "r") as file:
                results = file.read().splitlines()

            accs = []
            f1s = []

            for iter in tqdm(range(n_iters)):
                intent_corrects = 0

                intents_true = []
                intents_pred = []

                slots_true = []
                slots_pred = []

                sampled_results = choices(results, k=n)
                for i, sent in enumerate(sampled_results):
                    parts = sent.split("\t")
                    assert len(parts) == 5
                    sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

                    intents_true.append(intent_true)
                    intents_pred.append(intent_pred)

                    #####
                    slot_true_list = slot_true.split("\t")
                    slot_pred_list = slot_pred.split("\t")

                    if len(slot_true_list) != len(slot_pred_list):
                        print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                              " slot_true:", slot_true, " slot_pred:", slot_pred)

                    assert len(slot_true_list) == len(slot_pred_list)

                    slots_true.extend(slot_true_list)
                    slots_pred.extend(slot_pred_list)

                    intent_corrects += int(intent_pred == intent_true)

                intent_accuracy = float(intent_corrects) / n
                # print("slots_true:", slots_true)
                # print("slots_pred:", slots_pred)
                slot_f1 = f1_score(slots_true, slots_pred, average="macro")

                accs.append(intent_accuracy)
                f1s.append(slot_f1)

            mean_acc = np.mean(accs)
            s_acc = std_dev(accs, mean_acc, n_iters)
            d_acc = t * (s_acc / math.sqrt(n))

            mean_f1 = np.mean(f1s)
            s_f1 = std_dev(f1s, mean_f1, n_iters)
            d_f1 = t * (s_f1 / math.sqrt(n_iters))

            print("mean_acc:", mean_acc, " s_acc:", s_acc, " d_acc:", d_acc,
                  "mean_f1:", mean_f1, " s_f1:", s_f1, " d_f1:", d_f1)


if __name__ == "__main__":
    root_dir = ""
    model_dir = ""
    n = 600
    n_iters = 600
    t = 1.9639 # 95%

    compute_bootstrap(root_dir, model_dir, n, n_iters, t)
