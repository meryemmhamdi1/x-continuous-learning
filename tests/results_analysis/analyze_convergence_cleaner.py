import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from summarize_metrics import (
    acc_avg,
    fwt_avg,
    fwt_avg_mono,
    bwt_avg,
    forget_avg,
    final_perf,
)

EVAL_TYPE = "test"
TASK_NAME = "MTOP"
LANG_ORDER = "en_de_hi_th"
LANGS = LANG_ORDER.split("_")

ROOT_DIR = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"
    + TASK_NAME
    + "/HyperparamSearch/BertBaseMultilingualCased/"
)

MODEL_NAMES = [
    "vanilla",
    "incrjoint",
    "er",
    "cont-mono_fifo",
    "cont-multi_fifo",
    "cont-mono_rand",
    "cont-multi_rand",
]
# Add Variants of ER_ONLY to MODEL_NAMES
MODES = ["cont-mono", "cont-multi"]
ER_STRATEGIES = ["easy", "hard", "random", "balanced", "extreme", "equal-lang"]
RAND_STRATEGIES = ["fifo", "rand"]
for mode in MODES:
    for er_strategy in ER_STRATEGIES:
        for rand_strategy in RAND_STRATEGIES:
            MODEL_NAMES.append(mode + "_eronly_" + er_strategy + "_" + rand_strategy)

model_dicts = {model_name: ("", "", "") for model_name in MODEL_NAMES}


# TODO do the same for slot filling and other metrics for NER
# TODO add more metrics (launch experiments for monolingual and random performances)
# TODO change the plots saving place
# TODO save the metrics list into excel sheet
# TODO create plots output directory

for model_name in MODEL_NAMES:
    if model_name == "vanilla":
        path_part_1 = "Baseline/ltmode-cont-mono/"
        path_part_2 = "/W_SCHEDULER_0_WARMUP/"
        rep_name = "Naive Seq FT"
    elif model_name == "er":
        path_part_1 = "Baseline/ER/ER_PROP_0.0/ltmode-cont-mono/"
        path_part_2 = "/W_SCHEDULER_0_WARMUP/"
        rep_name = "Seq FT + MER(Rand)"
    elif model_name == "incrjoint":
        path_part_1 = "Baseline/ltmode-multi-incr-cll/"
        path_part_2 = "/W_SCHEDULER_0_WARMUP/"
        rep_name = "IncJoint"
    elif model_name in [
        "cont-mono_fifo",
        "cont-multi_fifo",
        "cont-mono_rand",
        "cont-multi_rand",
    ]:
        mode, rand_strategy = model_name.split("_")
        path_part_1 = (
            "LtnScheduler/DemotePrevious/ltnmodel-ltn/"
            + rand_strategy
            + "/schedtype-main_erstrategy-easy/ltmode-"
            + mode
            + "/"
        )
        path_part_2 = "/ndecks-5/sample-epoch/update-epoch/W_SCHEDULER_0_WARMUP/"
        rep_name = "Seq FT + LTN (" + rand_strategy.upper() + ")"
    else:  # ER LTN
        mode, _, er_strategy, rand_strategy = model_name.split("_")
        path_part_1 = (
            "LtnScheduler/ER/ER_PROP_0.0/DemotePrevious/ltnmodel-ltn/"
            + rand_strategy
            + "/schedtype-er_erstrategy-"
            + er_strategy
            + "/ltmode-"
            + mode
            + "/"
        )
        path_part_2 = (
            "/ndecks-5/sample-epoch/update-epoch-everything/W_SCHEDULER_0_WARMUP/"
        )
        rep_name = (
            "Seq FT + ER (LTN) - " + rand_strategy.upper() + " - " + er_strategy.upper()
        )

    model_dicts[model_name] = (path_part_1, path_part_2, rep_name)


# with open(
#     "/home1/mmhamdi/x-continuous-learning/outputs/metrics/mono_orig.pickle", "rb"
# ) as file:
#     mono_perf = pickle.load(file)

# with open(
#     "/home1/mmhamdi/x-continuous-learning/outputs/metrics/rand_orig.pickle", "rb"
# ) as file:
#     rand_perf = pickle.load(file)

mono_perf = {
    "en": (95.73643410852714, 76.24760759872846),
    "de": (94.25190194420964, 66.60486067106715),
    "hi": (91.86088203657224, 65.2597861156883),
    "th": (91.68173598553345, 65.21982534242415),
}
rand_perf = {
    "en": (0.1823985408116735, 0.1719692790338232),
    "de": (1.4933784164553396, 0.15755581155353093),
    "hi": (0.17927572606669057, 0.16128618931729896),
    "th": (0.6871609403254972, 0.13090873843779943),
}

# lang = "en"


def compute_metrics(model_name, lang_order):
    avg_forgetting_epochs = ([], [])
    avg_final_perf_epochs = ([], [])
    avg_acc_perf_epochs = ([], [])
    avg_fwt_perf_epochs = ([], [])
    avg_fwt_mono_perf_epochs = ([], [])

    if model_name == "er-rand":
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/BertBaseMultilingualCased/MTOP/Baseline/ER_prop-0.0_TrainERMain/_ERStrategy-random/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"
    elif model_name == "ltn-er-reproduce":
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-fifo/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-easy/ERSched-er-only/Mode-cont-mono/"
            + LANG_ORDER
            + "/metrics.json"
        )
        acc_alias = "class"
        slots_alias = "tags"
    else:
        root_results_dir = (
            ROOT_DIR
            + model_dicts[model_name][0]
            + lang_order
            + model_dicts[model_name][1]
            + "metrics.json"
        )
        acc_alias = "acc"
        slots_alias = "slots"

    with open(root_results_dir) as file:
        results_json = json.load(file)

    for epoch in range(10):
        metrics = [[0.0 for _ in range(14)] for _ in range(7)]

        for i, train_lang in enumerate(LANGS):
            for j, test_lang in enumerate(LANGS):
                metrics[i][2 * j] = (
                    results_json[i]["test"][acc_alias][test_lang][epoch] * 100
                )
                metrics[i][2 * j + 1] = (
                    results_json[i]["test"][slots_alias][test_lang][epoch] * 100
                )

        metrics_np = np.asarray(metrics)

        avg_forget = forget_avg(metrics_np, LANGS, LANGS)
        avg_fp = final_perf(metrics_np, LANGS)
        avg_acc = acc_avg(metrics_np, LANGS, LANGS)
        avg_fwt = fwt_avg(metrics_np, LANGS, rand_perf)
        avg_fwt_mono = fwt_avg_mono(metrics_np, LANGS, mono_perf)

        avg_forgetting_epochs[0].append(avg_forget[4])
        avg_forgetting_epochs[1].append(avg_forget[5])
        #
        avg_final_perf_epochs[0].append(avg_fp[0])
        avg_final_perf_epochs[1].append(avg_fp[1])
        #
        avg_acc_perf_epochs[0].append(avg_acc[2])
        avg_acc_perf_epochs[1].append(avg_acc[3])
        #
        avg_fwt_perf_epochs[0].append(avg_fwt[2])
        avg_fwt_perf_epochs[1].append(avg_fwt[3])
        # #
        avg_fwt_mono_perf_epochs[0].append(avg_fwt_mono[2])
        avg_fwt_mono_perf_epochs[1].append(avg_fwt_mono[3])

    return (
        avg_forgetting_epochs,
        avg_final_perf_epochs,
        avg_acc_perf_epochs,
        avg_fwt_perf_epochs,
        avg_fwt_mono_perf_epochs,
    )


metrics_epochs_models = {
    "Forgetting": ({}, {}),
    "Final Performance": ({}, {}),
    "Accuracy": ({}, {}),
    "Transfer": ({}, {}),
    "Zero-Shot Transfer": ({}, {}),
}
all_metrics_epochs_models = ({}, {})

for model in [
    "vanilla",
    "er",
    "cont-mono_eronly_easy_fifo",
    "cont-mono_eronly_equal-lang_fifo",
    "er-rand",
    "ltn-er-reproduce",
]:
    outputs = compute_metrics(model, LANG_ORDER)
    for i in range(2):
        metrics_epochs_models["Forgetting"][i].update({model: outputs[0][i]})
        metrics_epochs_models["Final Performance"][i].update({model: outputs[1][i]})
        metrics_epochs_models["Accuracy"][i].update({model: outputs[2][i]})
        metrics_epochs_models["Transfer"][i].update({model: outputs[3][i]})
        metrics_epochs_models["Zero-Shot Transfer"][i].update({model: outputs[4][i]})
        #
        all_metrics_epochs_models[i].update({"Forgetting_" + model: outputs[0][i]})
        all_metrics_epochs_models[i].update({"Final-Perf_" + model: outputs[1][i]})
        all_metrics_epochs_models[i].update({"Accuracy_" + model: outputs[2][i]})
        all_metrics_epochs_models[i].update({"Transfer_" + model: outputs[3][i]})
        all_metrics_epochs_models[i].update({"Zero-Transfer_" + model: outputs[4][i]})


# Saving this into an excel sheet for intent and slot separately
df_intents = pd.DataFrame(all_metrics_epochs_models[0])
df_slots = pd.DataFrame(all_metrics_epochs_models[1])

df_intents.to_csv("Intents_" + LANG_ORDER.upper() + ".csv")
df_slots.to_csv("Slots_" + LANG_ORDER.upper() + ".csv")

for metric_name in [
    "Forgetting",
    "Final Performance",
    "Accuracy",
    "Transfer",
    "Zero-Shot Transfer",
]:
    print(
        metric_name + " vanilla:",
        metrics_epochs_models[metric_name][0]["vanilla"],
        " er-rand:",
        metrics_epochs_models[metric_name][0]["er-rand"],
    )


plot_save_dir = "/home1/mmhamdi/x-continuous-learning_new/outputs/Plots/spacedrepetition/Test/MTOP/cont_new/"

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
        # plt.title(EVAL_TYPE+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average " + metric_name + " over " + EVAL_TYPE + " languages")
        plt.legend(loc="lower right")
        plt.savefig(
            plot_save_dir
            + LANG_ORDER
            + "/"
            + task
            + "/"
            + lang
            + "_"
            + metric_name
            + ".png"
        )
