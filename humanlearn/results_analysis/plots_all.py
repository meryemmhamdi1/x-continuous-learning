import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


model_names = {
    "vanilla": "Vanilla",
    "ltn-er-easy": "MER $LTN_{ini}$(easy) $LTN_{er}$(repeat)",
    "ltn-er-random": "MER $LTN_{ini}$(random) $LTN_{er}$(repeat)",
    "ltn-er-hard": "MER $LTN_{ini}$(hard) $LTN_{er}$(repeat)",
    "ltn-er-easy-wiped": "MER $LTN_{ini}$(easy) $LTN_{er}$(rand)",
    "ltn-er-random-wiped": "MER $LTN_{ini}$(random) $LTN_{er}$(rand)",
    "ltn-er-hard-wiped": "MER $LTN_{ini}$(hard) $LTN_{er}$(rand)",
    "ltn-er-rand": "MER $LTN_{ini}$(random) $LTN_{er}$(rand)",
    "ltn-er-easy-rand": "MER $LTN_{ini}$(easy) $LTN_{er}$(rand)",
    "ltn-er-hard-rand": "MER $LTN_{ini}$(hard) $LTN_{er}$(rand)",
    "ltn-er-easy-3-decks": "MER $LTN_{ini}$(easy) $LTN_{er}$(repeat) Decks(3)",
    "ltn-er-easy-5-decks": "MER $LTN_{ini}$(easy) $LTN_{er}$(repeat) Decks(5)",
    "ltn-er-easy-7-decks": "MER $LTN_{ini}$(easy) $LTN_{er}$(repeat) Decks(7)",
    "ltn-er-easy-fifo-5": "MER $LTN_{ini}$(easy) $LTN_{er}$(repeat)",
    "ltn-er-hard-fifo-5": "MER $LTN_{ini}$(hard) $LTN_{er}$(repeat)",
    "ltn-er-random-fifo-5": "MER $LTN_{ini}$(random) $LTN_{er}$(repeat)",
    "ltn-er-easy-rand-5": "MER $LTN_{ini}$(easy) $LTN_{er}$(rand)",
    "ltn-er-hard-rand-5": "MER $LTN_{ini}$(hard) $LTN_{er}$(rand)",
    "ltn-er-random-rand-5": "MER $LTN_{ini}$(random) $LTN_{er}$(rand)",
}

convergence_models = [
    "vanilla",
    # "ltn-er-easy",
    "ltn-er-easy-rand",
    # "ltn-er-random",
    "ltn-er-rand",
    # "ltn-er-hard",
    "ltn-er-hard-rand",
]

wiped_models = [
    "ltn-er-easy-wiped",
    "ltn-er-random-wiped",
    "ltn-er-hard-wiped",
]


def plot_all(lang_order, task_name, ablation_name):
    load_path = (
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new/outputs/plots/spacedrepetition/Test/"
        + task_name
        + "/"
        + ablation_name
        # + "/AVERAGE_BOOTSTRAP"
        + "/"
        + lang_order
        + "/"
    )

    intents = pd.read_csv(os.path.join(load_path, "Intents.csv"))
    slots = pd.read_csv(os.path.join(load_path, "Slots.csv"))

    print(intents.head())

    save_path = (
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Plots/Metrics/"
        + task_name
        + "/"
        + ablation_name
        # + "/BOOTSTRAP"
        + "/"
        + lang_order
        + "/"
    )

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    data = {"intents": intents, "slots": slots}
    for metric_name in [
        # "Forgetting",
        "Final Performance",
    ]:  # , "Transfer", "Accuracy"]:
        for subtask in data:
            plt.figure()
            if ablation_name == "convergence":
                dict_models = {
                    model_names[model_name]: [] for model_name in convergence_models
                }
            else:
                dict_models = {
                    model_names[model_name]: [] for model_name in wiped_models
                }

            for k, v in data[subtask].items():
                if (
                    metric_name in k
                    and "Zero-Shot" not in k
                    # and "vanilla" not in k
                    and "er-rand-prop" not in k
                    and "fifo" not in k
                    # and k.split("_")[1] != "ltn-er-rand"
                ):
                    model_name = " ".join(k.split("_")[1:])
                    if metric_name == "Forgetting":
                        values = [-x for x in v]
                    else:
                        values = v
                    dict_models[model_names[model_name]] = values

            df = pd.DataFrame(dict_models)
            print(subtask)  # , metric_name)
            for model_name in dict_models:
                if "repeat" not in model_name:
                    print(model_name)
                    for l in dict_models[model_name]:
                        print(l)
            print("******************")
            df.plot()
            if metric_name == "Forgetting":
                plt.title("Average Negative Forgetting over languages")
            elif metric_name == "Final Performance":
                plt.title("Average Final Performance over languages")
            elif metric_name == "Accuracy":
                plt.title("Average Accuracy over languages")
            else:
                plt.title("Average Transfer over languages")

            plt.xlabel("# Epochs")
            if subtask == "intents":
                plt.ylabel("Accuracies")
            else:
                plt.ylabel("F1 scores")
            plt.savefig(os.path.join(save_path, metric_name + "_" + subtask))


def plot_avg(task_name, ablation_name):
    save_path = (
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Plots/Metrics/"
        + task_name
        + "/"
        + ablation_name
        + "/average/"
    )

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for lang_order in [
        "EN_DE_HI_TH",
        "TH_HI_DE_EN",
        "HI_TH_EN_DE",
        "DE_EN_TH_HI",
    ]:
        load_path = (
            "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new/outputs/plots/spacedrepetition/Test/"
            + task_name
            + "/"
            + ablation_name
            + "/"
            + lang_order
            + "/"
        )

        intents = pd.read_csv(os.path.join(load_path, "Intents.csv"))
        slots = pd.read_csv(os.path.join(load_path, "Slots.csv"))

        data = {"intents": intents, "slots": slots}
        for metric_name in ["Forgetting", "Final Performance"]:
            for subtask in data:
                plt.figure()
                if ablation_name == "convergence":
                    dict_models = {
                        model_names[model_name]: [] for model_name in convergence_models
                    }
                else:
                    dict_models = {
                        model_names[model_name]: [] for model_name in wiped_models
                    }
                for k, v in data[subtask].items():
                    if (
                        metric_name in k
                        and "Zero-Shot" not in k
                        # and "vanilla" not in k
                        and "er-rand-prop" not in k
                        and "fifo" not in k
                        # and k.split("_")[1] != "ltn-er-rand"
                    ):
                        model_name = " ".join(k.split("_")[1:])
                        if metric_name == "Forgetting":
                            values = [-x for x in v]
                        else:
                            values = v
                        dict_models[model_names[model_name]] = values

            df = pd.DataFrame(dict_models)
            print(subtask)  # , metric_name)
            for model_name in dict_models:
                if "repeat" not in model_name:
                    print(model_name)
                    for l in dict_models[model_name]:
                        print(l)
            print("******************")
            df.plot()
            if metric_name == "Forgetting":
                plt.title("Average Negative Forgetting over languages")
            elif metric_name == "Final Performance":
                plt.title("Average Final Performance over languages")
            elif metric_name == "Accuracy":
                plt.title("Average Accuracy over languages")
            else:
                plt.title("Average Transfer over languages")

            plt.xlabel("# Epochs")
            if subtask == "intents":
                plt.ylabel("Accuracies")
            else:
                plt.ylabel("F1 scores")
            plt.savefig(os.path.join(save_path, metric_name + "_" + subtask))


task_name, ablation_name = "MTOP", "convergence"
# for lang_order in [
#     "EN_DE_HI_TH",
#     "TH_HI_DE_EN",
#     "HI_TH_EN_DE",
#     "DE_EN_TH_HI",
# ]:  # ["AVERAGE_BOOTSTRAP"]:
lang_order = "DE_EN_TH_HI"
plot_all(lang_order, task_name, ablation_name)
