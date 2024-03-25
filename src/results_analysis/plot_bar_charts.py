import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from adjustText import adjust_text
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import numpy.random as rnd   # just to generate some data

from scipy.optimize import curve_fit

lang_orders_dict = {
    "en_de_fr_hi_es_th": 0,
    "de_fr_th_es_en_hi": 1,
    "fr_th_de_en_hi_es": 2,
    "hi_en_es_th_fr_de": 3,
    "es_hi_en_de_th_fr": 4,
    "th_es_hi_fr_de_en": 5,
    "en_th": 0,
    "de_hi": 1,
    "fr_es": 2,
    "hi_de": 3,
    "es_fr": 4,
    "th_en": 5
}

# two_lang_orders_dict = {
#     "en_th": 0,
#     "de_hi": 1,
#     "fr_es": 2,
#     "hi_de": 3,
#     "es_fr": 4,
#     "th_en": 5
# }

model_names_dict = {"vanilla": "Naive Seq FT",
                    "joint_plus": "Inc Joint",
                    "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8": "Lang-Spec Enc[0-8]",
                    "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool": "Lang-Spec Trans",
                    "multi_head_out": "Lang-Spec Task",
                    "adapters/TUNED_BERT": "Lang-Spec Ada(T)",
                    "adapters/FROZEN_BERT": "Lang-Spec Ada(F)",
                    "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01": "EWC-Online",
                    "er_memsz-6000_type-reservoir_sample-random_k-16": "ER",
                    "kd-logits_memsz-6000_type-reservoir_sample-random_k-16": "KD-Logit", # TODO normalize the naming this used to be ring
                    "kd-logits_memsz-6000_type-ring_sample-random_k-16": "KD-Logit", # TODO normalize the naming this used to be ring
                    "kd-rep_memsz-6000_type-reservoir_sample-random_k-16": "KD-Rep",
                    "kd-rep_memsz-6000_type-ring_sample-random_k-16": "KD-Rep"} # TODO normalize the naming this used to be ring

model_names = ["vanilla",
               "joint_plus",
               "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
               "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
               "multi_head_out",
               "adapters/TUNED_BERT",
               "adapters/FROZEN_BERT",
               "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
               "er_memsz-6000_type-reservoir_sample-random_k-16",
               "kd-logits_memsz-6000_type-reservoir_sample-random_k-16", # TODO normalize the naming this used to be ring
               "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"] # TODO normalize the naming this used to be ring

languages = ["de", "en", "fr", "es", "hi", "th"]
languages_elonged_dict = {"en": "English",
                          "de": "German",
                          "fr": "French",
                          "es": "Spanish",
                          "hi": "Hindi",
                          "th": "Thai"}

fig_width = 12
fig_height = 8
size_text = 30
# _root_dir="/Users/meryem/USCGDrive/Spring21/Research/XContLearn/"
_root_dir="/Users/meryem/Desktop/"
# plot_save_path = _root_dir+"x-continuous-learning-repro-backup/Plots/"
plot_save_path = ""


### OLD

def try_seaborn():
    # read a titanic.csv file
    # from seaborn library
    my_cmap = sns.color_palette("colorblind", 3, as_cmap=True)
    sns.palplot(my_cmap)
    plt.show()
    exit(0)
    # df = sns.load_dataset('titanic')
    # color_pal = sns.color_palette("colorblind", 6).as_hex()
    # sns.palplot(sns.color_palette("colorblind"))
    # plt.show()
    # exit()
    # colors = ','.join(color_pal)
    # print("colors:", colors)
    raw_data = {
        # cat:    A                  B                  C                    D
        'x': ['Group 1',          'Group 1',         'Group 1',           'Group 1',
              'Group 2',          'Group 2',         'Group 2',           'Group 2',
              'Group 3',          'Group 3',         'Group 3',           'Group 3',
              'Group 4',          'Group 4',         'Group 4',           'Group 4',
              'Group 5',          'Group 5',         'Group 5',           'Group 5',
              'Group 6',          'Group 6',         'Group 6',           'Group 6'],
        'y': [47.66773437098896,  47.585408826566024, 45.426437828641106, 44.955787935926836,
              47.700582993718115, 47.59796443553682,  45.38896827262796,  44.80916093973529,
              47.66563311651776,  47.476571906259835, 45.21460968763448,  44.78683755963528,
              47.248523637295705, 47.42573841363118,  45.52890109500238,  45.10243082784969,
              47.14532745960979,  47.46958795222966,  45.4804195003332,   44.97715435208194,
              46.61620129160194,  47.316775886868584, 45.053032014046366, 44.527497508033704],
        'category': ['A', 'B', 'C', 'D','A', 'B', 'C', 'D','A', 'B', 'C', 'D','A', 'B', 'C', 'D','A', 'B', 'C', 'D','A', 'B', 'C', 'D']
    }

    # who v/s fare barplot
    sns.set_palette("colorblind")
    sns.barplot(x='x', y='y', hue='category', data=raw_data, )
    # sns.barplot(x='x', y='y', data=raw_data)

    # print("df:", df)

    # Show the plot
    plt.xlabel('Language Order', fontsize=25, family="Times")
    plt.ylabel('Accuracy/F1 Scores', fontsize=25, family="Times")

    plt.show()


def plot_lang_order_bar_charts(intent_vanilla, intent_joint, slot_vanilla, slot_joint, metric):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    #plt.figure(figsize=(10, 6))
    index = np.array([0, 1])
    bar_width = 0.10
    opacity = 0.8

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 50}

    plt.rcParams['font.size'] = '16'

    #matplotlib.rc('font', **font)
    #matplotlib.rc('text', usetex=True)
    #matplotlib.rc('axes', linewidth=2)
    #matplotlib.rc('font', weight='bold')

    # plt.grid(zorder=3)
    # ax.grid(zorder=5)
    rects1 = plt.bar(index, intent_vanilla, bar_width,
                     alpha=opacity,
                     color='blue',
                     hatch="+",
                     edgecolor='grey',
                     label='Naive Seq FT Intent')

    rects2 = plt.bar(index + bar_width, slot_vanilla, bar_width,
                     alpha=opacity,
                     color='yellowgreen',
                     hatch="/",
                     edgecolor='grey',
                     label='Naive Seq FT Slot')

    rects3 = plt.bar(index + 2.2 * bar_width, intent_joint, bar_width,
                     alpha=opacity,
                     color='darkblue',
                     hatch="+",
                     edgecolor='grey',
                     label='Inc Joint Intent')

    rects4 = plt.bar(index + 3.2 * bar_width, slot_joint, bar_width,
                     alpha=opacity,
                     color='darkgreen',
                     hatch="/",
                     edgecolor='grey',
                     label='Inc Joint Slot')

    numbers = []
    positions = []
    k = 0
    for i in range(len(intent_vanilla)):
        numbers.append(round(intent_vanilla[i], 1))
        positions.append(k*bar_width)

    for i in range(len(slot_vanilla)):
        k += 1
        numbers.append(round(slot_vanilla[i], 1))
        positions.append(k*bar_width)
        k += 1.1
    for i in range(len(intent_joint)):
        numbers.append(round(intent_joint[i], 1))
        positions.append(k*bar_width)
        k += 1.1

    for i in range(len(slot_joint)):
        numbers.append(round(slot_joint[i], 1))
        positions.append(k*bar_width)
        k += 1.1

    rects = ax.patches

    # Make some labels.
    labels = [str(numbers[i]) for i in range(len(rects))]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        if float(label) < 0:
            h = height - 0.5
        else:
            h = height + 0.2
        ax.text(
            rect.get_x() + rect.get_width() / 2, h, label, ha="center", va="bottom", fontsize=23
        )

    # positions = [0.0, 0.1, 0.22, 0.33, 1.1, 1.15, 1.2, 1.25]
    # for i, v in enumerate(numbers):
    #     if numbers[i] < 0:
    #         y_ = numbers[i] - 0.3
    #     else:
    #         y_ = numbers[i] + 0.1
    #     plt.text(x=positions[i]-0.05, y=y_, s=str(numbers[i]), size=20)

    plt.xlabel('Language Order', fontsize=25, family="Times")
    plt.ylabel('Accuracy/F1 Scores', fontsize=25, family="Times")
    #plt.title('Intent Accuracy for Spanish Comparison between Fine-tuning and X-METRA-ADA')
    plt.xticks(np.array([0.1, 1.2]), ('high2low', 'low2high'), size=23)

    #ax.xaxis.set_tick_params(labelsize=50)
    #ax.yaxis.set_tick_params(labelsize=50)

    if metric == "forgetting":
        plt.yticks(np.arange(0, 8, 1), size=23)
        plt.legend(frameon=False, bbox_to_anchor=(0., 1.1, 1.0, .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0., prop={"size": 23})
        # plt.legend(frameon=False, ncol=1, loc="upper center", prop={"size": 16})
    elif metric == "fwt":
        plt.yticks(np.arange(-1, 7, 1), size=23)
        # plt.legend(frameon=False, ncol=2, loc="upper right", prop={"size": 16})
    else:
        plt.yticks(np.arange(0, 110, 10), size=23)
        # plt.legend(frameon=False, ncol=2, loc="lower right", prop={"size": 16})

    # ax.yaxis.grid(zorder=0, linestyle='dashed')
    # plt.grid(zorder=0)

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    plt.tight_layout()
    # ax.xaxis.grid()
    plt.show()


def plot_base_orders():
    root_dir = _root_dir+"Pickle/"
    alias = ""
    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)
    intent_vanilla = {"forgetting": [], "fwt": [], "fp": []}
    slot_vanilla = {"forgetting": [], "fwt": [], "fp": []}
    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            if per_order_metric[0] == "vanilla":
                if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                    intent_vanilla["forgetting"].append(per_order_metric[2])
                    slot_vanilla["forgetting"].append(per_order_metric[4])

                    intent_vanilla["fwt"].append(per_order_metric[10])
                    slot_vanilla["fwt"].append(per_order_metric[12])

                    intent_vanilla["fp"].append(per_order_metric[14])
                    slot_vanilla["fp"].append(per_order_metric[16])

    intent_joint = {"forgetting": [], "fwt": [], "fp": []}
    slot_joint = {"forgetting": [], "fwt": [], "fp": []}
    # root_dir = _root_dir+"Approaches/x-continuous-learning/metrics/"
    root_dir = _root_dir + "metrics/"
    alias = "_joint"  # "_multi_head_in" #

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                intent_joint["forgetting"].append(per_order_metric[2])
                slot_joint["forgetting"].append(per_order_metric[4])

                intent_joint["fwt"].append(per_order_metric[10])
                slot_joint["fwt"].append(per_order_metric[12])

                intent_joint["fp"].append(per_order_metric[14])
                slot_joint["fp"].append(per_order_metric[16])

    for metric in intent_vanilla:
        plot_lang_order_bar_charts(intent_vanilla[metric], intent_joint[metric], slot_vanilla[metric], slot_joint[metric], metric)


def plot_cont_methods_avg_analysis():
    root_dir = _root_dir + "Pickle/"
    alias = ""
    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    intents = {model_name: {"forgetting": 0.0, "fwt": 0.0, "fp": 0.0} for model_name in model_names}
    slots = {model_name: {"forgetting": 0.0, "fwt": 0.0, "fp": 0.0} for model_name in model_names}

    print("--------------ALL----------------")
    for avg_metric in all_avg_metrics:
        print("avg_metric:", avg_metric)
        method = avg_metric[0]
        if method != "multi_head_out":
            intents[method]["forgetting"] = avg_metric[2]
            slots[method]["forgetting"] = avg_metric[4]

            intents[method]["fwt"] = avg_metric[10]
            slots[method]["fwt"] = avg_metric[12]

            intents[method]["fp"] = avg_metric[14]
            slots[method]["fp"] = avg_metric[16]

    root_dir = _root_dir + "Approaches/x-continuous-learning/metrics/"
    alias = "_joint"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    print("--------------JOINT----------------")
    for avg_metric in all_avg_metrics:
        print("avg_metric:", avg_metric)
        method = avg_metric[0]
        intents[method]["forgetting"] = avg_metric[2]
        slots[method]["forgetting"] = avg_metric[4]

        intents[method]["fwt"] = avg_metric[10]
        slots[method]["fwt"] = avg_metric[12]

        intents[method]["fp"] = avg_metric[14]
        slots[method]["fp"] = avg_metric[16]

    alias = "_multi_head_in"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    print("--------------MULTI-HEAD-IN----------------")
    for avg_metric in all_avg_metrics:
        print("avg_metric:", avg_metric)
        method = avg_metric[0]
        intents[method]["forgetting"] = avg_metric[2]
        slots[method]["forgetting"] = avg_metric[4]

        intents[method]["fwt"] = avg_metric[10]
        slots[method]["fwt"] = avg_metric[12]

        intents[method]["fp"] = avg_metric[14]
        slots[method]["fp"] = avg_metric[16]

    for metric in ["forgetting", "fwt", "fp"]:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        #plt.figure(figsize=(10, 6))
        index = np.array([0])
        bar_width = 0.10
        opacity = 0.8

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 50}

        plt.rcParams['font.size'] = '16'

        #matplotlib.rc('font', **font)
        #matplotlib.rc('text', usetex=True)
        #matplotlib.rc('axes', linewidth=2)
        #matplotlib.rc('font', weight='bold')

        # plt.grid(zorder=3)
        # ax.grid(zorder=5)
        k = 0
        positions = [0]
        my_cmap = sns.color_palette("colorblind", 3, as_cmap=True)
        for method in model_names:
            # plt.bar(index + k * bar_width,
            #         intents[method][metric],
            #         bar_width,
            #         alpha=opacity,
            #         # color='blue',
            #         # hatch="+",
            #         # edgecolor='grey',
            #         label=model_names_dict[method])

            plt.bar(index + k * bar_width,
                    slots[method][metric],
                    bar_width,
                    alpha=opacity,
                    # color='yellowgreen',
                    # hatch="/",
                    # edgecolor='grey',
                    label=model_names_dict[method],
                    cmap=my_cmap)

            # k += 1.2
            # positions.append(index + k * bar_width)
            # plt.bar(index + k * bar_width,
            #         slots[method][metric],
            #         bar_width,
            #         alpha=opacity,
            #         color='yellowgreen',
            #         hatch="/",
            #         edgecolor='grey',
            #         label=' Slot')
            k += 2
            positions.append(index + k * bar_width)

        numbers = []
        for method in model_names:
            # numbers.append(round(intents[method][metric], 1))
            numbers.append(round(slots[method][metric], 1))

        rects = ax.patches

        # Make some labels.
        labels = [str(numbers[i]) for i in range(len(rects))]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            if float(label) < 0:
                h = height - 0.5
            else:
                h = height + 0.1
            ax.text(
                rect.get_x() + rect.get_width() / 2, h, label, ha="center", va="bottom", fontsize=20
            )

        print("numbers:", numbers)

        plt.xlabel('Approaches', fontsize=25, family="Times")
        plt.ylabel('Accuracy/F1 Scores', fontsize=25, family="Times")
        #plt.title('Intent Accuracy for Spanish Comparison between Fine-tuning and X-METRA-ADA')
        # plt.xticks(np.array([0.05 + k * 0.32 for k in range(len(model_names))]), [model_names_dict[method] for method in model_names], size=23, rotation=90)
        if metric == "forgetting":
            plt.yticks(np.arange(0, 6, 2), size=20)
            plt.legend(bbox_to_anchor=(0., 1.0, 1.0, .102), loc=3, frameon=False,
                       ncol=3, mode="expand", borderaxespad=0., prop={"size": 20})
            # plt.legend(["Intent", "Slot"], ncol=2, loc="upper right", prop={"size": 16})
        elif metric == "fwt":
            plt.yticks(np.arange(0, 4, 1), size=20)
            # plt.legend(["Intent", "Slot"], loc="upper right", prop={"size": 16})
        else:
            plt.yticks(np.arange(0, 110, 20), size=20)
            # plt.legend(["Intent", "Slot"], loc="lower right", prop={"size": 16})

        #ax.xaxis.set_tick_params(labelsize=50)
        #ax.yaxis.set_tick_params(labelsize=50)

        # ax.yaxis.grid(zorder=0, linestyle='dashed')
        # plt.grid(zorder=0)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_axisbelow(True)
        right_side = ax.spines["right"]
        right_side.set_visible(False)

        upper_side = ax.spines["top"]
        upper_side.set_visible(False)

        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')

        plt.tight_layout()
        # ax.xaxis.grid()
        plt.show()


def plot_cont_methods_lang_order():
    root_dir = _root_dir + "Pickle/"
    alias = ""
    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    intents = {model_name: {"forgetting": [], "fwt": [], "fp": []} for model_name in model_names}
    slots = {model_name: {"forgetting": [], "fwt": [], "fp": []} for model_name in model_names}

    print("--------------ALL----------------")
    print("intents:", intents)
    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if method != "multi_head_out":
                if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                    intents[method]["forgetting"].append(per_order_metric[2])
                    slots[method]["forgetting"].append(per_order_metric[4])

                    intents[method]["fwt"].append(per_order_metric[10])
                    slots[method]["fwt"].append(per_order_metric[12])

                    intents[method]["fp"].append(per_order_metric[14])
                    slots[method]["fp"].append(per_order_metric[16])

    root_dir = _root_dir + "Approaches/x-continuous-learning/metrics/"
    alias = "_joint"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    print("--------------JOINT----------------")
    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                intents[method]["forgetting"].append(per_order_metric[2])
                slots[method]["forgetting"].append(per_order_metric[4])

                intents[method]["fwt"].append(per_order_metric[10])
                slots[method]["fwt"].append(per_order_metric[12])

                intents[method]["fp"].append(per_order_metric[14])
                slots[method]["fp"].append(per_order_metric[16])

    alias = "_multi_head_in"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    print("--------------MULTI-HEAD-IN----------------")
    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                intents[method]["forgetting"].append(per_order_metric[2])
                slots[method]["forgetting"].append(per_order_metric[4])

                intents[method]["fwt"].append(per_order_metric[10])
                slots[method]["fwt"].append(per_order_metric[12])

                intents[method]["fp"].append(per_order_metric[14])
                slots[method]["fp"].append(per_order_metric[16])

    # df =
    # df = pd.melt(df, id_vars="class", var_name="sex", value_name="survival rate")

    for metric in ["forgetting", "fwt", "fp"]:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        #plt.figure(figsize=(10, 6))
        index = np.array([0])
        bar_width = 0.20
        opacity = 0.8

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 50}

        plt.rcParams['font.size'] = '16'

        #matplotlib.rc('font', **font)
        #matplotlib.rc('text', usetex=True)
        #matplotlib.rc('axes', linewidth=2)
        #matplotlib.rc('font', weight='bold')

        # plt.grid(zorder=3)
        # ax.grid(zorder=5)
        k = 0
        positions = [0]
        colors_i = ['blue', 'darkblue']
        colors_s = ['yellowgreen', 'darkgreen']
        lang_order_dict = {"en_de_fr_hi_es_th": "high2low",
                           "th_es_hi_fr_de_en": "low2high"}
        for method in model_names:
            for i, lang_order in enumerate(["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]):
                plt.bar(index + k * bar_width,
                        intents[method][metric][i],
                        bar_width,
                        alpha=opacity,
                        color=colors_i[i],
                        hatch="+",
                        edgecolor='grey',
                        label='Intent '+lang_order_dict[lang_order])

                k += 1
                positions.append(index + k * bar_width)
                plt.bar(index + k * bar_width,
                        slots[method][metric][i],
                        bar_width,
                        alpha=opacity,
                        color=colors_s[i],
                        hatch="/",
                        edgecolor='grey',
                        label=' Slot '+lang_order_dict[lang_order])
                k += 1.2
                positions.append(index + k * bar_width)

            k += 3

        plt.axvline(x=1, color='gray')
        plt.axvline(x=2.5, color='gray')
        plt.axvline(x=4, color='gray')
        plt.axvline(x=5.5, color='gray')
        plt.axvline(x=5.5, color='gray')
        plt.axvline(x=7, color='gray')
        plt.axvline(x=8.5, color='gray')
        plt.axvline(x=10, color='gray')
        plt.axvline(x=11.5, color='gray')
        plt.axvline(x=13, color='gray')

        numbers = []
        for method in model_names:
            for i, lang_order in enumerate(["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]):
                numbers.append(intents[method][metric][i])
                numbers.append(slots[method][metric][i])

        # for i, v in enumerate(numbers):
        #     plt.text(x=positions[i]-0.07, y=numbers[i]+0.1, s=str(numbers[i]), size=10)

        print("numbers:", numbers)

        plt.xlabel('Approaches', fontsize=25, family="Times")
        plt.ylabel('Accuracy/F1 Scores', fontsize=25, family="Times")
        #plt.title('Intent Accuracy for Spanish Comparison between Fine-tuning and X-METRA-ADA')
        plt.xticks(np.array([0.5 + k*1.5 for k in range(len(model_names))]), [model_names_dict[method] for method in model_names], size=20, rotation=90)
        plt.xticks()
        if metric == "forgetting":
            plt.yticks(np.arange(0, 10, 2), size=20)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys())
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0., 1.0, 0.8, .102), loc=3, frameon=False,
                       ncol=2, mode="expand", borderaxespad=0., prop={"size": 20})
            # plt.legend(by_label.values(), by_label.keys(), loc="lower right", prop={"size": 10})
        elif metric == "fwt":
            plt.yticks(np.arange(0, 7, 1), size=20)
            # plt.legend(by_label.values(), by_label.keys(), loc="upper right", prop={"size": 16})
        else:
            plt.yticks(np.arange(0, 110, 10), size=20)
            # plt.legend(by_label.values(), by_label.keys(), loc="lower right", prop={"size": 16})

        #ax.xaxis.set_tick_params(labelsize=50)
        #ax.yaxis.set_tick_params(labelsize=50)

        # ax.yaxis.grid(zorder=0, linestyle='dashed')
        # plt.grid(zorder=0)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_axisbelow(True)
        right_side = ax.spines["right"]
        right_side.set_visible(False)

        upper_side = ax.spines["top"]
        upper_side.set_visible(False)

        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')

        plt.tight_layout()
        # ax.xaxis.grid()
        plt.show()

### NEW
root_dir = _root_dir + "x-continuous-learning-repro-backup/metrics/seeds/" #"Approaches/x-continuous-learning/metrics/"
def read_average_results():
    alias = "_other-approaches"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    intents = {model_name: {"forgetting": 0.0, "fwt": 0.0, "fp": 0.0} for model_name in model_names}
    slots = {model_name: {"forgetting": 0.0, "fwt": 0.0, "fp": 0.0} for model_name in model_names}

    print("--------------ALL----------------")
    for avg_metric in all_avg_metrics:
        print("avg_metric:", avg_metric)
        method = avg_metric[0]
        if method != "multi_head_out":
            intents[method]["forgetting"] = avg_metric[2]
            slots[method]["forgetting"] = avg_metric[4]

            intents[method]["fwt"] = avg_metric[10]
            slots[method]["fwt"] = avg_metric[12]

            intents[method]["fp"] = avg_metric[14]
            slots[method]["fp"] = avg_metric[16]

    alias = "_joint_multi-head"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    print("--------------JOINT----------------")
    for avg_metric in all_avg_metrics:
        print("avg_metric:", avg_metric)
        method = avg_metric[0]
        if method in model_names:
            intents[method]["forgetting"] = avg_metric[2]
            slots[method]["forgetting"] = avg_metric[4]

            intents[method]["fwt"] = avg_metric[10]
            slots[method]["fwt"] = avg_metric[12]

            intents[method]["fp"] = avg_metric[14]
            slots[method]["fp"] = avg_metric[16]

    return intents, slots


def read_lang_order_results():
    alias = "_other-approaches"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    intents = {model_name: {"forgetting": [], "fwt": [], "fp": []} for model_name in model_names}
    slots = {model_name: {"forgetting": [], "fwt": [], "fp": []} for model_name in model_names}

    print("--------------ALL----------------")
    print("intents:", intents)
    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if method != "multi_head_out":
                if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                    intents[method]["forgetting"].append(per_order_metric[2])
                    slots[method]["forgetting"].append(per_order_metric[4])

                    intents[method]["fwt"].append(per_order_metric[10])
                    slots[method]["fwt"].append(per_order_metric[12])

                    intents[method]["fp"].append(per_order_metric[14])
                    slots[method]["fp"].append(per_order_metric[16])

    alias = "_joint_multi-head"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    print("--------------JOINT----------------")
    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if method in model_names:
                if per_order_metric[1] in ["en_de_fr_hi_es_th", "th_es_hi_fr_de_en"]:
                    intents[method]["forgetting"].append(per_order_metric[2])
                    slots[method]["forgetting"].append(per_order_metric[4])

                    intents[method]["fwt"].append(per_order_metric[10])
                    slots[method]["fwt"].append(per_order_metric[12])

                    intents[method]["fp"].append(per_order_metric[14])
                    slots[method]["fp"].append(per_order_metric[16])

    return intents, slots

def read_all_lang_order_results():
    intents = {model_name: {"forgetting": [], "fwt": [], "fwt_0": [], "fp": []} for model_name in model_names}
    slots = {model_name: {"forgetting": [], "fwt": [], "fwt_0": [], "fp": []} for model_name in model_names}

    # for alias in ["_other-approaches", "_joint_multi-head", "_multi_head_out"]:
    aliases = ["all"]

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, aliases)

    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if method in model_names:
                intents[method]["forgetting"].append((per_order_metric[1], per_order_metric[2]))
                slots[method]["forgetting"].append((per_order_metric[1], per_order_metric[4]))

                intents[method]["fwt"].append((per_order_metric[1], per_order_metric[10]))
                slots[method]["fwt"].append((per_order_metric[1], per_order_metric[12]))

                intents[method]["fwt_0"].append((per_order_metric[1], per_order_metric[6]))
                slots[method]["fwt_0"].append((per_order_metric[1], per_order_metric[8]))

                intents[method]["fp"].append((per_order_metric[1], per_order_metric[14]))
                slots[method]["fp"].append((per_order_metric[1], per_order_metric[16]))

    return intents, slots

def read_all_lang_order_results_seeds():
    intents = {model_name: {"forgetting": [], "fwt": [], "fwt_0": [], "fp": []} for model_name in model_names}
    slots = {model_name: {"forgetting": [], "fwt": [], "fwt_0": [], "fp": []} for model_name in model_names}

    # for alias in ["_other-approaches", "_joint_multi-head", "_multi_head_out"]:

    with open(root_dir +"all_except_adaptersfrozen-multienc8/all_per_order_metrics_bootstrap.pickle", "rb") as file:
        all_per_order_metrics = pickle.load(file)

    print("all_per_order_metrics:", all_per_order_metrics)

    for per_orders_metric in all_per_order_metrics:
        for per_order_metric in per_orders_metric:
            method = per_order_metric[0]
            if method in model_names:
                intents[method]["forgetting"].append((per_order_metric[1], per_order_metric[2]))
                slots[method]["forgetting"].append((per_order_metric[1], per_order_metric[4]))

                intents[method]["fwt"].append((per_order_metric[1], per_order_metric[10]))
                slots[method]["fwt"].append((per_order_metric[1], per_order_metric[12]))

                intents[method]["fwt_0"].append((per_order_metric[1], per_order_metric[6]))
                slots[method]["fwt_0"].append((per_order_metric[1], per_order_metric[8]))

                intents[method]["fp"].append((per_order_metric[1], per_order_metric[14]))
                slots[method]["fp"].append((per_order_metric[1], per_order_metric[16]))

    return intents, slots

def read_forgetting_fwt_fp_no_outliers():
    alias = "_other-approaches"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    forgetting = {"intents": [], "slots": []}
    fwt = {"intents": [], "slots": []}
    fp = {"intents": [], "slots": []}

    labels = []
    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        if method not in ["joint_plus",
                          "multi_head_out",
                          "adapters/FROZEN_BERT",
                          "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool"]:
            #["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool", "multi_head_out", "adapters/FROZEN_BERT"]:
            labels.append(model_names_dict[method])
            forgetting["intents"].append(-avg_metric[2])
            fwt["intents"].append(avg_metric[10])
            fp["intents"].append(avg_metric[14])

            forgetting["slots"].append(-avg_metric[4])
            fwt["slots"].append(avg_metric[12])
            fp["slots"].append(avg_metric[16])

    ###
    alias = "_joint_multi-head"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        if method in model_names:
            if method not in ["multi_head_out", "Lang-Spec Trans", "joint_plus"]: #["joint_plus", "multi_head_out"]: #"adapters/FROZEN_BERT"
                labels.append(model_names_dict[method])
                forgetting["intents"].append(-avg_metric[2])
                fwt["intents"].append(avg_metric[10])
                fp["intents"].append(avg_metric[14])

                forgetting["slots"].append(-avg_metric[4])
                fwt["slots"].append(avg_metric[12])
                fp["slots"].append(avg_metric[16])

    alias = "_multi_head_out"

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, alias)

    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        if method in model_names:
            labels.append(model_names_dict[method])
            forgetting["intents"].append(-avg_metric[2])
            fwt["intents"].append(avg_metric[10])
            fp["intents"].append(avg_metric[14])

            forgetting["slots"].append(-avg_metric[4])
            fwt["slots"].append(avg_metric[12])
            fp["slots"].append(avg_metric[16])

    return forgetting, fwt, fp, labels


def read_forgetting_fwt_fp():
    # aliases = ["er-ablation_vanilla", "multi_head_in", "multi_head_out", "vanilla_adapters_ewc_joint_kdlogit_kdrep"]
    aliases = ["all_except_adaptersfrozen-multienc8"]

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, aliases)

    forgetting = {"intents": [], "slots": []}
    fwt = {"intents": [], "slots": []}
    fp = {"intents": [], "slots": []}

    labels = []
    covered = []
    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        if method in model_names_dict.keys() and method not in ["joint_plus",
                          "er_memsz-750_type-reservoir_sample-random_k-16",
                          "er_memsz-1500_type-reservoir_sample-random_k-16",
                          "er_memsz-3000_type-reservoir_sample-random_k-16",
                          "er_memsz-4500_type-reservoir_sample-random_k-16"]:#, "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool"]:
            #["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool", "multi_head_out", "adapters/FROZEN_BERT"]:
            if model_names_dict[method] not in covered:
                labels.append(model_names_dict[method])
                forgetting["intents"].append(-avg_metric[2])
                fwt["intents"].append(avg_metric[10])
                fp["intents"].append(avg_metric[14])

                forgetting["slots"].append(-avg_metric[4])
                fwt["slots"].append(avg_metric[12])
                fp["slots"].append(avg_metric[16])

            covered.append(model_names_dict[method])


    return forgetting, fwt, fp, labels


def read_from_pickle_short(root_dir, alias):
    with open(root_dir+alias+"/all_per_order_metrics_bootstrap.pickle", "rb") as file:
        all_per_order_metrics = pickle.load(file)

    with open(root_dir+alias+"/all_avg_metrics_bootstrap.pickle", "rb") as file:
        all_avg_metrics = pickle.load(file)

    return all_per_order_metrics, all_avg_metrics


def read_from_pickle_results(root_dir, aliases):
    all_per_order_metrics = []
    all_avg_metrics = []
    all_forget_lang_strings = []
    all_fwt_lang_strings = []
    all_fwt_mono_lang_strings = []

    for alias in aliases:
        with open(root_dir+alias+"/all_per_order_metrics_bootstrap.pickle", "rb") as file:
            all_per_order_metrics += pickle.load(file)

        with open(root_dir+alias+"/all_avg_metrics_bootstrap.pickle", "rb") as file:
            all_avg_metrics += pickle.load(file)

        # with open(root_dir+alias+"/all_forget_lang_strings_bootstrap.pickle", "rb") as file:
        #     all_forget_lang_strings += pickle.load(file)
        #
        # with open(root_dir+alias+"/all_fwt_lang_strings_bootstrap.pickle", "rb") as file:
        #     all_fwt_lang_strings += pickle.load(file)
        #
        # with open(root_dir+alias+"/all_fwt_mono_lang_strings_bootstrap.pickle", "rb") as file:
        #     all_fwt_mono_lang_strings += pickle.load(file)

    return all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings

#### PLOTS
def plot_cont_methods_avg_analysis_seaborn(option="intents"):  # option could be either intents or slots
    metric_labels_dict = {"forgetting": "Forgetting",
                          "fwt": "Transfer",
                          "fp": "Final Performance"}
    intents, slots = read_average_results()
    if option == "intents":
        data = intents
    else:
        data = slots

    for metric in ["forgetting", "fwt", "fp"]:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.rcParams['font.size'] = '16'

        sns.set_palette("colorblind")

        plot_data = {
            'x': model_names,
            'y': [data[method][metric] for method in model_names],
            'l': [model_names_dict[model_name] for model_name in model_names]
        }

        if metric == "forgetting":
            splot = sns.barplot(x='x', y='y', hue='l', data=plot_data, dodge=False)
        else:
            splot = sns.barplot(x='x', y='y', data=plot_data, dodge=False)

        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points', family="Times", size=size_text)

        plt.xlabel('Approaches', fontsize=25, family="Times")
        if option == "intents":
            ylabel = metric_labels_dict[metric] + " of Accuracy Scores"
        else:
            ylabel = metric_labels_dict[metric] + " of F1 Scores"
        plt.ylabel(ylabel, fontsize=25, family="Times")
        plt.xticks([])
        if option == "intents":
            forg_y_max = 5
            fwt_y_max = 4
            fp_y_max = 110
        else:
            forg_y_max = 8
            fwt_y_max = 6
            fp_y_max = 100

        if metric == "forgetting":
            plt.yticks(np.arange(0, forg_y_max, 1), size=size_text, family="Times")
            plt.legend(bbox_to_anchor=(0., 1.0, 1.0, .102), loc=3, frameon=False,
                       ncol=3, mode="expand", borderaxespad=0., prop={"size": 24, "family": "Times"})
        elif metric == "fwt":
            plt.yticks(np.arange(0, fwt_y_max, 1), size=size_text, family="Times")
        else:
            plt.yticks(np.arange(0, fp_y_max, 20), size=size_text, family="Times")

        ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_axisbelow(True)
        right_side = ax.spines["right"]
        right_side.set_visible(False)

        upper_side = ax.spines["top"]
        upper_side.set_visible(False)

        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')

        plt.tight_layout()
        # plt.show()
        filename = plot_save_path+"all/q2_"+metric
        if option == "slots":
            filename = filename+"_s"

        plt.savefig(filename+".png", format="png")


def plot_cont_methods_lang_order_seaborn(option="intents"):

    metric_labels_dict = {"forgetting": "Forgetting",
                          "fwt": "Transfer",
                          "fp": "Final Performance"}
    intents, slots = read_lang_order_results()
    if option == "intents":
        data = intents
    else:
        data = slots
    for metric in ["forgetting"]:#["forgetting", "fwt", "fp"]:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        plt.rcParams['font.size'] = '16'
        xs = []
        ys = []
        labels = []
        hatches = []
        for method in model_names:
            for i in range(2):
                xs.append(method+"_"+str(i))
                ys.append(data[method][metric][i])
                labels.append(model_names_dict[method])
                if i == 0:
                    hatches.append("/")
                else:
                    hatches.append("+")

        plot_data = {
            'x': xs,
            'y': ys,
            'l': labels,
        }

        sns.set_palette("colorblind")
        if metric == "forgetting":
            splot = sns.barplot(x='x', y='y', hue='l', data=plot_data, dodge=False)
        else:
            splot = sns.barplot(x='x', y='y', hue='l', data=plot_data, hatch="/", dodge=False)

        # Loop over the bars
        for i, thisbar in enumerate(splot.patches):
            if i % 2 == 0:
                thisbar.set_hatch("/")
            else:
                thisbar.set_hatch("+")

        for p in splot.patches:
            if p.get_height() < 0:
                text_height = -0.5
            else:
                text_height = p.get_height()
            if metric == "fp":
                text_size = 15
            else:
                text_size = 26
            splot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., text_height),#p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points', family="Times", size=text_size)

        plt.xlabel('Approaches', fontsize=25, family="Times")
        if option == "intents":
            ylabel = metric_labels_dict[metric] + " of Accuracy Scores"
        else:
            ylabel = metric_labels_dict[metric] + " of F1 Scores"
        plt.ylabel(ylabel, fontsize=25, family="Times")
        plt.xticks([])

        if option == "intents":
            forg_y_max = 8
            fwt_y_max = 5
            fwt_y_min = 0
            fp_y_max = 110
        else:
            forg_y_max = 10
            fwt_y_max = 6
            fwt_y_min = -1
            fp_y_max = 80

        # 'Naive Seq FT': '#0173b2',
        # 'Inc Joint': '#de8f05',
        # 'Lang Spec Enc [0-8]': '#029e73',
        # 'Lang Spec Trans': '#d55e00',
        # 'Lang Spec Ada (Tuned)': '#cc78bc',
        # 'Lang Spec Ada (Frozen)':  '#ca9161',
        # 'EWC-Online': '#fbafe4',
        # 'ER': '#949494',
        # 'KD-Rep': '#ece133',
        # 'KD-Logit': '#56b4e9'

        if metric == "forgetting":
            plt.yticks(np.arange(0, forg_y_max, 2), size=size_text, family="Times")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            slash_patch = mpatches.Patch(hatch='/', facecolor=[1.0, 1.0, 1.0, 1], label='high2low')
            cross_patch = mpatches.Patch(hatch='+', facecolor=[1.0, 1.0, 1.0, 1], label='low2high')
            # plt.legend(handles=[slash_patch, cross_patch])

            # slash_patch = mpatches.Patch(hatch='/', facecolor=[1.0, 1.0, 1.0, 1], label='high2low')
            # cross_patch = mpatches.Patch(hatch='+', facecolor=[1.0, 1.0, 1.0, 1], label='low2high')
            # plt.legend(handles=[slash_patch, cross_patch], bbox_to_anchor=(0., 1.0, 0.8, .102), loc=3, frameon=False,
            #            ncol=2, mode="expand", borderaxespad=0., prop={"size": 20, "family": "Times"})

            naive_seq_ft = mpatches.Patch(facecolor=[1/255, 115/255, 178/255, 1], label='Naive Seq FT') # '#0173b2',
            inc_joint = mpatches.Patch(facecolor=[222/255, 143/255, 5/255, 1], label='Inc Joint') # '#de8f05',
            lang_spec_enc = mpatches.Patch(facecolor=[2/255, 158/255, 115/255, 1], label='Lang-Spec Enc[0-8]') # '#029e73',
            lang_spec_trans = mpatches.Patch(facecolor=[213/255, 94/255, 0/255, 1], label='Lang-Spec Trans') #  '#d55e00',
            lang_spec_ada_tuned = mpatches.Patch(facecolor=[204/255, 120/255, 188/255, 1], label='Lang-Spec Ada(T)') # '#cc78bc',
            lang_spec_ada_frozen = mpatches.Patch(facecolor=[202/255, 145/255, 97/255, 1], label='Lang-Spec Ada(F)') # '#ca9161',
            ewc_online = mpatches.Patch(facecolor=[251/255, 175/255, 228/255, 1], label='EWC-Online') # '#fbafe4',
            er = mpatches.Patch(facecolor=[148/255, 148/255, 148/255, 1], label='ER') # '#949494',
            kd_logit = mpatches.Patch(facecolor=[236/255, 225/255, 51/255, 1], label='KD-Logit') # '#ece133',
            kd_rep = mpatches.Patch(facecolor=[86/255, 180/255, 233/255, 1], label='KD-Rep') # '#56b4e9'

            legend1 = plt.legend(handles=[slash_patch, cross_patch],
                                 loc='lower right')
            # bbox_to_anchor=(2.0, 3.0, 0.8, .102),
            # loc=2, frameon=False,
            # ncol=2, mode="expand", borderaxespad=0., prop={"size": 20, "family": "Times"})


            plt.gca().add_artist(legend1)

            plt.legend(handles=[naive_seq_ft, inc_joint, lang_spec_enc, lang_spec_trans, lang_spec_ada_tuned,
                                lang_spec_ada_frozen, ewc_online, er, kd_logit, kd_rep],
                       # loc='upper right')
                       bbox_to_anchor=(0., 1.0, 0.8, .102),
                       loc=3, frameon=False,
                       ncol=3, mode="expand", borderaxespad=0., prop={"size": 20, "family": "Times"})

            # plt.legend(["red", "blue"], ["high2low", "low2high"], bbox_to_anchor=(0., 1.0, 0.8, .102), loc=3, frameon=False,
            #            ncol=2, mode="expand", borderaxespad=0., prop={"size": 20})
            # plt.legend(by_label.values(), by_label.keys(), loc="lower right", prop={"size": 10})
        elif metric == "fwt":
            plt.yticks(np.arange(fwt_y_min, fwt_y_max, 1), size=size_text, family="Times")
            plt.legend([],  bbox_to_anchor=(0., 1.0, 0.8, .102), loc=3, frameon=False,
                       ncol=2, mode="expand", borderaxespad=0., prop={"size": 20, "family": "Times"})
            # plt.legend(by_label.values(), by_label.keys(), loc="upper right", prop={"size": 16})
        else:
            plt.yticks(np.arange(0, fp_y_max, 10), size=20)
            plt.legend([],  bbox_to_anchor=(0., 1.0, 0.8, .102), loc=3, frameon=False,
                       ncol=2, mode="expand", borderaxespad=0., prop={"size": 20, "family": "Times"})
            # plt.legend(by_label.values(), by_label.keys(), loc="lower right", prop={"size": 16})

        ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_axisbelow(True)
        right_side = ax.spines["right"]
        right_side.set_visible(False)

        upper_side = ax.spines["top"]
        upper_side.set_visible(False)

        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')

        plt.tight_layout()
        plt.show()

        # filename = plot_save_path+"all/q3_"+metric
        # if option == "slots":
        #     filename = filename+"_slots"
        #
        # plt.savefig(filename+".png", format="png")


def plot_transfer_interference_fp_curve(option="intents", option2="fwt", option3="f"):
    forgetting, fwt, fp, labels = read_forgetting_fwt_fp()

    if option3 == "f":
        xlabel = "Negative Forgetting"
        list_a = forgetting[option]
    elif option3 == "fwt":
        xlabel = "Transfer"
        list_a = fwt[option]
    else:
        xlabel = "Final Performance"
        list_a = fp[option]

    if option2 == "fwt":
        list_b = fwt[option]
        ylabel = "Transfer"
    else:
        list_b = fp[option]
        ylabel = "Final Performance"

    if option == "intents":
        ylabel += " of Accuracy Scores"
        xlabel += " of Accuracy Scores"
    else:
        ylabel += " of F1 Scores"
        xlabel += " of F1 Scores"

    list_c = labels

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # list_a[:], list_b[:] = zip(*sorted(zip(list_a, list_b), key=lambda p: (p[0], p[1])))
    plt.scatter(list_a, list_b, label=option, color="black")

    texts = []
    for x, y, s in zip(list_a, list_b, list_c):
        # if option2 == "fwt":
        #     if option == "slots":
        #         if s in ["Lang-Spec Enc[0-8]", "Inc Joint", "Lang-Spec Trans"]:
        #             c = "red"
        #         else:
        #             c = "green"
        #     else:
        #         # if s in ["EWC-Online", "Naive Seq FT", "Lang-Spec Task", "Lang-Spec Enc[0-8]"]:
        #         if s in ["Lang-Spec Trans"]:
        #             c = "red"
        #         else:
        #             c = "green"
        # else:
        #     if s in ["Lang-Spec Ada(F)"]:
        #         c = "red"
        #     else:
        #         c = "green"

        c = "green"

        font = FontProperties()
        if c == "green":
            font.set_weight('bold')
            texts.append(plt.text(x, y, s, size=25, color=c, family="Times", fontproperties=font))
        else:
            font.set_style('italic')
            texts.append(plt.text(x, y, s, size=25, color=c, family="Times", fontproperties=font))

    adjust_text(texts,
                only_move={'points': 'y', 'texts': 'y'},
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

    plt.xlabel(xlabel, fontsize=size_text, family="Times")
    plt.ylabel(ylabel, fontsize=size_text, family="Times")

    plt.xticks(size=size_text, family="Times")
    plt.yticks(size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    plt.tight_layout()

    if option2 == "fwt":
        filename = plot_save_path+"seeds/q4_transfer_interference"
    else: # fp
        if option3 == "fwt":
            filename = plot_save_path + "seeds/q4_finalperf_transfer"
        else:
            filename = plot_save_path+"seeds/q4_finalperf_interference"

    if option == "slots":
        filename = filename + "_slots"

    # plt.show()

    plt.savefig(filename+".png", format="png")


def plot_two_steps(option="intents", option2="two_steps", option3="forgetting"): # option2 can be two_steps or multiple_steps
    model_names = ["vanilla",
                   "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                   "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-logits_memsz-6000_type-ring_sample-random_k-16"]

    root_dir = _root_dir + "Approaches/x-continuous-learning/metrics/"
    if option2 == "two_steps":
        alias = "two_steps"
    else:
        alias = "other-approaches"

    with open(root_dir+"all_per_order_metrics_bootstrap_"+alias+".pickle", "rb") as file:
        data = pickle.load(file)


    forgetting = {}
    fwts = {}
    fps = {}
    xs = {}
    for method_two_steps in data:
        for one_order_two_steps in method_two_steps:
            method = one_order_two_steps[0]
            lang_order = one_order_two_steps[1]

            if method not in forgetting:
                forgetting.update({method: []})
            if method not in fwts:
                fwts.update({method: []})
            if method not in fps:
                fps.update({method: []})
            if method not in xs:
                xs.update({method: []})

            if option == "intents":
                forgetting[method].append(one_order_two_steps[2])
                fwts[method].append(one_order_two_steps[10])
                fps[method].append(one_order_two_steps[14])
            else:
                forgetting[method].append(one_order_two_steps[4])
                fwts[method].append(one_order_two_steps[12])
                fps[method].append(one_order_two_steps[16])
            # labels.append(method)
            xs[method].append(lang_orders_dict[lang_order])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # sns.set_palette("colorblind")

    colors = cm.rainbow(np.linspace(0, 1, len(model_names)))
    # for method, c in zip(model_names, colors):

    # from matplotlib.colors import ListedColormap
    # colors = np.linspace(0,1,len(model_names))
    # my_cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    # with sns.color_palette("husl", len(model_names)):

    # current_palette = sns.color_palette("colorblind")
    # sns.palplot(current_palette)
    sns.set_palette("colorblind")
    for method, c in zip(model_names, colors):
        if option3 == "forgetting":
            ys = forgetting[method]
        elif option3 == "fwt":
            ys = fwts[method]
        else:
            ys = fps[method]

        sns.scatterplot(xs[method], ys, label=model_names_dict[method])#, c=c, cmap=my_cmap)

    plt.xlabel('Language Order', fontsize=25, family="Times")
    if option3 == "forgetting":
        ylabel = "Forgetting"
    elif option3 == "fwt":
        ylabel = "Transfer"
    else:
        ylabel = "Final Performance"

    plt.ylabel(ylabel, fontsize=25, family="Times")

    plt.xticks(size=size_text, family="Times")
    plt.yticks(size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # plt.tight_layout()

    plt.legend(bbox_to_anchor=(0., 1.0, 1.0, .102), loc=3, #frameon=False,
               ncol=3, mode="expand", borderaxespad=0., prop={"size": 24, "family": "Times"})

    plt.show()
    filename = plot_save_path+"all/"+option2+"_"+option3

    if option == "slots":
        filename = filename + "_slots"

    # plt.savefig(filename+".png", format="png")


def read_boxplot_contents(option, option2):
    if option2 == "two_steps":
        # aliases = ["two_steps"]
        alias = "two_steps"
    else:
        alias = "all"
        # aliases = ["er-ablation_vanilla", "multi_head_in", "multi_head_out", "vanilla_adapters_ewc_joint_kdlogit_kdrep"]

    # for alias in aliases:
    with open(root_dir+alias+"/all_per_order_metrics_bootstrap.pickle", "rb") as file:
        data = pickle.load(file)


    forgetting = {}
    fwts = {}
    fps = {}
    xs = {}
    for method_two_steps in data:
        print("method_two_steps:", method_two_steps)
        # if option2 == "two_steps":
        #     method_two_steps = method_two_steps[3:5]
        # else:
        #     method_two_steps = method_two_steps[2:4]
        print("After ", method_two_steps)
        for one_order_two_steps in method_two_steps:
            method = one_order_two_steps[0]
            lang_order = one_order_two_steps[1]
            print("lang_order:", lang_order)

            if method not in forgetting:
                forgetting.update({method: []})
            if method not in fwts:
                fwts.update({method: []})
            if method not in fps:
                fps.update({method: []})
            if method not in xs:
                xs.update({method: []})

            if option == "intents":
                forgetting[method].append(one_order_two_steps[2])
                fwts[method].append(one_order_two_steps[10])
                fps[method].append(one_order_two_steps[14])
            else:
                forgetting[method].append(one_order_two_steps[4])
                fwts[method].append(one_order_two_steps[12])
                fps[method].append(one_order_two_steps[16])
            # labels.append(method)
            xs[method].append(lang_orders_dict[lang_order])

    return forgetting, fwts, fps, xs


def boxplot_two_steps(option="intents", option2="two_steps", option3="forgetting"): # option2 can be two_steps or multiple_steps
    model_names = ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                   "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-logits_memsz-6000_type-ring_sample-random_k-16",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "vanilla"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # sns.set_palette("colorblind")

    forgetting, fwts, fps, xs = read_boxplot_contents(option, option2)
    # forgetting_m, fwts_m, fps_m, xs_m = read_boxplot_contents(option, option2="multiple_steps")

    box_plot_data = []
    lang_order_box_plot_data = []
    method_box_plot_data = []
    for method in model_names:
        if option3 == "forgetting":
            ys = forgetting[method]
        elif option3 == "fwt":
            ys = fwts[method]
        else:
            ys = fps[method]
        # sns.scatterplot(xs[method], ys, label=model_names_dict[method])#, c=c, cmap=my_cmap)
        print("len(xs[method]):", len(xs[method]))
        # lang_order_box_plot_data.extend(xs[method])
        lang_order_box_plot_data.extend(2*[1, 2, 3, 4, 5, 6])
        method_box_plot_data.extend(2*[model_names_dict[method]])
        box_plot_data.extend(ys)

    ###
    # box_plot_data_m = []
    # lang_order_box_plot_data_m = []
    # method_box_plot_data_m = []
    # for method in model_names:
    #     if option3 == "forgetting":
    #         ys_m = forgetting_m[method]
    #     elif option3 == "fwt":
    #         ys_m = fwts_m[method]
    #     else:
    #         ys_m = fps[method]
    #     # sns.scatterplot(xs[method], ys, label=model_names_dict[method])#, c=c, cmap=my_cmap)
    #     # lang_order_box_plot_data_m.extend([x+1 for x in xs_m[method]])
    #     lang_order_box_plot_data_m.extend(2*[1, 2, 3, 4, 5, 6])
    #     method_box_plot_data_m.extend(2*[model_names_dict[method]])
    #     box_plot_data_m.extend(ys_m)
    #
    # print("box_plot_data:", box_plot_data)
    # print("lang_order_box_plot_data:", lang_order_box_plot_data)
    # print("method_box_plot_data:", method_box_plot_data)

    # plot_data = {
    #     'x': method_box_plot_data,
    #     'y': box_plot_data,
    #     'l': [model_names_dict[model_name] for model_name in model_names]
    # }
    # sns.boxplot(x='x', y='y', hue='l', data=plot_data, dodge=False, palette="colorblind")

    # plt.boxplot(box_plot_data, showfliers=False)
    pal = sns.color_palette("colorblind")
    colors = pal.as_hex()
    print("colors:", colors)

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Lang-Spec Enc[0-8]': '#ca9161',
                      'Lang-Spec Trans': '#d55e00',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Logit': '#ece133'}#, '#fbafe4', '#949494', '#ece133', '#56b4e9'}
    if option2 == "two_steps":
        sns.violinplot(x=method_box_plot_data, y=box_plot_data, hue=method_box_plot_data, palette=custom_palette,
                       showfliers=False)
    else:
        sns.violinplot(x=method_box_plot_data, y=box_plot_data, palette=custom_palette,
                       showfliers=False)

    # sns.boxplot(x=method_box_plot_data_m, y=box_plot_data_m, hue=method_box_plot_data_m, palette=custom_palette,
    #             showfliers=False)

    plt.xlabel('Approaches', fontsize=25, family="Times")
    if option3 == "forgetting":
        ylabel = "Forgetting"
    elif option3 == "fwt":
        ylabel = "Transfer"
    else:
        ylabel = "Final Performance"

    plt.ylabel(ylabel, fontsize=25, family="Times")

    plt.xticks([])
    plt.yticks(size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # plt.tight_layout()

    if option2 == "two_steps":
        plt.legend(bbox_to_anchor=(0., 1.0, 1.0, .102), loc=3, #frameon=False,
                   ncol=3, mode="expand", borderaxespad=0., prop={"size": 24, "family": "Times"})

    plt.show()
    # filename = plot_save_path+"all/"+option2+"_"+option3
    #
    # if option == "slots":
    #     filename = filename + "_slots"
    #
    # plt.savefig(filename+".png", format="png")


def boxplot_merged_old(option="intents", option2="two_steps", option3="forgetting"): # option2 can be two_steps or multiple_steps
    model_names = ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                   "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-logits_memsz-6000_type-ring_sample-random_k-16",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "vanilla"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # sns.set_palette("colorblind")

    forgetting, fwts, fps, xs = read_boxplot_contents(option, option2)
    forgetting_m, fwts_m, fps_m, xs_m = read_boxplot_contents(option, option2="multiple_steps")

    box_plot_data = []
    lang_order_box_plot_data = []
    method_box_plot_data = []
    for method in model_names:
        if option3 == "forgetting":
            ys = forgetting[method]
        elif option3 == "fwt":
            ys = fwts[method]
        else:
            ys = fps[method]
        # sns.scatterplot(xs[method], ys, label=model_names_dict[method])#, c=c, cmap=my_cmap)
        print("ys:", ys, " len(ys):", len(ys))
        print("len(xs[method]):", len(xs[method]))
        # lang_order_box_plot_data.extend(xs[method])
        lang_order_box_plot_data.extend(6*[1, 2, 3, 4, 5, 6])
        method_box_plot_data.extend(6*[model_names_dict[method]])
        box_plot_data.extend(ys)

    ##
    box_plot_data_m = []
    lang_order_box_plot_data_m = []
    method_box_plot_data_m = []
    for method in model_names:
        if option3 == "forgetting":
            ys_m = forgetting_m[method]
        elif option3 == "fwt":
            ys_m = fwts_m[method]
        else:
            ys_m = fps_m[method]
        # sns.scatterplot(xs[method], ys, label=model_names_dict[method])#, c=c, cmap=my_cmap)
        # lang_order_box_plot_data_m.extend([x+1 for x in xs_m[method]])
        lang_order_box_plot_data_m.extend(6*[1, 2, 3, 4, 5, 6])
        method_box_plot_data_m.extend(6*[model_names_dict[method]+" Z"])
        box_plot_data_m.extend(ys_m)
    #
    # print("box_plot_data:", box_plot_data)
    # print("lang_order_box_plot_data:", lang_order_box_plot_data)
    # print("method_box_plot_data:", method_box_plot_data)

    # plot_data = {
    #     'x': method_box_plot_data,
    #     'y': box_plot_data,
    #     'l': [model_names_dict[model_name] for model_name in model_names]
    # }
    # sns.boxplot(x='x', y='y', hue='l', data=plot_data, dodge=False, palette="colorblind")

    # plt.boxplot(box_plot_data, showfliers=False)
    # pal = sns.color_palette("colorblind")
    # colors = pal.as_hex()
    # print("colors:", colors)

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Lang-Spec Enc[0-8]': '#029e73',
                      'Lang-Spec Trans': '#d55e00',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Logit': '#ece133',
                      'KD-Rep': '#56b4e9',
                      'Naive Seq FT Z': '#0173b2',
                      'Lang-Spec Enc[0-8] Z': '#029e73',
                      'Lang-Spec Trans Z': '#d55e00',
                      'EWC-Online Z': '#fbafe4',
                      'ER Z': '#949494',
                      'KD-Logit Z': '#ece133',
                      'KD-Rep Z': '#56b4e9'}#, '#fbafe4', '#949494', '#ece133', '#56b4e9'}


    # if option2 == "two_steps":
    #     sns.violinplot(x=method_box_plot_data, y=box_plot_data, hue=method_box_plot_data, palette=custom_palette,
    #                    showfliers=False)
    # else:

    print("method_box_plot_data:", method_box_plot_data)
    print("box_plot_data:", box_plot_data)

    sns.boxplot(x=method_box_plot_data, y=box_plot_data, hue=method_box_plot_data, palette=custom_palette,
                   showfliers=False)

    print("method_box_plot_data_m:", method_box_plot_data_m)
    sns.boxplot(x=method_box_plot_data_m, y=box_plot_data_m, palette=custom_palette,
                 width=0.3, showfliers=False)

    # data.boxplot(positions=[1, 5, 6, 10])
    # data = pd.DataFrame(rnd.randn(10, 4))
    # data.boxplot(positions=[1, 5, 6, 10], color=method_box_plot_data, showfliers=False)
    # data = pd.DataFrame(method_box_plot_data)
    # print("box_plot_data:", box_plot_data)
    # print("rnd.randn(10, 4):", rnd.randn(10, 4))
    # data.boxplot(positions=[1, 2, 3, 4, 5, 6], showfliers=False)
    # sns.boxplot(x=method_box_plot_data, y=box_plot_data, positions=[1, 5, 6, 10])

    plt.axvline(x=0.5, color='gray')
    plt.axvline(x=1.5, color='gray')
    plt.axvline(x=2.5, color='gray')
    plt.axvline(x=3.5, color='gray')
    plt.axvline(x=4.5, color='gray')

    plt.xlabel('Approaches', fontsize=size_text, family="Times")
    if option3 == "forgetting":
        ylabel = "Forgetting"
    elif option3 == "fwt":
        ylabel = "Transfer"
    else:
        ylabel = "Final Performance"

    if option == "intents":
        ylabel += " of Accuracy Scores"
    else:
        ylabel += " of F1 Scores"

    plt.ylabel(ylabel, fontsize=size_text, family="Times")

    plt.xticks(np.array([0, 1, 2, 3, 4, 5]),
               ["Lang-Spec Trans", "Lang-Spec Enc[0-8]", "ER", "KD-Logit", "EWC-Online", "Naive Seq FT"],
               fontsize=15, family="Times", rotation=0)

    plt.yticks(size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    plt.legend(bbox_to_anchor=(0., 1.0, 1.0, .102), loc=4, frameon=False,
               ncol=3, mode="expand", borderaxespad=0., prop={"size": 24, "family": "Times"})

    plt.tight_layout()

    # my_cmap = sns.color_palette("colorblind", 3, as_cmap=True)
    # sns.palplot(my_cmap)

    plt.show()
    # filename = plot_save_path+"all/"+option2+"_"+option3
    #
    # if option == "slots":
    #     filename = filename + "_slots"
    #
    # plt.savefig(filename+".png", format="png")


def boxplot_merged(option="intents", option2="two_steps", option3="forgetting"): # option2 can be two_steps or multiple_steps
    if option3 == "forgetting":
        model_names = ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                       "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                       "er_memsz-6000_type-reservoir_sample-random_k-16",
                       "kd-logits_memsz-6000_type-ring_sample-random_k-16",
                       "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                       "vanilla"]
    elif option3 == "fwt":
        model_names = ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                       "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                       "vanilla",
                       "kd-logits_memsz-6000_type-ring_sample-random_k-16",
                       "er_memsz-6000_type-reservoir_sample-random_k-16",
                       "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8"]
    else:
        model_names = ["vanilla",
                       "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                       "kd-logits_memsz-6000_type-ring_sample-random_k-16",
                       "er_memsz-6000_type-reservoir_sample-random_k-16",
                       "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                       "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    forgetting, fwts, fps, xs = read_boxplot_contents(option, option2)
    forgetting_m, fwts_m, fps_m, xs_m = read_boxplot_contents(option, option2="multiple_steps")

    box_plot_data = []
    lang_order_box_plot_data = []
    method_box_plot_data = []
    for method in model_names:
        if option3 == "forgetting":
            ys = forgetting[method]
        elif option3 == "fwt":
            ys = fwts[method]
        else:
            ys = fps[method]
        lang_order_box_plot_data.extend(6*[1, 2, 3, 4, 5, 6])
        method_box_plot_data.extend(6*[model_names_dict[method]])
        box_plot_data.append(ys)

    ##
    box_plot_data_m = []
    lang_order_box_plot_data_m = []
    method_box_plot_data_m = []
    for method in model_names:
        if method == "kd-logits_memsz-6000_type-ring_sample-random_k-16":
            method = "kd-logits_memsz-6000_type-reservoir_sample-random_k-16"
        if option3 == "forgetting":
            ys_m = forgetting_m[method]
        elif option3 == "fwt":
            ys_m = fwts_m[method]
        else:
            ys_m = fps_m[method]
        lang_order_box_plot_data_m.extend(6*[1, 2, 3, 4, 5, 6])
        method_box_plot_data_m.extend(6*[model_names_dict[method]+" Z"])
        box_plot_data_m.append(ys_m)

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Lang-Spec Enc[0-8]': '#029e73',
                      'Lang-Spec Trans': '#d55e00',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Logit': '#ece133',
                      'KD-Rep': '#56b4e9',
                      'Naive Seq FT Z': '#0173b2',
                      'Lang-Spec Enc[0-8] Z': '#029e73',
                      'Lang-Spec Trans Z': '#d55e00',
                      'EWC-Online Z': '#fbafe4',
                      'ER Z': '#949494',
                      'KD-Logit Z': '#ece133',
                      'KD-Rep Z': '#56b4e9'}#, '#fbafe4', '#949494', '#ece133', '#56b4e9'}

    # sns.boxplot(x=method_box_plot_data, y=box_plot_data, hue=method_box_plot_data, palette=custom_palette,
    #                showfliers=False)
    #
    # print("method_box_plot_data_m:", method_box_plot_data_m)
    # sns.boxplot(x=method_box_plot_data_m, y=box_plot_data_m, palette=custom_palette,
    #             width=0.3, showfliers=False)

    # data = pd.DataFrame(rnd.randn(10, 4))
    # print("data:", rnd.randn(10, 4))
    # data.boxplot(positions=[1, 5, 6, 10], showfliers=False)
    # print("box_plot_data:", box_plot_data)
    # print("box_plot_data_m:", box_plot_data_m)

    # data = pd.DataFrame(np.transpose(np.array(box_plot_data)))
    # bplot1 = data.boxplot(positions=[0, 2, 4, 6, 8, 10], showfliers=False)
    # for patch, color in zip(bplot1['boxes'], [custom_palette[method]
    #                         for method in ['Naive Seq FT', 'Lang-Spec Enc[0-8]', 'Lang-Spec Trans', 'EWC-Online',
    #                                        'ER', 'KD-Logit', 'KD-Rep']]):
    #     patch.set_facecolor(color)
    bplot1 = plt.boxplot(box_plot_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         # positions=[0, 2, 4, 6, 8, 10],
                         positions=[0, 1, 2, 3, 4, 5],
                         # labels=["0", "1", "2", "3", "4", "5"],
                         labels=["", "", "", "", "", ""],
                         showfliers=False)  # will be used to label x-ticks

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bplot1[element], color="black")

    for patch, color in zip(bplot1['boxes'], [custom_palette[model_names_dict[method]]
                                              for method in model_names]):
        patch.set_facecolor(color)
        patch.set_hatch("x")

    # data_m = pd.DataFrame(np.transpose(np.array(box_plot_data_m)))
    # data_m.boxplot(positions=[1, 3, 5, 7, 9, 11], showfliers=False)
    # plt.grid(False)
    # sns.boxplot(x=method_box_plot_data, y=box_plot_data, positions=[1, 5, 6, 10])
    bplot2 = plt.boxplot(box_plot_data_m,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         # positions=[1, 3, 5, 7, 9, 11],
                         positions=[6, 7, 8, 9, 10, 11],
                         # labels=["0", "1", "2", "3", "4", "5"],
                         # labels=["", "", "", "", "", ""],
                         labels=["", "", "", "", "", ""],
                         showfliers=False)  # will be used to label x-ticks

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bplot2[element], color="black")


    for patch, color in zip(bplot2['boxes'], [custom_palette[model_names_dict[method]]
                                              for method in model_names]):
        patch.set_facecolor(color)
        patch.set_hatch(".")

    # plt.axvline(x=1.5, color='gray')
    # plt.axvline(x=3.5, color='gray')
    plt.axvline(x=5.5, color='gray')
    # plt.axvline(x=7.5, color='gray')
    # plt.axvline(x=9.5, color='gray')

    naive_seq_ft = mpatches.Patch(facecolor=[1/255, 115/255, 178/255, 1], label='Naive Seq FT') # '#0173b2',
    lang_spec_enc = mpatches.Patch(facecolor=[2/255, 158/255, 115/255, 1], label='Lang-Spec Enc[0-8]') # '#029e73',
    lang_spec_trans = mpatches.Patch(facecolor=[213/255, 94/255, 0/255, 1], label='Lang-Spec Trans') #  '#d55e00',
    lang_spec_ada_tuned = mpatches.Patch(facecolor=[204/255, 120/255, 188/255, 1], label='Lang-Spec Ada(T)') # '#cc78bc',
    ewc_online = mpatches.Patch(facecolor=[251/255, 175/255, 228/255, 1], label='EWC-Online') # '#fbafe4',
    er = mpatches.Patch(facecolor=[148/255, 148/255, 148/255, 1], label='ER') # '#949494',
    kd_logit = mpatches.Patch(facecolor=[236/255, 225/255, 51/255, 1], label='KD-Logit') # '#ece133',

    plt.legend(handles=[lang_spec_trans, lang_spec_enc, er, kd_logit, ewc_online, naive_seq_ft],
               # loc='upper right')
               bbox_to_anchor=(0., 1.0, 0.9, .102),
               loc=3, frameon=False,
               ncol=3, mode="expand", borderaxespad=0., prop={"size": 20, "family": "Times"})

    plt.xlabel('Approaches', fontsize=size_text, family="Times")
    if option3 == "forgetting":
        ylabel = "Forgetting"
    elif option3 == "fwt":
        ylabel = "Transfer"
    else:
        ylabel = "Final Performance"

    if option == "intents":
        ylabel += " of Accuracy Scores"
    else:
        ylabel += " of F1 Scores"

    plt.ylabel(ylabel, fontsize=size_text, family="Times")

    # plt.xticks(np.array([0.7, 2.5, 4.5, 6.5, 8.5, 10.5]),
    #            [model_names_dict[model] for model in model_names],
    #            fontsize=13, family="Times", rotation=0)

    plt.xticks(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
               [model_names_dict[model] for model in 2*model_names],
               fontsize=13, family="Times", rotation=90)

    plt.yticks(size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    plt.tight_layout()

    # my_cmap = sns.color_palette("colorblind", 3, as_cmap=True)
    # sns.palplot(my_cmap)

    # plt.show()
    plot_save_path = ""
    filename = plot_save_path+"all/"+option2+"_"+option3

    if option == "slots":
        filename = filename + "_slots"

    plt.savefig(filename+".png", format="png")


def plot_fwt_k():
    alias = "fwt_k"

    all_per_order_metrics, all_avg_metrics = read_from_pickle_short(root_dir, alias)

    intents_fwts = [0.0 for _ in range(5)] #{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    slots_fwts = [0.0 for _ in range(5)] #{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    fig = plt.figure(figsize=(12, 7.3))
    ax = fig.add_subplot(1, 1, 1)

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Inc Joint': '#de8f05',
                      'Lang-Spec Enc[0-8]': '#029e73',
                      'Lang-Spec Trans': '#d55e00',
                      'Lang-Spec Ada(T)': '#cc78bc',
                      'Lang-Spec Ada(F)':  '#ca9161',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Logit': '#ece133',
                      'KD-Rep': '#56b4e9'}

    for avg_metrics in all_avg_metrics:
        print("avg_metrics:", avg_metrics, " len(avg_metrics):", len(avg_metrics))
        method = avg_metrics[0]
        if model_names_dict[method] not in ["Lang-Spec Enc[0-8]", "Lang-Spec Trans", "Lang-Spec Ada(F)", "Lang-Spec Task"]:
            # intents_fwts[0] = avg_metrics[6]
            # slots_fwts[0] = avg_metrics[8]

            # intents_fwts[1] = avg_metrics[10]
            # slots_fwts[1] = avg_metrics[12]
            #
            # intents_fwts[2] = avg_metrics[14]
            # slots_fwts[2] = avg_metrics[16]
            #
            # intents_fwts[3] = avg_metrics[18]
            # slots_fwts[3] = avg_metrics[20]
            #
            # intents_fwts[4] = avg_metrics[22]
            # slots_fwts[4] = avg_metrics[24]
            #
            # intents_fwts[5] = avg_metrics[26]
            # slots_fwts[5] = avg_metrics[28]

            intents_fwts[0] = avg_metrics[10]
            slots_fwts[0] = avg_metrics[12]

            intents_fwts[1] = avg_metrics[14]
            slots_fwts[1] = avg_metrics[16]

            intents_fwts[2] = avg_metrics[18]
            slots_fwts[2] = avg_metrics[20]

            intents_fwts[3] = avg_metrics[22]
            slots_fwts[3] = avg_metrics[24]

            intents_fwts[4] = avg_metrics[26]
            slots_fwts[4] = avg_metrics[28]

            print("method:", method)
            plt.plot(list(range(1, 6)), intents_fwts, '-p', label=model_names_dict[method],
                     color=custom_palette[model_names_dict[method]],)
            plt.legend(bbox_to_anchor=(0., 0.95, 1., .102), loc=3, frameon=False,
                       ncol=3, mode="expand", borderaxespad=0., prop={"size": 20})

    plt.xlabel('After # languages', fontsize=25, family="Times")
    plt.ylabel('Zero-Shot Transfer of Accuracy Scores', fontsize=25, family="Times")
    plt.xticks(np.array(list(range(1, 6))), ('1', '2', '3', '4', '5'), size=23)
    plt.yticks(np.arange(44, 56, 2), size=20, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # plt.show()
    filename = plot_save_path+"all/FWT_k_intents"

    plt.savefig(filename+".png", format="png")


def plot_fwt_k_s():
    alias = "fwt_k"

    all_per_order_metrics, all_avg_metrics = read_from_pickle_short(root_dir, alias)

    intents_fwts = [0.0 for _ in range(5)] #{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    slots_fwts = [0.0 for _ in range(5)] #{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    fig = plt.figure(figsize=(12, 7.3))
    ax = fig.add_subplot(1, 1, 1)

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Inc Joint': '#de8f05',
                      'Lang-Spec Enc[0-8]': '#029e73',
                      'Lang-Spec Trans': '#d55e00',
                      'Lang-Spec Ada(T)': '#cc78bc',
                      'Lang-Spec Ada(F)':  '#ca9161',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Logit': '#ece133',
                      'KD-Rep': '#56b4e9'}

    for avg_metrics in all_avg_metrics:
        print("avg_metrics:", avg_metrics, " len(avg_metrics):", len(avg_metrics))
        method = avg_metrics[0]
        if model_names_dict[method] not in ["Lang-Spec Enc[0-8]", "Lang-Spec Trans", "Lang-Spec Ada(F)", "Lang-Spec Task"]:
            # intents_fwts[0] = avg_metrics[6]
            # slots_fwts[0] = avg_metrics[8]

            # intents_fwts[1] = avg_metrics[10]
            # slots_fwts[1] = avg_metrics[12]
            #
            # intents_fwts[2] = avg_metrics[14]
            # slots_fwts[2] = avg_metrics[16]
            #
            # intents_fwts[3] = avg_metrics[18]
            # slots_fwts[3] = avg_metrics[20]
            #
            # intents_fwts[4] = avg_metrics[22]
            # slots_fwts[4] = avg_metrics[24]
            #
            # intents_fwts[5] = avg_metrics[26]
            # slots_fwts[5] = avg_metrics[28]

            intents_fwts[0] = avg_metrics[10]
            slots_fwts[0] = avg_metrics[12]

            intents_fwts[1] = avg_metrics[14]
            slots_fwts[1] = avg_metrics[16]

            intents_fwts[2] = avg_metrics[18]
            slots_fwts[2] = avg_metrics[20]

            intents_fwts[3] = avg_metrics[22]
            slots_fwts[3] = avg_metrics[24]

            intents_fwts[4] = avg_metrics[26]
            slots_fwts[4] = avg_metrics[28]

            plt.plot(list(range(1, 6)), slots_fwts, '-p', label=model_names_dict[method],
                     color=custom_palette[model_names_dict[method]])
            # plt.plot(list(range(1, 6)), slots_fwts, '-p', label=method+"_slots")
            # plt.legend()
            plt.legend(bbox_to_anchor=(0., 0.95, 1., .102), loc=3, frameon=False,
                       ncol=3, mode="expand", borderaxespad=0., prop={"size": 20})

    plt.xlabel('After # languages', fontsize=25, family="Times")
    plt.ylabel('Zero-Shot Transfer of F1 Scores', fontsize=25, family="Times")
    plt.xticks(np.array(list(range(1, 6))), ('1', '2', '3', '4', '5'), size=23, family="Times")
    plt.yticks(size=20, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # plt.show()
    filename = plot_save_path+"all/FWT_k_slots"
    plt.savefig(filename+".png", format="png")


def plot_average_order_langs(option):
    with open(root_dir+"per_lang_rank/intents_avg_all.pickle", "rb") as file:
        per_lang_rank_intents_avg_all = pickle.load(file)

    with open(root_dir+"per_lang_rank/slots_avg_all.pickle", "rb") as file:
        per_lang_rank_slots_avg_all = pickle.load(file)

    print("per_lang_rank_intents_avg_all:", len(per_lang_rank_intents_avg_all))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    rand_intent_langs = [1.46, 0.19, 0.75, 1.39, 0.17, 0.68]
    rand_slot_langs = [0.16, 0.18, 0.13, 0.21, 0.17, 0.15]

    pal = sns.color_palette("colorblind")
    # sns.palplot(pal)
    # plt.show()
    colors = pal.as_hex()
    print("colors:", colors)

    model_names = ["vanilla",
                   "adapters/TUNED_BERT"]

    model_names_more = ["multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                        "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                        # "multi_head_out",
                        "joint_plus",
                        "adapters/FROZEN_BERT",
                        "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                        "kd-logits_memsz-6000_type-ring_sample-random_k-16",
                        "kd-rep_memsz-6000_type-ring_sample-random_k-16"]

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Inc Joint': '#de8f05',
                      'Lang-Spec Enc[0-8]': '#029e73',
                      'Lang-Spec Trans': '#d55e00',
                      'Lang-Spec Ada(T)': '#cc78bc',
                      'Lang-Spec Ada(F)':  '#ca9161',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Rep': '#ece133',
                      'KD-Logit': '#56b4e9'}

    # for j, lang in enumerate(languages[5:6]):
    #     for i, per_lang_rank_intents_avg in enumerate(per_lang_rank_intents_avg_all):
    #         if i == 0:
    #             list_intents_lang = [per_lang_rank_intents_avg[lang][rank] for rank in per_lang_rank_intents_avg[lang]]
    #             per_lang_rank_slots_avg = per_lang_rank_slots_avg_all[i]
    #             list_slots_lang = [per_lang_rank_slots_avg[lang][rank] for rank in per_lang_rank_slots_avg[lang]]
    #             list_ranks = [rank for rank in per_lang_rank_intents_avg[lang]]
    #
    #             if option == "intents":
    #                 plt.plot(list_ranks, list_intents_lang, '-p', label=languages_elonged_dict[lang])
    #             else:
    #                 plt.plot(list_ranks, list_slots_lang, '-p', label=languages_elonged_dict[lang])
    #
    #             # plt.plot(list_ranks, list_intents_lang, '--', color=custom_palette[model_names_dict[model_names[i]]], label=model_names_dict[model_names[i]])
    #
    #
    # for j, lang in enumerate(languages[3:4]):
    #     for i, per_lang_rank_intents_avg in enumerate(per_lang_rank_intents_avg_all):
    #         if i == 0:
    #             list_intents_lang = [per_lang_rank_intents_avg[lang][rank] for rank in per_lang_rank_intents_avg[lang]]
    #             per_lang_rank_slots_avg = per_lang_rank_slots_avg_all[i]
    #             list_slots_lang = [per_lang_rank_slots_avg[lang][rank] for rank in per_lang_rank_slots_avg[lang]]
    #             list_ranks = [rank for rank in per_lang_rank_intents_avg[lang]]
    #
    #             # plt.plot(list_ranks, list_intents_lang, '+-', color=custom_palette[model_names_dict[model_names[i]]], label=model_names_dict[model_names[i]])
    #             if option == "intents":
    #                 plt.plot(list_ranks, list_intents_lang, '-p', label=languages_elonged_dict[lang])
    #             else:
    #                 plt.plot(list_ranks, list_slots_lang, '-p', label=languages_elonged_dict[lang])

    for j, lang in enumerate(languages):
        for i, per_lang_rank_intents_avg in enumerate(per_lang_rank_intents_avg_all):
            if i == 0:
                list_intents_lang = [per_lang_rank_intents_avg[lang][rank] for rank in per_lang_rank_intents_avg[lang]]
                per_lang_rank_slots_avg = per_lang_rank_slots_avg_all[i]
                list_slots_lang = [per_lang_rank_slots_avg[lang][rank] for rank in per_lang_rank_slots_avg[lang]]
                list_ranks = [rank for rank in per_lang_rank_intents_avg[lang]]

                # plt.plot(list_ranks, list_intents_lang, '-p', color=custom_palette[model_names_dict[model_names[i]]], label=model_names_dict[model_names[i]])
                if option == "intents":
                    plt.plot(list_ranks, list_intents_lang, '-p', label=languages_elonged_dict[lang])
                else:
                    plt.plot(list_ranks, list_slots_lang, '-p', label=languages_elonged_dict[lang])


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, bbox_to_anchor=(0., 0.95, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., prop={"size": 18, "family":"Times"})

    plt.xlabel('After # languages', fontsize=25, family="Times")
    plt.xticks(np.array([1, 2, 3, 4, 5]), ('1', '2', '3', '4', '5'), size=23, family="Times")
    if option == "intents":
        plt.ylabel('Accuracy Scores', fontsize=25, family="Times")
        plt.yticks(np.arange(0, 90, 10), size=20)
    else:
        plt.ylabel('F1 Scores', fontsize=25, family="Times")
        plt.yticks(np.arange(0, 70, 10), size=20)

    plt.xticks(size=20)

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # plt.show()
    filename = plot_save_path+"all/PerLangPerf"

    if option == "intents":
        filename = filename + "Intents"
    else:
        filename = filename + "Slots"

    print(filename+".png")
    plt.savefig(filename+".png", format="png")


def plot_order_langs(option="intents"):
    with open(root_dir+"per_lang_rank/intents_avg_all.pickle", "rb") as file:
        per_lang_rank_intents_avg_all = pickle.load(file)

    with open(root_dir+"per_lang_rank/slots_avg_all.pickle", "rb") as file:
        per_lang_rank_slots_avg_all = pickle.load(file)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(1, 1, 1)

    # rand_intent_langs = [random["random_perf"][lang][0] for lang in languages]
    # rand_slot_langs = [random["random_perf"][lang][1] for lang in languages]

    # rand_intent_langs = [1.46, 0.19, 0.75, 1.39, 0.17, 0.68] # TODO REPLACE
    # rand_slot_langs = [0.16, 0.18, 0.13, 0.21, 0.17, 0.15] # TODO REPLACE

    pal = sns.color_palette("colorblind")
    # sns.palplot(pal)
    # plt.show()
    colors = pal.as_hex()
    print("colors:", colors)

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

    custom_palette = {'Naive Seq FT': '#0173b2',
                      'Inc Joint': '#de8f05',
                      'Lang-Spec Enc[0-8]': '#029e73',
                      'Lang-Spec Trans': '#d55e00',
                      'Lang-Spec Ada(T)': '#cc78bc',
                      'Lang-Spec Ada(F)':  '#ca9161',
                      'EWC-Online': '#fbafe4',
                      'ER': '#949494',
                      'KD-Logit': '#ece133',
                      'KD-Rep': '#56b4e9'}

    # for set in [languages[5:6], languages[3:4], languages[2:3]]:
    per_language_intents = {method: [] for method in model_names} #{method: []}
    for j, lang in enumerate(languages):
        for i, per_lang_rank_intents_avg in enumerate(per_lang_rank_intents_avg_all):
            if option == "intents":
                list_intents_lang = [per_lang_rank_intents_avg[lang][rank] for rank in per_lang_rank_intents_avg[lang]]
                per_language_intents[model_names[i]].append(list_intents_lang)
            else:
                per_lang_rank_slots_avg = per_lang_rank_slots_avg_all[i]
                list_slots_lang = [per_lang_rank_slots_avg[lang][rank] for rank in per_lang_rank_slots_avg[lang]]
                per_language_intents[model_names[i]].append(list_slots_lang)

    list_ranks = [1, 2, 3, 4, 5]
    for method in per_language_intents:
        if method not in ["multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                          "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                          "adapters/FROZEN_BERT",
                          "multi_head_out",
                          "joint_plus"]:
            # list_slots_lang = [per_lang_rank_slots_avg[lang][rank] for rank in per_lang_rank_slots_avg[lang]]
            print("method:", method, " per_language_intents:", per_language_intents[method])
            print("np.mean(np.array(per_language_intents[method]), axis=0):", np.mean(np.array(per_language_intents[method]), axis=0))

            print("per_language_intents[method]:", per_language_intents[method])

            plt.plot(list_ranks,
                     np.mean(np.array(per_language_intents[method]), axis=0),
                     '-p',
                     color=custom_palette[model_names_dict[method]],
                     label=model_names_dict[method])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, bbox_to_anchor=(0., 0.95, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., prop={"size": 25, "family": "Times"})

    plt.xlabel('# Hops (After Training on # languages)', fontsize=size_text, family="Times")
    plt.xticks(np.array([1, 2, 3, 4, 5]), ('1', '2', '3', '4', '5'), size=size_text, family="Times")
    if option == "intents":
        ylabel = 'Accuracy Scores'
    else:
        ylabel = 'F1 Scores'

    plt.ylabel(ylabel, fontsize=size_text, family="Times")

    # plt.xticks(size=size_text, family="Times")
    if option == "intents":
        plt.yticks(np.arange(38, 55, 2), size=size_text, family="Times")
    else:
        plt.yticks(np.arange(24, 42, 2), size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # plt.show()
    filename = plot_save_path+"all/PerLangPer"+option

    if option == "slots":
        filename = filename + "_slots"

    plt.savefig(filename+".png", format="png")


def read_forgetting_fwt_0():
    aliases = ["all"]

    all_per_order_metrics, all_avg_metrics, all_forget_lang_strings, all_fwt_lang_strings, all_fwt_mono_lang_strings \
        = read_from_pickle_results(root_dir, aliases)

    forgetting = {"intents": [], "slots": []}
    fwt_0 = {"intents": [], "slots": []}

    labels = []
    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        if method not in [#"multi_head_out",
                          #"Lang-Spec Ada(T)",
                          "er_memsz-750_type-reservoir_sample-random_k-16",
                          "er_memsz-1500_type-reservoir_sample-random_k-16",
                          "er_memsz-3000_type-reservoir_sample-random_k-16",
                          "er_memsz-4500_type-reservoir_sample-random_k-16",
                          "multi_head_inembed",
                          "multi_head_inenc.0-enc.1-enc.2",
                          "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5",
                          "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11",
                          "multi_head_inenc.3-enc.4-enc.5",
                          "multi_head_inenc.6-enc.7-enc.8",
                          "multi_head_inenc.6-enc.7-enc.8-enc.9-enc.10-enc.11",
                          "multi_head_inenc.9-enc.10-enc.11",
                          "joint_plus"]:
            labels.append(model_names_dict[method])
            forgetting["intents"].append(-avg_metric[2])
            fwt_0["intents"].append(avg_metric[6])

            forgetting["slots"].append(-avg_metric[4])
            fwt_0["slots"].append(avg_metric[8])


    return forgetting, fwt_0, labels


def plot_generalization_transfer(option):
    forgetting, fwt_0, labels = read_forgetting_fwt_0()
    list_a = forgetting[option]
    list_b = fwt_0[option]
    list_c = labels

    not_zoomed = ["Lang-Spec Enc[0-8]",
                  "Lang-Spec Trans",
                  "Lang-Spec Task",
                  "Lang-Spec Ada(T)",
                  "Lang-Spec Ada(F)"]
    print("labels:", list_c)

    ylabel = "Zero-shot Generalization of"
    xlabel = "Negative Forgetting of"

    if option == "intents":
        ylabel += " Accuracy Scores"
        xlabel += " Accuracy Scores"
    else:
        ylabel += " F1 Scores"
        xlabel += " F1 Scores"

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1)

    # list_a[:], list_b[:] = zip(*sorted(zip(list_a, list_b), key=lambda p: (p[0], p[1])))
    list_a_,  list_b_, list_c_ = [list_a[i] for i in range(len(list_a)) if list_c[i] not in not_zoomed], \
                                 [list_b[i] for i in range(len(list_b)) if list_c[i] not in not_zoomed], \
                                 [list_c[i] for i in range(len(list_c)) if list_c[i] not in not_zoomed]
    plt.scatter(list_a_, list_b_, label=option, color="black")

    texts = []
    for x, y, s in zip(list_a_, list_b_, list_c_):
        # if s in ["Lang-Spec Enc[0-8]", "Lang-Spec Task", "Lang-Spec Trans"]:
        #     c = "red"
        # else:
        #     c = "green"
        c = "green"
        texts.append(plt.text(x, y, s, size=23, color=c, family="Times"))

    adjust_text(texts,
                only_move={'points': 'y', 'texts': 'y'},
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

    plt.xlabel(xlabel, fontsize=size_text, family="Times")
    plt.ylabel(ylabel, fontsize=size_text, family="Times")

    plt.xticks(size=size_text, family="Times")
    plt.yticks(size=size_text, family="Times")

    ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_axisbelow(True)
    right_side = ax.spines["right"]
    right_side.set_visible(False)

    upper_side = ax.spines["top"]
    upper_side.set_visible(False)

    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    # location for the zoomed portion
    # sub_axes = plt.axes([.6, .6, .25, .25])
    sub_axes = plt.axes([.62, .2, .35, .35])
    rect = patches.Rectangle((50, 60), -3, 0, linewidth=1, edgecolor='r', facecolor='gray')

    # Add the patch to the Axes
    ax.add_patch(rect)
    # sub_axes.set_position([1, 2, 3, 4])

    # not_zoomed = ["Lang-Spec Enc[0-8]",
    #               "Lang-Spec Trans",
    #               "Lang-Spec Task",
    #               "Lang-Spec Ada(T)"]
    not_zoomed = []

    X_detail = [a for i, a in enumerate(list_a) if list_c[i] not in not_zoomed]

    # Y_detail = [b for i, b in list_b if list_c[i] not in ["Lang-Spec Enc[0-8]", "Lang-Spec Trans", "Lang-Spec Task"]]
    Y_detail = [b for i, b in enumerate(list_b) if list_c[i] not in not_zoomed]

    labels_detail = [c for i, c in enumerate(list_c) if list_c[i] not in not_zoomed]

    # # plot the zoomed portion
    sub_axes.scatter(X_detail, Y_detail, c='k', s=4)

    texts = []
    for x, y, s in zip(X_detail, Y_detail, labels_detail):
        if s in ["Lang-Spec Enc[0-8]", "Lang-Spec Task", "Lang-Spec Trans", "Lang-Spec Ada(T)", "Lang-Spec Ada(F)"]:
            c = "red"
        else:
            c = "green"
        texts.append(plt.text(x, y, s, size=12, color=c, family="Times"))

    adjust_text(texts,
                only_move={'points': 'y', 'texts': 'y'},
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

    plt.tight_layout()

    # plt.show()

    filename = plot_save_path+"all/q5_transfer_generalization"

    if option == "slots":
        filename = filename + "_slots"

    plt.savefig(filename+".png", format="png")


if __name__ == "__main__":

    # try_seaborn()
    # plot_base_orders()
    # plot_cont_methods_avg_analysis()
    # Figure 3
    # plot_cont_methods_avg_analysis_seaborn(option="intents")
    # plot_cont_methods_avg_analysis_seaborn(option="slots")
    # # # Figure 4
    # plot_cont_methods_lang_order_seaborn(option="intents")
    # plot_cont_methods_lang_order_seaborn(option="slots")
    # # Figure 5
    # plot_transfer_interference_fp_curve(option="intents", option2="fwt")
    # plot_transfer_interference_fp_curve(option="intents", option2="fp")
    # plot_transfer_interference_fp_curve(option="intents", option2="fp", option3="fwt")
    # plot_transfer_interference_fp_curve(option="slots", option2="fwt")
    # plot_transfer_interference_fp_curve(option="slots", option2="fp")
    # plot_transfer_interference_fp_curve(option="slots", option2="fp", option3="fwt")
    #
    # ## Generalization
    # plot_generalization_transfer(option="intents")
    # plot_generalization_transfer(option="slots")

    # # Figure 7
    # # boxplot_two_steps(option="intents", option2="two_steps", option3="forgetting")
    # # boxplot_two_steps(option="intents", option2="multiple_steps", option3="forgetting")
    boxplot_merged(option="intents", option2="two_steps", option3="forgetting")
    boxplot_merged(option="slots", option2="two_steps", option3="forgetting")
    boxplot_merged(option="intents", option2="two_steps", option3="fwt")
    boxplot_merged(option="slots", option2="two_steps", option3="fwt")
    boxplot_merged(option="intents", option2="two_steps", option3="fp")
    boxplot_merged(option="slots", option2="two_steps", option3="fp")
    # plot_two_steps(option="intents", option2="two_steps", option3="forgetting")
    # plot_two_steps(option="intents", option2="two_steps", option3="fwt")
    # plot_two_steps(option="intents", option2="two_steps", option3="fp")
    #
    # plot_two_steps(option="slots", option2="two_steps", option3="forgetting")
    # plot_two_steps(option="slots", option2="two_steps", option3="fwt")
    # plot_two_steps(option="slots", option2="two_steps", option3="fp")
    #
    # plot_two_steps(option="intents", option2="multiple_steps", option3="forgetting")
    # plot_two_steps(option="intents", option2="multiple_steps", option3="fwt")
    # plot_two_steps(option="intents", option2="multiple_steps", option3="fp")
    #
    # plot_two_steps(option="slots", option2="multiple_steps", option3="forgetting")
    # plot_two_steps(option="slots", option2="multiple_steps", option3="fwt")
    # plot_two_steps(option="slots", option2="multiple_steps", option3="fp")

    # # FIGURE 11 (a, b)
    # plot_fwt_k()
    # plot_fwt_k_s()
    #
    # # FIGURE 11 (c, d)
    # plot_order_langs(option="intents")
    # plot_order_langs(option="slots")

    # plot_average_order_langs(option="intents")
    # plot_average_order_langs(option="slots")
