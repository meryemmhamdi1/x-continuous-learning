import pickle
from matplotlib import pyplot as plt
import numpy as np
import math
import os


# layer view
def plot_layer_view(changes, layer_key, model_name):
    n_groups = len(changes)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.6
    opacity = 0.8

    plt.bar(index,
            changes,
            bar_width,
            alpha=opacity,
            color='g',
            label=model_name)

    plt.xlabel('Training Language')
    plt.ylabel(layer_key)
    plt.xticks(index + bar_width, ('EN', 'DE', 'FR', 'HI', 'ES', 'TH'))
    #plt.yticks(np.arange(0, 100, 10))
    plt.legend(loc="upper right")

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join("/Users/d22admin/USCGDrive/Spring21/Research/XContLearn/Results/cll/vanilla/"
                "high2low/multi_head_in/SEED_42/WeightChangesPlots/layer_view/", layer_key+".png"))


# language view
def plot_language_view(changes, lang, layer_keys, model_name, group=None):
    n_groups = len(changes)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8

    plt.bar(index,
            changes,
            bar_width,
            alpha=opacity,
            color='g',
            label=model_name)

    plt.xlabel('Layer')
    plt.ylabel('Change in weights for lang '+lang)
    plt.xticks(index + bar_width, tuple(layer_keys), rotation=90)
    #plt.yticks(np.arange(0, 100, 10))
    #plt.legend(loc="upper right")

    plt.tight_layout()
    #plt.show()
    save_path = lang
    if group:
        save_path += "_group"

    plt.savefig(os.path.join("/Users/d22admin/USCGDrive/Spring21/Research/XContLearn/Results/cll/vanilla/"
                             "high2low/multi_head_in/SEED_42/WeightChangesPlots/language_view/", save_path+".png"))


if __name__ == "__main__":
    langs = ["EN", "DE", "FR", "HI", "ES", "TH"]
    order = "high2low"
    multi = "multi_head_in"
    with open("/Users/d22admin/USCGDrive/Spring21/Research/XContLearn/Results/cll/vanilla/"+order+"/"+multi+
              "/SEED_42/mean_all_stream_mbert.pickle", "rb") as file:
        data = pickle.load(file)

    keys_list = list(data[0].keys())
    main_encoder_layers = ["embeddings",
                           "encoder.layer.0",
                           "encoder.layer.1",
                           "encoder.layer.2",
                           "encoder.layer.3",
                           "encoder.layer.4",
                           "encoder.layer.5",
                           "encoder.layer.6",
                           "encoder.layer.7",
                           "encoder.layer.8",
                           "encoder.layer.9",
                           "encoder.layer.10",
                           "encoder.layer.11",
                           "pooler.dense",
                           "intent",
                           "slot",
                           "mbert"]

    model_name = multi + " " + order

    count_per_layer = {key: 0 for key in main_encoder_layers}

    layers_weights = {key: [0.0 for _ in range(len(data))] for key in data[0].keys()}
    group_layers_weights = {key: [0.0 for _ in range(len(data))] for key in main_encoder_layers}

    languages_weights = {i: [0.0 for k in data[0].keys()] for i in range(len(data))}
    lang_layers_weights = {i: {k: 0.0 for k in main_encoder_layers} for i in range(len(data))}

    for i in range(len(data)):
        for k, v in data[i].items():
            if v != 0:
                v = math.log(abs(v))
            layers_weights[k][i] = -1 * v
            languages_weights[i][keys_list.index(k)] = -1 * v

            for key in main_encoder_layers:
                if key in k:
                    count_per_layer[key] += 1
                    lang_layers_weights[i][key] += -1 * v
                    group_layers_weights[key][i] += -1 * v

    for i in range(len(lang_layers_weights)):
        for key in lang_layers_weights[i]:
            lang_layers_weights[i][key] = lang_layers_weights[i][key]/count_per_layer[key]
            group_layers_weights[key][i] = group_layers_weights[key][i]/count_per_layer[key]

    # print("Plotting Layer_view")
    # for layer_key in layers_weights:
    #     print(layer_key, layers_weights[layer_key])
    #     plot_layer_view(layers_weights[layer_key], layer_key, model_name)
    #
    # print("Plotting Group_Layer_view")
    # for layer_key in group_layers_weights:
    #     print(layer_key, group_layers_weights[layer_key])
    #     plot_layer_view(group_layers_weights[layer_key], layer_key, model_name)

    # print("Plotting Language_view")
    # for i, lang in enumerate(langs):
    #     print(lang, languages_weights[i])
    #     plot_language_view(languages_weights[i], lang, keys_list, model_name)

    print("Plotting Lang_group_layer_view")
    for i, lang in enumerate(langs):
        print(lang, lang_layers_weights[i])
        plot_language_view([v for k, v in lang_layers_weights[i].items()], lang, main_encoder_layers, model_name, "group")



