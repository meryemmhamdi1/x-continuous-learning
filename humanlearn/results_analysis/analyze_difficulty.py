import os
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import collections
import configparser
import sys

sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
from sys import platform

from src.data_utils import MultiPurposeDataset, AugmentedList
from src.transformers_config import MODELS_dict
from src.consts import DOMAIN_TYPES, INTENT_TYPES, NAVA_TAGS, NOISY_WORDS, SPLIT_NAMES


paths = configparser.ConfigParser()
paths.read("scripts/paths.ini")

if platform == "darwin":
    location = "LOCAL"
else:
    location = "CARC"

out_dir = str(paths.get(location, "OUT_DIR"))

LANG_ORDER = "en_de_hi_th"

PATH_MOST_FREQUENT_DOMAINS = os.path.join(
    out_dir,
    LANG_ORDER + "_most_frequent_words_domains.pickle",
)

PATH_MOST_FREQUENT_INTENTS = os.path.join(
    out_dir,
    LANG_ORDER + "_most_frequent_words_intents.pickle",
)

ER_STRATEGY = "easy"


class DataArgs:
    def __init__(self):
        self.data_root = (
            # "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Datasets/"
        )
        self.task_name = "tod"
        self.data_name = "mtop"
        self.data_format = "txt"
        self.data_root = os.path.join(
            str(paths.get(location, "DATA_ROOT")),
            self.task_name.upper(),
            self.data_name.upper(),
        )
        self.setup_opt = "cll-er_kd"
        self.order_lst = "en_de_hi_th"
        self.languages = ["en", "de", "hi", "th"]
        self.order_class = 0
        self.order_lang = 0
        self.trans_model = "BertBaseMultilingualCased"
        self.use_slots = True
        self.seed = 42
        self.num_class_tasks = 10
        self.num_lang_tasks = 2
        self.multi_head_out = False
        self.max_mem_sz = 100
        self.er_strategy = "random"
        self.order_class = 0


### 1. For every language and intent, take all examples and strip out stop words and calculate the most three appearing words for that intent and do that for all intents. (do it for each domain look for most frequent words and for each intent look for most frequent words that is not within the most frequent words for intents.)
args = DataArgs()

tokenizer = MODELS_dict[args.trans_model][1].from_pretrained(
    MODELS_dict[args.trans_model][0], do_lower_case=True, do_basic_tokenize=False
)

dataset = MultiPurposeDataset(args, tokenizer)
lemmatizer = WordNetLemmatizer()

class_types = dataset.class_types


def extract_only_nava_words_per_sentence(sentence):
    """
    Expected one sentence
    """
    navas = []
    for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
        if pos in NAVA_TAGS:
            navas.append(lemmatizer.lemmatize(word.lower()))
    return navas


def extract_only_nava_words(sentences):
    """
    Expected sentences from a given domain or intent
    """
    navas = []

    for sentence in sentences:
        navas.extend(extract_only_nava_words_per_sentence(sentence))
    return navas


def find_most_frequent_words_intents_domains(languages):
    sentences_domains = {domain.lower(): [] for domain in DOMAIN_TYPES[args.data_name]}
    sentences_intents = {intent: [] for intent in INTENT_TYPES[args.data_name]}

    #### Look for most frequent words within a domain for each language
    most_frequent_words_domains = {domain: [] for domain in sentences_domains}
    most_frequent_words_intents = {intent: [] for intent in sentences_intents}

    #### Stores all possible keywords of the sentences
    for lang in languages.split("_"):
        # Do it per language
        for eg in dataset.train_set[lang]:
            domain = class_types[eg.label].split(":")[0]
            sentences_domains[domain].append(eg.text)
            sentences_intents[class_types[eg.label]].append(eg.text)

        for domain in sentences_domains:
            domain_words = extract_only_nava_words(sentences_domains[domain])
            for word, count in collections.Counter(domain_words).most_common(10):
                most_frequent_words_domains[domain].append(word)

        ##### Look for most frequent words within an intent that are not the most frequent within its corresponding intent:
        for intent in sentences_intents:
            intent_words = extract_only_nava_words(sentences_intents[intent])
            domain = intent.split(":")[0]
            filtered_intent_words = list(
                filter(
                    lambda word: word not in most_frequent_words_domains[domain],
                    intent_words,
                )
            )
            # 100 per intent per language
            for word, count in collections.Counter(filtered_intent_words).most_common(
                20
            ):
                most_frequent_words_intents[intent].append(word)

    #### Store most frequent words per domain
    with open(PATH_MOST_FREQUENT_DOMAINS, "wb") as file:
        pickle.dump(most_frequent_words_domains, file)

    #### Store most frequent words per intent
    with open(PATH_MOST_FREQUENT_INTENTS, "wb") as file:
        pickle.dump(most_frequent_words_intents, file)

    return most_frequent_words_domains, most_frequent_words_intents


def print_most_frequent_words_intents_domains(
    most_frequent_words_domains, most_frequent_words_intents
):
    # Print most frequent words per domain
    print("Printing most frequent words per DOMAIN")
    for domain in most_frequent_words_domains:
        print(
            "domain:",
            domain,
            " most_frequent_words[domain]:",
            most_frequent_words_domains[domain],
        )

    # Print most frequent words per intent
    print("Printing most frequent words per INTENT")
    for intent in most_frequent_words_intents:
        print(
            "intent:",
            intent,
            " most_frequent_words[intent]:",
            most_frequent_words_intents[intent],
        )


def load_most_frequent_words_intents_domains():
    with open(PATH_MOST_FREQUENT_DOMAINS, "rb") as file:
        most_frequent_words_domains = pickle.load(file)

    with open(PATH_MOST_FREQUENT_INTENTS, "rb") as file:
        most_frequent_words_intents = pickle.load(file)

    return most_frequent_words_domains, most_frequent_words_intents


# (
#     most_frequent_words_domains,
#     most_frequent_words_intents,
# ) = load_most_frequent_words_intents_domains()


# Define a simple lexicon-based classifier to check the degree of easiness:
def lexicon_classifier(languages):
    pred_intents = {split_name: [] for split_name in SPLIT_NAMES}
    true_intents = {split_name: [] for split_name in SPLIT_NAMES}
    overall_acc = 0
    for split_name in SPLIT_NAMES:
        for lang in languages.split("_"):
            for i, eg in enumerate(dataset.streams[split_name][lang]):
                # Tokenize and extract only NAVAs
                sentence_navas = extract_only_nava_words_per_sentence(eg.text)
                scores = {intent: 0 for intent in most_frequent_words_intents}
                for nava_word in sentence_navas:
                    # Go over lexicon for each intent and get their score
                    for intent in most_frequent_words_intents:
                        for words in most_frequent_words_intents[intent]:
                            if nava_word in words:
                                scores[intent] += 1

                # Find most frequent words
                sortedDict = sorted(scores.items(), reverse=True, key=lambda x: x[1])
                pred_intent = next(iter(sortedDict))[0]
                true_intent = class_types[eg.label]
                pred_intents[split_name].append(pred_intent)
                true_intents[split_name].append(true_intent)

            acc = 0
            for true_intent, pred_intent in zip(
                true_intents[split_name], pred_intents[split_name]
            ):
                if true_intent == pred_intent:
                    acc += 1
            acc = acc / len(true_intents[split_name])
            overall_acc += acc
            print("Accuracy:", acc)

        print("overall_acc:", overall_acc // len(languages.split("_")))


(
    most_frequent_words_domains,
    most_frequent_words_intents,
) = find_most_frequent_words_intents_domains(LANG_ORDER)
# TODO do the same thing for slots


def get_statistics_queues(er_lt_data):
    for i in range(5):
        dist_langs = [item.split("_")[1] for item in er_lt_data[i]]
        print(
            "Deck #: ",
            i,
            " #Items :",
            len(er_lt_data[i]),
            " language distribution:",
            Counter(dist_langs),
        )


# Analyze the keywords of the different Leitner Queues and see if there are any correlations between their lexical easiness and the queue number


## 1. Read the different contents of Leitner Queues at the end of each epoch
def read_leitner_queue(k, ep):
    root_results_dir = (
        out_dir
        + "BertBaseMultilingualCased/MTOP/"
        + "Ltn/LtnModel-ltn/DemotePrevious/"
        + "LtnSampling-fifo/FreqSample-epoch/"
        + "FreqUpdate-epoch-everything/NumDecks-5/"
        + "ER_prop-0.0_TrainERMain/_ERStrategy-"
        + ER_STRATEGY
        + "//ER_MaxSize-10105/ERSched-er-only/Mode-cont-mono/"
        + LANG_ORDER
        + "/lt_scheduler-decks/er_lt_scheduler_0decks_trainon-"
        + str(k)
        + "_epoch-"
        + str(ep)
        + ".json"
    )

    with open(root_results_dir) as file:
        er_lt_data = json.load(file)

    return er_lt_data


def quantities_easy_hard(er_lt_data):
    quant_easy = [0] * 5
    quant_hard = [0] * 5
    for i in range(5):
        queue_ids = er_lt_data[i]
        if len(queue_ids) != 0:
            for eg_id in queue_ids:
                eg = dataset.get_item_by_id(eg_id)

                # Tokenize and extract only NAVAs
                sentence_navas = extract_only_nava_words_per_sentence(eg.text)
                scores = {intent: 0 for intent in most_frequent_words_intents}
                for nava_word in sentence_navas:
                    # Go over lexicon for each intent and get their score
                    for intent in most_frequent_words_intents:
                        for words in most_frequent_words_intents[intent]:
                            if nava_word in words:
                                scores[intent] += 1

                # Find most frequent words
                sortedDict = sorted(scores.items(), reverse=True, key=lambda x: x[1])
                pred_intent = next(iter(sortedDict))[0]
                true_intent = class_types[eg.label]

                if true_intent == pred_intent:
                    quant_easy[i] += 1
                else:
                    quant_hard[i] += 1

    return quant_easy, quant_hard


def score_lexical_easiness(er_lt_data):
    accs = {i: 0.0 for i in range(5)}
    for i in range(5):
        queue_ids = er_lt_data[i]
        acc = 0
        if len(queue_ids) != 0:
            for eg_id in queue_ids:
                eg = dataset.get_item_by_id(eg_id)

                # Tokenize and extract only NAVAs
                sentence_navas = extract_only_nava_words_per_sentence(eg.text)
                scores = {intent: 0 for intent in most_frequent_words_intents}
                for nava_word in sentence_navas:
                    # Go over lexicon for each intent and get their score
                    for intent in most_frequent_words_intents:
                        for words in most_frequent_words_intents[intent]:
                            if nava_word in words:
                                scores[intent] += 1

                # Find most frequent words
                sortedDict = sorted(scores.items(), reverse=True, key=lambda x: x[1])
                pred_intent = next(iter(sortedDict))[0]
                true_intent = class_types[eg.label]

                if true_intent == pred_intent:
                    acc += 1

            accs[i] = acc / len(queue_ids)
        else:
            accs[i] = acc

    return accs


# ## Print those most frequent words
# print_most_frequent_words_intents_domains(
#     most_frequent_words_domains, most_frequent_words_intents
# )
## 2. Scores of lexical easiness each sentence for each epoch and each queue
scores = {
    k: {  # language trained on
        ep: {deck_n: 0.0 for deck_n in range(5)}  # epoch  # deck number
        for ep in range(10)
    }
    for k in range(1, 4, 1)
}

lang_dist = {
    k: {
        ep: {deck_n: {} for deck_n in range(5)} for ep in range(10)
    }  # language trained on  # epoch
    for k in range(1, 4, 1)
}

num_items = {
    k: {  # language trained on
        ep: {deck_n: 0 for deck_n in range(5)} for ep in range(10)  # epoch
    }
    for k in range(1, 4, 1)
}


quantities_easy = {
    "Ep " + str(ep): {"Hop 1": [], "Hop 2": [], "Hop 3": []} for ep in range(10)
}
quantities_hard = {
    "Ep " + str(ep): {"Hop 1": [], "Hop 2": [], "Hop 3": []} for ep in range(10)
}

for k in range(1, 4, 1):
    for ep in range(10):
        er_lt_data = read_leitner_queue(k, ep)

        # Quantities of Easy/Hard
        (
            quantities_easy["Ep " + str(ep)]["Hop " + str(k)],
            quantities_hard["Ep " + str(ep)]["Hop " + str(k)],
        ) = quantities_easy_hard(er_lt_data)

        # Score that
        scores[k][ep] = score_lexical_easiness(er_lt_data)

        # Language distribution per queue
        for i in range(5):
            queue_langs = [item.split("_")[1] for item in er_lt_data[i]]
            lang_dist[k][ep][i] = dict(Counter(queue_langs))
            num_items[k][ep][i] = len(er_lt_data[i])

print("**** quantities_easy:", quantities_easy)
print("**** quantities_hard:", quantities_hard)

print("scores:", scores)

plot_save_dir = (
    "/home1/mmhamdi/x-continuous-learning_new/outputs/Plots/spacedrepetition/QueuesViz/"
    + ER_STRATEGY
    + "/"
)

print("lang_dist:", lang_dist)
print("num_items:", num_items)

if not os.path.isdir(plot_save_dir):
    os.makedirs(plot_save_dir)

with open(plot_save_dir + "scores.pickle", "wb") as file:
    pickle.dump(scores, file)


with open(plot_save_dir + "lang_dist.pickle", "wb") as file:
    pickle.dump(lang_dist, file)

with open(plot_save_dir + "num_items.pickle", "wb") as file:
    pickle.dump(num_items, file)

with open(plot_save_dir + "quantities_easy.pickle", "wb") as file:
    pickle.dump(quantities_easy, file)

with open(plot_save_dir + "quantities_hard.pickle", "wb") as file:
    pickle.dump(quantities_hard, file)

exit(0)

for k in range(1, 4, 1):
    ## 3. Bar Plot for each epoch after training on each language: x-axis: queue number, y-axis: percentage easiness of items
    for ep in range(10):
        df = pd.DataFrame(
            {
                "Queue Numbers": list(range(1, 5, 1)),
                "Scores after Training on "
                + k: [
                    scores[k][ep][0],
                    scores[k][ep][1],
                    scores[k][ep][2],
                    scores[k][ep][3],
                    scores[k][ep][4],
                ],
            }
        )

        sns.lineplot(data=df, x="Queue Numbers", y="Scores after Training on " + k)
        plt.show()
        plt.savefig(
            plot_save_dir
            + "scores_per_queue_train-"
            + str(k)
            + "_epoch-"
            + str(ep)
            + ".png"
        )

    ## Easiness Scores: Curve Plot for all epochs and queues: 5 curves
    queues_all = []
    easiness_all = []
    for i in range(5):
        queues_all.extend(["Queue " + str(i + 1)] * 10)
        easiness_all.extend([scores[k][ep][i] for ep in range(10)])

    df = pd.DataFrame(
        {
            "Epochs": list(range(1, 11, 1)) * 5,
            "Queues": queues_all,
            "Scores": easiness_all,
        }
    )

    sns.lineplot(data=df, x="Epochs", y="Scores", hue="Queues")
    plt.show()
    plt.savefig(plot_save_dir + "queues_easiness_after_train-" + str(k) + ".png")

    ## Number of items: Curve Plot for all epochs and queues: 5 curves
    queues_all = []
    quantities_all = []
    for i in range(5):
        queues_all.extend(["Queue " + str(i + 1)] * 10)
        quantities_all.extend([num_items[k][ep][i] for ep in range(10)])

    df = pd.DataFrame(
        {
            "Epochs": list(range(1, 11, 1)) * 5,
            "Queues": queues_all,
            "Quantity": quantities_all,
        }
    )

    sns.lineplot(data=df, x="Epochs", y="Quantities", hue="Queues")
    plt.show()
    plt.savefig(plot_save_dir + "queues_quantity_after_train-" + str(k) + ".png")

    ## Number of items per language: Curve Plot for all epochs and queues: 5 curves
    for lang in LANG_ORDER.split("_"):
        queues_all = []
        quantities_all = []
        for i in range(5):
            queues_all.extend(["Queue " + str(i + 1)] * 10)
            quantities_all.extend([lang_dist[k][ep][i][lang] for ep in range(10)])

        df = pd.DataFrame(
            {
                "Epochs": list(range(1, 11, 1)) * 5,
                "Queues": queues_all,
                "Quantity": quantities_all,
            }
        )

        sns.lineplot(data=df, x="Epochs", y="Quantities", hue="Queues")
        plt.show()
        plt.savefig(
            plot_save_dir
            + "queues_quantity_lang-"
            + lang
            + "_after_train-"
            + str(k)
            + ".png"
        )
