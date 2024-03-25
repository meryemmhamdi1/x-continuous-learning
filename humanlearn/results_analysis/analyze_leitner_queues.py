import json
import pickle
import os
from collections import Counter

import sys
from nltk.stem import WordNetLemmatizer

sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
from src.data_utils import MultiPurposeDataset, AugmentedList
from src.transformers_config import MODELS_dict
from src.consts import DOMAIN_TYPES, INTENT_TYPES
import fasttext
import nltk


class DataArgs:
    def __init__(self):
        self.data_root = "/project/jonmay_231/meryem/Datasets/"
        self.task_name = "tod"
        self.data_name = "mtop"
        self.data_format = "txt"
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

        self.data_root = os.path.join(
            self.data_root,
            self.task_name.upper(),
            self.data_name.upper(),
        )


args = DataArgs()
print(args.task_name)
print(args.setup_opt)

tokenizer = MODELS_dict[args.trans_model][1].from_pretrained(
    MODELS_dict[args.trans_model][0], do_lower_case=True, do_basic_tokenize=False
)

dataset = MultiPurposeDataset(args, tokenizer)

er_strategy = "easy"
LANG_ORDER = "en_de_hi_th"
for k in range(4):
    print("Training on " + LANG_ORDER.split("_")[k])
    for ep in range(10):
        print("----Epoch: ", ep)
        root_results_dir = (
            "/project/jonmay_231/meryem/ResultsSpacedRepetitionNew/x-continuous-learn/BertBaseMultilingualCased/MTOP/"
            + "/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-fifo/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-"
            + er_strategy
            + "/ERSched-er-only"
            + "/Mode-cont-mono/"
            + LANG_ORDER
            + "/lt_scheduler-decks/"
            + "er_lt_scheduler_0decks_trainon-"
            + str(k)
            + "_epoch-"
            + str(ep)
            + ".json"
        )

        with open(
            "/home1/mmhamdi/x-continuous-learning_new/outputs/most_frequent_words_intents.pickle",
            "rb",
        ) as file:
            most_frequent_words_intents = pickle.load(file)

        with open(root_results_dir) as file:
            er_lt_data = json.load(file)

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

NAVA_TAGS = [
    # Nouns
    "NN",
    "NNP",
    "NNS",
    "NNPS",
    # Adjectives
    "JJ",
    "JJR",
    "JJS",
    # Adverbs
    "RB",
    "RBR",
    "RBS",
    # Verbs
    "VB",
    "VBG",
    "VBD",
    "VBN",
    "VBP" "VBZ",
]

lemmatizer = WordNetLemmatizer()


def extract_only_nava_words(sentences):
    """
    Expected sentences from a given domain or intent
    """
    navas = []

    for sentence in sentences:
        for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
            if pos in NAVA_TAGS:
                navas.append(lemmatizer.lemmatize(word.lower()))
    return navas


def extract_only_nava_words_per_sentence(sentence):
    """
    Expected one sentence
    """
    navas = []
    for word, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
        if pos in NAVA_TAGS:
            navas.append(lemmatizer.lemmatize(word.lower()))
    return navas


class_types = dataset.class_types
for j in range(5):
    for i, eg_id in enumerate(er_lt_data[j]):
        # Tokenize and extract only NAVAs
        eg = dataset.get_item_by_id(eg_id)
        sentence_navas = extract_only_nava_words_per_sentence(eg.text)
        scores = {intent: 0 for intent in most_frequent_words_intents}
        for nava_word in sentence_navas:
            # Go over lexicon for each intent and get their score
            for intent in most_frequent_words_intents:
                for words in most_frequent_words_intents[intent]:
                    if nava_word in words:
                        scores[intent] += 1
        sortedDict = sorted(scores.items(), reverse=True, key=lambda x: x[1])
        pred_intent = next(iter(sortedDict))[0]
        true_intent = class_types[eg.label]
