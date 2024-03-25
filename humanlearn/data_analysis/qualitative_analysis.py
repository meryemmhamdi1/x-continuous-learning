import json
import pickle
import os
from collections import Counter

import sys
import torch
import importlib
import configparser

sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
from src.data_utils import MultiPurposeDataset, AugmentedList
from src.transformers_config import MODELS_dict
from src.consts import DOMAIN_TYPES, INTENT_TYPES, SLOT_TYPES
from src.utils import transfer_batch_cuda
from scipy import spatial

if sys.platform == "darwin":
    sys.path.append(
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new"
    )
    location = "LOCAL"
else:
    sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
    location = "CARC"

paths = configparser.ConfigParser()
paths.read("scripts/paths.ini")


class DataArgs:
    def __init__(self):
        self.data_root = "/project/jonmay_231/meryem/Datasets/"
        self.model_root = os.path.join(str(paths.get(location, "MODEL_ROOT")))
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

        self.data_root = os.path.join(
            self.data_root,
            self.task_name.upper(),
            self.data_name.upper(),
        )

        self.use_crf = True
        self.seed = 42
        self.num_class_tasks = 10
        self.num_lang_tasks = 2
        self.multi_head_out = False
        self.max_mem_sz = 100
        self.er_strategy = "random"
        self.order_class = 0
        self.num_labels = len(INTENT_TYPES[self.data_name])
        self.num_slots = len(SLOT_TYPES[self.data_name])
        self.num_tasks = -1
        self.eff_num_classes_task = -1
        self.use_adapters = False

        self.use_k_means = False


args = DataArgs()
print(args.task_name)
print(args.setup_opt)

tokenizer = MODELS_dict[args.trans_model][1].from_pretrained(
    MODELS_dict[args.trans_model][0], do_lower_case=True, do_basic_tokenize=False
)

dataset = MultiPurposeDataset(args, tokenizer)
class_types = dataset.class_types

hop_n = 1
train_lang = "de"

intent_stats = {}
ids_per_intent = {}
for eg in dataset.train_set[train_lang]:
    intent = class_types[eg.label]
    if intent not in intent_stats:
        intent_stats.update({intent: 0})
        ids_per_intent.update({intent: []})
    intent_stats[intent] += 1
    ids_per_intent[intent].append(eg.unique_id)


intent_stats_sorted = dict(sorted(intent_stats.items(), key=lambda item: item[1]))
print("intent_stats_sorted: ", intent_stats_sorted)


# TODO do the counts for German, Hindi, and Thai
# Load the checkpoints of the model at the end of the first hop
model_path = (
    "/project/jonmay_231/meryem/ResultsSpacedRepetitionALL/x-continuous-learn/BertBaseMultilingualCased/MTOP/Ltn/LtnModel-ltn/DemotePrevious/LtnSampling-rand/FreqSample-epoch/FreqUpdate-epoch-everything/NumDecks-5/ER_prop-0.0_TrainERMain/_ERStrategy-easy/_WIPEStrategy-random/ER_MaxSize-10105/ERSched-er-only/Mode-cont-mono/en_de_hi_th/checkpoints/hop_"
    + str(hop_n)
    + "_model.bin"
)

model_checkpoint = torch.load(model_path)
DownstreamModel = importlib.import_module("src.basemodels.transTOD")

trans_model = "BertBaseMultilingualCased"
model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[trans_model]

if args.model_root:
    model_load_alias = os.path.join(args.model_root, model_name)
else:
    model_load_alias = model_name

config = config_alias.from_pretrained(
    model_load_alias,
    output_hidden_states=True,
    output_attentions=True,
)

tokenizer = tokenizer_alias.from_pretrained(
    model_load_alias, do_lower_case=True, do_basic_tokenize=False
)

model_trans = model_trans_alias.from_pretrained(model_load_alias, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DownstreamModel.TransModel(trans_model=model_trans, args=args, device=device)
model.to(device)

import numpy as np


def euclid(vec1, vec2):
    euclidean_dist = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return euclidean_dist


def sortkey(item):
    return item[1]


def knearest(vec, data, k):
    result = []
    for row in range(0, len(data)):
        distance = euclid(vec, data[row])
        result.append([row, distance])
    sortedResult = sorted(result, key=sortkey)
    indices = []
    if k < len(data):
        for r in range(0, k):
            indices.append(sortedResult[r][0])
    else:
        indices = [i[0] for i in sortedResult]
    return indices


centroids_per_intent = []
centroids_dict = {}
for intent in INTENT_TYPES[args.data_name]:
    corpus_embeddings = []
    for id_ in ids_per_intent[intent]:
        # Compute the embeddings for all training examples with that model
        batch, _, _ = dataset.get_batch_one(
            dataset=dataset, identifier=id_, identifiers=None
        )

        batch = transfer_batch_cuda(batch, device)

        batch["train_idx"] = 0

        model.eval()
        with torch.no_grad():
            output = model(**batch)

        embeddings = output.pool_out[0].cpu()
        corpus_embeddings.append(embeddings)

    centroid = torch.mean(torch.stack(corpus_embeddings), axis=0).numpy()
    print("centroid:", centroid.shape)
    # Compute the centroid of each intent
    centroids_per_intent.append(centroid)
    centroids_dict.update({intent: centroid})

with open("centroids_dict.pickle", "wb") as file:
    pickle.dump(centroids_dict, file)

with open("centroids_per_intent.pickle", "wb") as file:
    pickle.dump(centroids_per_intent, file)

exit(0)

# Find closest vectors
for int_idx in range(len(INTENT_TYPES[args.data_name])):
    k = 5
    knn = knearest(
        centroids_dict[INTENT_TYPES[args.data_name][int_idx]],
        np.stack(centroids_per_intent),
        k,
    )

    print(
        " Nearest neighbors of ",
        INTENT_TYPES[args.data_name][int_idx],
    )
    for idx in knn:
        print("intent: ", INTENT_TYPES[args.data_name][idx])

    print("*************************************")
# print(knn)


# k = 5
# knn = knearest(centroids_dict["news:GET_DETAILS_NEWS"], centroids_per_intent, k)
# print("Row IDs of ", k, " nearest neighbors:")
# print(knn)
# Check if the intents in questions are indeed closer by distance or something
