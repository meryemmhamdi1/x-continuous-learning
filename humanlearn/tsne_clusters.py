from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import importlib
import os
from sys import platform
import sys
import configparser
import torch
from tqdm import tqdm
import numpy as np
import pickle
import random
from kneed import KneeLocator
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

if platform == "darwin":
    sys.path.append(
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new"
    )
    location = "LOCAL"
else:
    sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
    location = "CARC"

from src.transformers_config import MODELS_dict
from src.data_utils import MultiPurposeDataset, AugmentedList
from src.consts import INTENT_TYPES, SLOT_TYPES
from src.utils import transfer_batch_cuda

DownstreamModel = importlib.import_module("src.basemodels.transTOD")

paths = configparser.ConfigParser()
paths.read("scripts/paths.ini")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Corpus with example sentences
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
        self.model_root = os.path.join(str(paths.get(location, "MODEL_ROOT")))
        self.setup_opt = "cll-er_kd"
        self.order_lst = "en_de_hi_th"
        self.languages = ["en", "de", "hi", "th"]
        self.order_class = 0
        self.order_lang = 0
        self.trans_model = "BertBaseMultilingualCased"
        self.use_slots = True
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


args = DataArgs()

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

model = DownstreamModel.TransModel(trans_model=model_trans, args=args, device=device)
model.to(device)


tokenizer = MODELS_dict[args.trans_model][1].from_pretrained(
    MODELS_dict[args.trans_model][0], do_lower_case=True, do_basic_tokenize=False
)

dataset = MultiPurposeDataset(args, tokenizer)


## Load the clusters and centroids
clusters_save_dir = "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new/outputs/k-means/"

# cluster_labels = []
# embeddings_all = []
for lang in ["en", "de", "hi", "th"]:
    ## Loading Centroids
    with open(
        os.path.join(clusters_save_dir, "centroids_" + lang + ".pickle"),
        "rb",
    ) as file:
        centroids = pickle.load(file)

    ## Loading Clusters IDs and their items
    with open(
        os.path.join(clusters_save_dir, "clusters_ids_" + lang + ".pickle"),
        "rb",
    ) as file:
        clustered_sentences = pickle.load(file)

    ## Loading Clusters embeddings
    with open(
        os.path.join(clusters_save_dir, "clusters_embeds_" + lang + ".pickle"),
        "rb",
    ) as file:
        clustered_embeddings = pickle.load(file)

    cluster_labels = []
    embeddings_all = []
    for k in range(len(clustered_embeddings)):
        for item_id in clustered_embeddings[k]:
            cluster_labels.append(k)
            embeddings_all.append(item_id)

    ## TSNE of memory items or centroids and memory items to see how separable
    embeddings_np = np.array(embeddings_all)
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))
    uniq = np.unique(cluster_labels)
    for i in range(10):
        idx_i = [j for j in range(len(cluster_labels)) if cluster_labels[j] == i]
        plt.scatter(
            X_embedded[idx_i, 0],
            X_embedded[idx_i, 1],
            label=i,
        )
        # plt.scatter(centers[:, 0], centers[:, 1], marker="x", color="k")
        # This is done to find the centroid for each clusters.
    plt.legend()
    print("Saving the figure for language:", lang)
    plt.savefig(os.path.join(clusters_save_dir, "clusters_tsne_" + lang + ".png"))
    # plt.show()
