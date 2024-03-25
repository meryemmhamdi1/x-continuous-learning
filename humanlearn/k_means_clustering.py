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


def elbow_method(all_embeddings):
    sse = []
    silhouettes = []
    for k in range(5, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(all_embeddings)
        sse.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(all_embeddings, kmeans.labels_))

    kl = KneeLocator(range(5, 20), sse, curve="convex", direction="decreasing")
    best = kl.elbow
    print("Elbow approach best number of clusters: ", best)
    print(
        "Silhouette approach best number of clusters: ",
        list(range(5, 20))[silhouettes.index(max(silhouettes))],
        " silhouettes: ",
        silhouettes,
        " max:",
        max(silhouettes),
    )
    return best


clusters_save_dir = "/home1/mmhamdi/x-continuous-learning_new/outputs/k-means/"
for i in range(4):
    print("-- Language ", args.languages[i])
    iterator = dataset.train_stream[i]

    corpus_embeddings = []
    all_examples_ids = []
    for step_num in tqdm(range(iterator["size"])):
        batch, examples, features = dataset.next_batch(
            dataset=dataset,
            batch_size=1,
            data_split=iterator["examples"],
        )

        all_examples_ids.append(examples[0].unique_id)

        batch = transfer_batch_cuda(batch, device)

        batch["train_idx"] = 0

        model.eval()
        with torch.no_grad():
            output = model(**batch)

        corpus_embeddings.append(output.pool_out[0].cpu())

    all_embeddings = torch.stack(corpus_embeddings).detach().numpy()

    # Perform k-means clustering
    num_clusters = elbow_method(all_embeddings)
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for _ in range(num_clusters)]
    clustered_embeddings = [[] for _ in range(num_clusters)]

    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(all_examples_ids[sentence_id])
        clustered_embeddings[cluster_id].append(all_embeddings[sentence_id])

    # Saving clusters with sentence ids
    for j, cluster in enumerate(clustered_sentences):
        print("Cluster ", j + 1, len(cluster))

    with open(
        os.path.join(
            clusters_save_dir, "clusters_ids_" + args.languages[i] + ".pickle"
        ),
        "wb",
    ) as file:
        pickle.dump(clustered_sentences, file)

    with open(
        os.path.join(
            clusters_save_dir, "clusters_embeds_" + args.languages[i] + ".pickle"
        ),
        "wb",
    ) as file:
        pickle.dump(clustered_embeddings, file)

    # Saving centroids
    centroids = clustering_model.cluster_centers_

    with open(
        os.path.join(clusters_save_dir, "centroids_" + args.languages[i] + ".pickle"),
        "wb",
    ) as file:
        pickle.dump(centroids, file)

    ## Pick points close to the centroids for each language by picking num_closest_rep elements from each cluster
    num_closest_rep = 10105 // (len(args.languages) - 1) // num_clusters
    mem_rep = []
    for j, centroid in enumerate(centroids):
        euc_res = euclidean_distances(
            np.array([centroid]), np.array(clustered_embeddings[j])
        )

        sorted_res = list(np.argsort(euc_res[0])[:num_closest_rep])

        mem_rep.extend([clustered_sentences[j][k] for k in sorted_res])

    with open(
        os.path.join(
            clusters_save_dir, "memory_language_" + args.languages[i] + ".pickle"
        ),
        "wb",
    ) as file:
        pickle.dump(mem_rep, file)
