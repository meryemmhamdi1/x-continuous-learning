import numpy as np
from sklearn.manifold import TSNE
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open(
    "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/centroids_per_intent.pickle",
    "rb",
) as file:
    centroids = pickle.load(file)

with open(
    "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/centroids_dict.pickle",
    "rb",
) as file:
    centroids_dict = pickle.load(file)

print("centroids.shape:", centroids[0])

intents = list(centroids_dict.keys())
domains = [k.split(":")[0] for k in list(centroids_dict.keys())]

focus = [
    "people:GET_MAJOR",
    "people:GET_UNDERGRAD",
    "music:LOOP_MUSIC",
    "music:REPLAY_MUSIC",
    "news:GET_DETAILS_NEWS",
    # "news:QUESTION_NEWS",
    "news:GET_STORIES_NEWS",
    "calling:SET_UNAVAILABLE",
    "calling:SET_AVAILABLE",
    "recipes:GET_INFO_RECIPES",
    "recipes:GET_RECIPES",
    "weather:GET_SUNRISE",
    "weather:GET_SUNSET",
]
# centroids_only = [v for k, v in centroids_dict.items() if k in focus]

X = np.array(centroids)


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
    return indices, sortedResult


# dists = []
# labels_i = []
# for int_idx in range(1):  # range(len(intents)):
#     k = 5
#     knn, results = knearest(
#         centroids_dict[intents[int_idx]],
#         np.stack(centroids),
#         k,
#     )

#     print(" Nearest neighbors of ", intents[int_idx], " results: ", results)
#     for idx in knn:
#         print("intent: ", idx, intents[idx])
#         dists.append(results[idx])
#         labels_i.append(intents[idx])

#     print("*************************************")

# exit(0)

X_embedded = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=3
).fit_transform(X)

print(X_embedded)


df_subset = pd.DataFrame()
df_subset["tsne1"] = X_embedded[:, 0]
df_subset["tsne2"] = X_embedded[:, 1]


print("x: ")
for el in list(X_embedded[:, 0]):
    print(el)

print("y: ")
for el in list(X_embedded[:, 1]):
    print(el)

print("intents:")
for intent in intents:
    print(intent.split(":")[1])

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne1",
    y="tsne2",
    hue=domains,
    palette="Set1",
    data=df_subset,
    legend="full",
    s=120,
)
x = X_embedded[:, 0]
y = X_embedded[:, 1]
for i in range(len(x)):
    if intents[i] in focus:
        plt.annotate(intents[i].split(":")[1], (x[i], y[i] + 0.2), fontsize=7)


plt.show()
