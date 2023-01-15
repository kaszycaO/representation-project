import numpy as np
from sklearn.metrics import silhouette_score, cluster, davies_bouldin_score


def silhouette(data: np.array, clusters: np.array):
    return silhouette_score(data, clusters)


def davies_bouldin(data: np.array, clusters: np.array):
    return davies_bouldin_score(data, clusters)