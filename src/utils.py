import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from umap import UMAP
from typing import Tuple
from sklearn.cluster import KMeans

import pandas as pd

from .clustering_metrics import silhouette, davies_bouldin


def evaluate_embeddings(z: torch.Tensor, y: torch.Tensor, random_state: int = 345) -> Tuple[plt.figure, dict]:
    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

    z2d_pca = PCA(n_components=2).fit_transform(z)
    z2d_umap = UMAP(n_components=2).fit_transform(z)
    
    kmeans = KMeans(
        init="random",
        n_clusters=len(y.unique()),
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    kmeans.fit_transform(z)
    
    data = {
        "silhoute": silhouette(z, kmeans.labels_),
        "davies-bouldin": davies_bouldin(z, kmeans.labels_)
    }

    sns.scatterplot(
        x=z2d_pca[:, 0],
        y=z2d_pca[:, 1],
        hue=y,
        palette="Set2",
        ax=axs[0],
    )
    axs[0].set(title="PCA")
    axs[0].get_legend().remove()

    sns.scatterplot(
        x=z2d_umap[:, 0],
        y=z2d_umap[:, 1],
        hue=y,
        palette="Set2",
        ax=axs[1],
    )
    axs[1].set(title="UMAP")
    axs[1].get_legend().remove()
    

    return fig, data

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for model_name in list(df["model_name"].unique()):

        mean_df = pd.DataFrame(df[df.model_name == model_name].mean(), columns=["0"])
        std_df = pd.DataFrame(df[df.model_name == model_name].std(), columns=["1"])
        stats_df = pd.concat([mean_df, std_df], axis=1)
        
        stats_df[model_name] = stats_df.apply(lambda row: f"{round(row['0'], 3)} +- {round(row['1'], 3)}", axis=1)
        stats_df = stats_df.drop(["0", "1"], axis=1).T
        dfs.append(stats_df)
    
    return pd.concat(dfs)
