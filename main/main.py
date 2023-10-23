import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

# Data Science Libraries
import numpy as np
import pandas as pd

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# SKLearn Libraries
from sklearn.mixture import GaussianMixture

# Miscellaneous
from typing import Union

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)

from k_means import (
    KMeans,
)


def fit_k_means(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.Series,
    n_clusters: int,
    num_iterations: int = 10,
    metric: str = "euclidean",
    get_f1: bool = False,
) -> Union[tuple[float, list[int]], tuple[float, list[int], float, float]]:
    k_means = KMeans(
        X=auction_train_X,
        true_labels=auction_train_y.values.astype(int).flatten(),
        n_clusters=n_clusters,
        metric=metric,
    )
    k_means.get_k_means(num_iterations)
    inertia = k_means.compute_inertia()
    cluster_sizes = k_means.get_cluster_sizes()
    if get_f1 == True:
        (accuracy, f1) = k_means.get_accuracy_and_f1_score()
        return (inertia, cluster_sizes, accuracy, f1)
    return (inertia, cluster_sizes)


def save_elbow_graph(
    df: pd.DataFrame,
    output_filepath: str = "../output/clustering/",
    filename: str = "temp.png",
) -> None:
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x="Clusters", y="Inertia", marker="o", linewidth=2.5, color="royalblue"
    )

    plt.title("Inertia vs. Number of Clusters", fontsize=16)
    plt.xlabel("Number of Clusters", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(rf"{output_filepath}{filename}")
    plt.close()


def get_k_means_elbow_graph(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.Series,
    max_clusters: int = 30,
    num_iterations: int = 10,
    metric: str = "euclidean",
) -> None:
    elbow_data = []
    for n_clusters in range(2, max_clusters + 1):
        inertia, _ = fit_k_means(
            auction_train_X, auction_train_y, n_clusters, num_iterations, metric
        )
        elbow_data.append([n_clusters, inertia])
    elbow_df = pd.DataFrame(elbow_data, columns=["Clusters", "Inertia"])

    save_elbow_graph(df=elbow_df, filename=rf"k_means_elbow_graph.png")


def get_distance_metric_bar_plot(
    df: pd.DataFrame,
    output_filepath: str = "../output/clustering/",
    filename: str = "temp.png",
) -> None:
    df_melted = df.melt(
        id_vars=["Distance Metric"],
        value_vars=["Accuracy", "F1 Score"],
        var_name="Metric",
        value_name="Value",
    )

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df_melted,
        x="Distance Metric",
        y="Value",
        hue="Metric",
        palette=["blue", "red"],
    )

    plt.title("Accuracy and F1 Score by Distance Metric", fontsize=16)
    plt.xlabel("Distance Metric", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Metric")
    plt.tight_layout()

    plt.savefig(rf"{output_filepath}{filename}")
    plt.close()


def get_k_means_metric_vs_f1_score(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.Series,
    num_iterations: int = 10,
) -> None:
    n_clusters = int(auction_train_y.nunique())

    all_distance_metrics = ["euclidean", "manhattan", "cosine", "chebyshev"]

    all_accuracies = []
    all_f1_scores = []
    for distance_metric in all_distance_metrics:
        _, _, accuracy, f1 = fit_k_means(
            auction_train_X=auction_train_X,
            auction_train_y=auction_train_y,
            n_clusters=n_clusters,
            num_iterations=num_iterations,
            metric=distance_metric,
            get_f1=True,
        )
        all_accuracies.append(accuracy)
        all_f1_scores.append(f1)

    all_distance_metrics_np = np.array([all_distance_metrics]).T
    all_accuracies_np = np.array([all_accuracies]).T
    all_f1_scores_np = np.array([all_f1_scores]).T

    per_metric_performance = pd.DataFrame(
        np.hstack((all_distance_metrics_np, all_accuracies_np, all_f1_scores_np)),
        columns=["Distance Metric", "Accuracy", "F1 Score"],
    ).astype({"Distance Metric": str, "Accuracy": float, "F1 Score": float})

    get_distance_metric_bar_plot(
        df=per_metric_performance, filename="k_means_distance_metric_vs_accuracy_f1.png"
    )


if __name__ == "__main__":
    (
        # Auction
        auction_train_X,
        auction_train_y,
        auction_val_X,
        auction_val_y,
        auction_test_X,
        auction_test_y,
        # Dropout
        dropout_train_X,
        dropout_train_y,
        dropout_val_X,
        dropout_val_y,
        dropout_test_X,
        dropout_test_y,
    ) = preprocess_datasets()

    auction_train_y = auction_train_y.iloc[:, 0]
    auction_val_y = auction_val_y.iloc[:, 0]
    auction_test_y = auction_test_y.iloc[:, 0]

    max_clusters = 30
    num_iterations = 10
    metric = "euclidean"

    get_k_means_elbow_graph(
        auction_train_X=auction_train_X,
        auction_train_y=auction_train_y,
        max_clusters=max_clusters,
        num_iterations=num_iterations,
        metric=metric,
    )

    num_iterations = 50
    get_k_means_metric_vs_f1_score(
        auction_train_X=auction_train_X,
        auction_train_y=auction_train_y,
        num_iterations=num_iterations,
    )
