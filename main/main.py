import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

# Data Science Libraries
import numpy as np
import pandas as pd

# Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# SKLearn Libraries
from sklearn.mixture import GaussianMixture

# Miscellaneous
from typing import Union

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)

from clustering.k_means import (
    KMeans,
)

from clustering.expected_maximization import (
    get_gmm_bic_aic_accuracy_f1,
)


def fit_k_means(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    n_clusters: int,
    num_iterations: int = 10,
    metric: str = "euclidean",
    get_f1: bool = False,
) -> Union[tuple[float, list[int]], tuple[float, list[int], float, float]]:
    k_means = KMeans(
        X=train_X,
        true_labels=train_y.values.astype(int).flatten(),
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
    train_X: pd.DataFrame,
    train_y: pd.Series,
    max_clusters: int = 30,
    num_iterations: int = 10,
    metric: str = "euclidean",
    dataset_type: str = "auction",
) -> None:
    elbow_data = []
    for n_clusters in range(2, max_clusters + 1):
        inertia, _ = fit_k_means(train_X, train_y, n_clusters, num_iterations, metric)
        elbow_data.append([n_clusters, inertia])
    elbow_df = pd.DataFrame(elbow_data, columns=["Clusters", "Inertia"])

    save_elbow_graph(df=elbow_df, filename=rf"{dataset_type}_k_means_elbow_graph.png")


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
    train_X: pd.DataFrame,
    train_y: pd.Series,
    num_iterations: int = 10,
    dataset_type: str = "auction",
) -> None:
    n_clusters = int(train_y.nunique())

    all_distance_metrics = ["euclidean", "manhattan", "cosine", "chebyshev"]

    all_accuracies = []
    all_f1_scores = []
    for distance_metric in all_distance_metrics:
        _, _, accuracy, f1 = fit_k_means(
            train_X=train_X,
            train_y=train_y,
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
        df=per_metric_performance,
        filename=rf"{dataset_type}_k_means_distance_metric_vs_accuracy_f1.png",
    )


def get_expected_maximization_performance_line_charts(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    n_components_array: np.ndarray = np.arange(2, 9, 1).astype(int),
    output_filepath: str = "../output/clustering/",
    dataset_type: str = "auction",
) -> None:
    def plot_linechart(
        df: pd.DataFrame,
        labels: list[str] = ["AIC", "BIC"],
        output_filename: str = "",
    ) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[labels[0]], color="red", label=labels[0], linewidth=2)
        plt.plot(df.index, df[labels[1]], color="blue", label=labels[1], linewidth=2)

        plt.title(rf"{labels[0]} and {labels[1]} per N Clusters", fontsize=16)
        plt.xlabel("N-Clusters", fontsize=14)
        plt.ylabel("Metric Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        plt.savefig(output_filename)
        plt.close()

    aic_bic_list = []
    for n_components in n_components_array:
        metrics = get_gmm_bic_aic_accuracy_f1(
            train_X=train_X, train_y=train_y, n_components=n_components
        )
        _aic = metrics[1]
        _bic = metrics[2]
        aic_bic_list.append([_aic, _bic])

    aic_bic_labels = ["AIC", "BIC"]
    aic_bic_df = pd.DataFrame(aic_bic_list, columns=aic_bic_labels)

    output_filename = (
        rf"{output_filepath}{dataset_type}_aic_bic_performance_per_n_clusters.png"
    )
    plot_linechart(
        df=aic_bic_df, labels=aic_bic_labels, output_filename=output_filename
    )


def part_1_1(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    max_clusters = 30
    metric = "euclidean"

    #######################################
    # K-Means: Auction
    #######################################
    num_iterations = 10
    get_k_means_elbow_graph(
        train_X=auction_train_X,
        train_y=auction_train_y,
        max_clusters=max_clusters,
        num_iterations=num_iterations,
        metric=metric,
        dataset_type="auction",
    )

    num_iterations = 50
    get_k_means_metric_vs_f1_score(
        train_X=auction_train_X,
        train_y=auction_train_y,
        num_iterations=num_iterations,
        dataset_type="auction",
    )

    #######################################
    # K-Means: Dropout
    #######################################
    num_iterations = 10
    get_k_means_elbow_graph(
        train_X=dropout_train_X,
        train_y=dropout_train_y,
        max_clusters=max_clusters,
        num_iterations=num_iterations,
        metric=metric,
        dataset_type="dropout",
    )

    num_iterations = 50
    get_k_means_metric_vs_f1_score(
        train_X=dropout_train_X,
        train_y=dropout_train_y,
        num_iterations=num_iterations,
        dataset_type="dropout",
    )


def part_1_2(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    #######################################
    # Expected Maximization: Auction
    #######################################
    get_expected_maximization_performance_line_charts(
        train_X=auction_train_X,
        train_y=auction_train_y,
        dataset_type="auction",
    )

    #######################################
    # Expected Maximization: Dropout
    #######################################
    get_expected_maximization_performance_line_charts(
        train_X=dropout_train_X,
        train_y=dropout_train_y,
        dataset_type="dropout",
    )


if __name__ == "__main__":
    RUN_PART_1 = True
    # RUN_PART_2 = True

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

    if RUN_PART_1:
        part_1_1(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
        part_1_2(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
