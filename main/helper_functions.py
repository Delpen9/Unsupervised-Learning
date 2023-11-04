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

from clustering.k_means import (
    KMeans,
)

from clustering.expected_maximization import (
    get_gmm_bic_aic_accuracy_f1,
)

from dimensionality_reduction.principal_component_analysis import (
    PCADimensionalityReduction,
)

from dimensionality_reduction.independent_component_analysis import (
    ICADimensionalityReduction,
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


def get_pca_explained_variance(
    auction_train_X: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    output_filepath: str = "../output/dimensionality_reduction/",
) -> None:
    def plot_linechart(
        df: pd.DataFrame,
        output_filename: str = "output_chart.png",
        dataset_type: str = "auction",
        color: str = "red",
    ) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            df["N-Components"],
            df["Explained Variance"],
            color=color,
            linewidth=2,
            label="Explained Variance",
        )

        plt.title(
            f"{dataset_type.title()}: Explained Variance per N-Components", fontsize=16
        )
        plt.xlabel("N-Components", fontsize=14)
        plt.ylabel("Explained Variance", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        plt.savefig(output_filename)
        plt.close()

    ##########################
    # PCA - Auction
    ##########################
    n_components_list = np.arange(1, 7).astype(int).tolist()

    auction_explained_variance_list = []
    for n_components in n_components_list:
        _pca = PCADimensionalityReduction(n_components=n_components)
        _pca.fit(data=auction_train_X)
        _pca.transform(data=auction_train_X)
        explained_variance = _pca.get_explained_variance()
        auction_explained_variance_list.append([n_components, explained_variance])

    auction_explained_variance_df = pd.DataFrame(
        auction_explained_variance_list,
        columns=["N-Components", "Explained Variance"],
    )

    dataset_type = "auction"
    output_filename = (
        rf"{output_filepath}{dataset_type}_pca_explained_variance_per_n_components.png"
    )
    plot_linechart(
        df=auction_explained_variance_df,
        output_filename=output_filename,
        dataset_type=dataset_type,
        color="red",
    )
    ##########################
    # PCA - Dropout
    ##########################
    n_components_list = np.arange(1, 31).astype(int).tolist()

    dropout_explained_variance_list = []
    for n_components in n_components_list:
        _pca = PCADimensionalityReduction(n_components=n_components)
        _pca.fit(data=dropout_train_X)
        _pca.transform(data=dropout_train_X)
        explained_variance = _pca.get_explained_variance()
        dropout_explained_variance_list.append([n_components, explained_variance])

    dropout_explained_variance_df = pd.DataFrame(
        dropout_explained_variance_list,
        columns=["N-Components", "Explained Variance"],
    )

    dataset_type = "dropout"
    output_filename = (
        rf"{output_filepath}{dataset_type}_pca_explained_variance_per_n_components.png"
    )
    plot_linechart(
        df=dropout_explained_variance_df,
        output_filename=output_filename,
        dataset_type=dataset_type,
        color="blue",
    )


def get_pca_transformed_output(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    pca_reduction = PCADimensionalityReduction(n_components=2)
    pca_reduction.fit(train_X)
    transformed_train_X = pca_reduction.transform(train_X)

    data = pd.concat(
        [pd.DataFrame(transformed_train_X), train_y.reset_index(drop=True)], axis=1
    )
    data.columns = ["Principal Component 1", "Principal Component 2", "Label"]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x="Principal Component 1",
        y="Principal Component 2",
        hue="Label",
        palette="viridis",
        s=100,
        alpha=0.7,
    )

    plt.grid(True)

    plt.title(f"{dataset_type.capitalize()}: PCA - 2 Principal Components", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title="Label", fontsize="large", title_fontsize="13", loc="best")

    plt.savefig(
        rf"{output_filepath}{dataset_type}_pca_2_principal_component_scatter_plot.png"
    )
    plt.close()


def get_optimal_ica_components(
    data: pd.DataFrame,
    max_components: int = 10,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    avg_kurtosis_values = []
    for n in range(1, max_components + 1):
        ica_reduction = ICADimensionalityReduction(n_components=n, random_state=42)
        kurtosis_vals = ica_reduction.calculate_kurtosis(data)
        avg_kurtosis = abs(kurtosis_vals).mean()
        avg_kurtosis_values.append(avg_kurtosis)

    optimal_n_components = avg_kurtosis_values.index(max(avg_kurtosis_values)) + 1

    plt.plot(range(1, max_components + 1), avg_kurtosis_values, marker="o")
    plt.xlabel("Number of ICA Components")
    plt.ylabel("Average Absolute Kurtosis")
    plt.title("Average Absolute Kurtosis vs. Number of ICA Components")
    plt.axvline(x=optimal_n_components, color="r", linestyle="--")

    plt.savefig(
        rf"{output_filepath}{dataset_type}_ica_average_absolute_kurtosis_per_n_components.png"
    )
    plt.close()


def get_ica_transformed_output(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    ica_reduction = ICADimensionalityReduction(n_components=2)
    ica_reduction.fit(train_X)
    transformed_train_X = ica_reduction.transform(train_X)

    data = pd.concat(
        [pd.DataFrame(transformed_train_X), train_y.reset_index(drop=True)], axis=1
    )
    data.columns = ["Independent Component 1", "Independent Component 2", "Label"]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x="Independent Component 1",
        y="Independent Component 2",
        hue="Label",
        palette="viridis",
        s=100,
        alpha=0.7,
    )

    plt.grid(True)

    plt.title(
        f"{dataset_type.capitalize()}: ICA - 2 Independent Components", fontsize=16
    )
    plt.xlabel("Independent Component 1", fontsize=14)
    plt.ylabel("Independent Component 2", fontsize=14)
    plt.legend(title="Label", fontsize="large", title_fontsize="13", loc="best")

    plt.savefig(
        rf"{output_filepath}{dataset_type}_ica_2_independent_component_scatter_plot.png"
    )
    plt.close()
