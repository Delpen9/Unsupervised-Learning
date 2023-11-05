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

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Model Evaluation
from sklearn.metrics import roc_auc_score, accuracy_score

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

from dimensionality_reduction.randomized_projection import (
    RandomProjectionDimensionalityReduction,
)

from dimensionality_reduction.t_distributed_stochastic_neighbor_embedding import (
    TSNEReduction,
)

from models.NeuralNetwork import (
    train_neural_network,
    evaluate_model,
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
    plt.savefig(filename)
    plt.close()


def get_k_means_elbow_graph(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    max_clusters: int = 30,
    num_iterations: int = 10,
    metric: str = "euclidean",
    output_filepath: str = "../output/clustering/",
    dataset_type: str = "auction",
) -> None:
    elbow_data = []
    for n_clusters in range(2, max_clusters + 1):
        inertia, _ = fit_k_means(train_X, train_y, n_clusters, num_iterations, metric)
        elbow_data.append([n_clusters, inertia])
    elbow_df = pd.DataFrame(elbow_data, columns=["Clusters", "Inertia"])

    save_elbow_graph(
        df=elbow_df,
        filename=rf"{output_filepath}{dataset_type}_k_means_elbow_graph.png",
    )


def get_distance_metric_bar_plot(
    df: pd.DataFrame,
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

    plt.savefig(filename)
    plt.close()


def get_k_means_metric_vs_f1_score(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    num_iterations: int = 10,
    output_filepath: str = "../output/clustering/",
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
        filename=rf"{output_filepath}{dataset_type}_k_means_distance_metric_vs_accuracy_f1.png",
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


def get_optimal_randomized_projection_components(
    data: pd.DataFrame,
    max_components: int = 10,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    errors = []
    components_range = range(1, max_components + 1)
    for n_components in components_range:
        rp_reduction = RandomProjectionDimensionalityReduction(
            n_components=n_components, random_state=42
        )
        error = rp_reduction.reconstruction_error(data)
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.plot(components_range, errors, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Reconstruction Error")
    plt.title(
        "Reconstruction Error as a Function of the Number of Randomized Projection Components"
    )
    plt.grid(True)
    plt.savefig(
        rf"{output_filepath}{dataset_type}_randomized_projection_reconstruction_error_per_n_components.png"
    )
    plt.close()


def get_randomized_projection_transformed_output(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    rp_reduction = RandomProjectionDimensionalityReduction(n_components=2)
    rp_reduction.fit(train_X)
    transformed_train_X = rp_reduction.transform(train_X)

    data = pd.concat(
        [pd.DataFrame(transformed_train_X), train_y.reset_index(drop=True)], axis=1
    )
    data.columns = [
        "Randomized Projection Component 1",
        "Randomized Projection Component 2",
        "Label",
    ]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x="Randomized Projection Component 1",
        y="Randomized Projection Component 2",
        hue="Label",
        palette="viridis",
        s=100,
        alpha=0.7,
    )

    plt.grid(True)

    plt.title(
        f"{dataset_type.capitalize()}: Randomized Projection - 2 Randomized Projection Components",
        fontsize=16,
    )
    plt.xlabel("Randomized Projection Component 1", fontsize=14)
    plt.ylabel("Randomized Projection Component 2", fontsize=14)
    plt.legend(title="Label", fontsize="large", title_fontsize="13", loc="best")

    plt.savefig(
        rf"{output_filepath}{dataset_type}_rp_2_randomized_projection_component_scatter_plot.png"
    )
    plt.close()


def get_optimal_tsne_components(
    data: pd.DataFrame,
    max_components: int = 10,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    max_components = min(max_components, data.shape[0], data.shape[1])

    kl_divergences = []
    components_range = range(1, max_components + 1)
    for n_components in components_range:
        tsne_reduction = TSNEReduction(n_components=n_components, random_state=42)
        tsne_reduction.fit_transform(data)
        kl_divergences.append(tsne_reduction.kl_divergence())

    plt.figure(figsize=(10, 6))
    plt.plot(components_range, kl_divergences, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Final KL Divergence")
    plt.title("t-SNE Final KL Divergence for Different Numbers of Components")
    plt.grid(True)

    plt.savefig(
        rf"{output_filepath}{dataset_type}_tsne_kl_divergence_per_n_components.png"
    )
    plt.close()


def get_t_sne_transformed_output(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    output_filepath: str = "../output/dimensionality_reduction/",
    dataset_type: str = "auction",
) -> None:
    tsne_reduction = RandomProjectionDimensionalityReduction(n_components=2)
    tsne_reduction.fit(train_X)
    transformed_train_X = tsne_reduction.transform(train_X)

    data = pd.concat(
        [pd.DataFrame(transformed_train_X), train_y.reset_index(drop=True)], axis=1
    )
    data.columns = ["t-SNE Component 1", "t-SNE Component 2", "Label"]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x="t-SNE Component 1",
        y="t-SNE Component 2",
        hue="Label",
        palette="viridis",
        s=100,
        alpha=0.7,
    )

    plt.grid(True)

    plt.title(f"{dataset_type.capitalize()}: t-SNE  - 2 t-SNE Components", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(title="Label", fontsize="large", title_fontsize="13", loc="best")

    plt.savefig(
        rf"{output_filepath}{dataset_type}_t_sne_2_t_sne_component_scatter_plot.png"
    )
    plt.close()


def get_k_means_for_all_dimensionality_reduction_techniques(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    auction_optimal_component_selection = 4
    dropout_optimal_component_selection = 10  # t-SNE takes too long on dropout; omit it

    algorithm_acronyms = ["pca", "ica", "rp", "t_sne"]

    auction_algorithms = [
        PCADimensionalityReduction(n_components=auction_optimal_component_selection),
        ICADimensionalityReduction(n_components=auction_optimal_component_selection),
        RandomProjectionDimensionalityReduction(
            n_components=auction_optimal_component_selection
        ),
        TSNEReduction(n_components=auction_optimal_component_selection),
    ]

    dropout_algorithms = [
        PCADimensionalityReduction(n_components=dropout_optimal_component_selection),
        ICADimensionalityReduction(n_components=dropout_optimal_component_selection),
        RandomProjectionDimensionalityReduction(
            n_components=dropout_optimal_component_selection
        ),
        None,
    ]

    max_clusters = 30
    metric = "euclidean"
    for auction_algorithm, dropout_algorithm, algorithm_acronym in zip(
        auction_algorithms, dropout_algorithms, algorithm_acronyms
    ):
        transformed_auction_train_X = pd.DataFrame(
            auction_algorithm.fit_transform(data=auction_train_X)
        )
        if algorithm_acronym != "t_sne":
            transformed_dropout_train_X = pd.DataFrame(
                dropout_algorithm.fit_transform(data=dropout_train_X)
            )

        output_filepath = rf"../output/combined_clustering_dimensionality_reduction/{algorithm_acronym}/"

        #######################################
        # K-Means: Auction
        #######################################
        num_iterations = 10
        get_k_means_elbow_graph(
            train_X=transformed_auction_train_X,
            train_y=auction_train_y,
            max_clusters=max_clusters,
            num_iterations=num_iterations,
            metric=metric,
            output_filepath=output_filepath,
            dataset_type="auction",
        )

        num_iterations = 50
        get_k_means_metric_vs_f1_score(
            train_X=transformed_auction_train_X,
            train_y=auction_train_y,
            num_iterations=num_iterations,
            output_filepath=output_filepath,
            dataset_type="auction",
        )

        #######################################
        # K-Means: Dropout
        #######################################
        if algorithm_acronym != "t_sne":
            num_iterations = 10
            get_k_means_elbow_graph(
                train_X=transformed_dropout_train_X,
                train_y=dropout_train_y,
                max_clusters=max_clusters,
                num_iterations=num_iterations,
                metric=metric,
                output_filepath=output_filepath,
                dataset_type="dropout",
            )

            num_iterations = 50
            get_k_means_metric_vs_f1_score(
                train_X=transformed_dropout_train_X,
                train_y=dropout_train_y,
                num_iterations=num_iterations,
                output_filepath=output_filepath,
                dataset_type="dropout",
            )


def get_expected_maximization_for_all_dimensionality_reduction_techniques(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    auction_optimal_component_selection = 4
    dropout_optimal_component_selection = 10  # t-SNE takes too long on dropout; omit it

    algorithm_acronyms = ["pca", "ica", "rp", "t_sne"]

    auction_algorithms = [
        PCADimensionalityReduction(n_components=auction_optimal_component_selection),
        ICADimensionalityReduction(n_components=auction_optimal_component_selection),
        RandomProjectionDimensionalityReduction(
            n_components=auction_optimal_component_selection
        ),
        TSNEReduction(n_components=auction_optimal_component_selection),
    ]

    dropout_algorithms = [
        PCADimensionalityReduction(n_components=dropout_optimal_component_selection),
        ICADimensionalityReduction(n_components=dropout_optimal_component_selection),
        RandomProjectionDimensionalityReduction(
            n_components=dropout_optimal_component_selection
        ),
        None,
    ]

    max_clusters = 30
    metric = "euclidean"
    for auction_algorithm, dropout_algorithm, algorithm_acronym in zip(
        auction_algorithms, dropout_algorithms, algorithm_acronyms
    ):
        transformed_auction_train_X = pd.DataFrame(
            auction_algorithm.fit_transform(data=auction_train_X)
        )
        if algorithm_acronym != "t_sne":
            transformed_dropout_train_X = pd.DataFrame(
                dropout_algorithm.fit_transform(data=dropout_train_X)
            )

        output_filepath = rf"../output/combined_clustering_dimensionality_reduction/{algorithm_acronym}/"

        #######################################
        # Expected Maximization: Auction
        #######################################
        get_expected_maximization_performance_line_charts(
            train_X=transformed_auction_train_X,
            train_y=auction_train_y,
            output_filepath=output_filepath,
            dataset_type="auction",
        )

        #######################################
        # Expected Maximization: Dropout
        #######################################
        get_expected_maximization_performance_line_charts(
            train_X=transformed_dropout_train_X,
            train_y=dropout_train_y,
            output_filepath=output_filepath,
            dataset_type="dropout",
        )


def get_neural_network_performance(
    # Auction
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    auction_val_X: pd.DataFrame,
    auction_val_y: pd.DataFrame,
    auction_test_X: pd.DataFrame,
    auction_test_y: pd.DataFrame,
    # Dropout
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
    dropout_val_X: pd.DataFrame,
    dropout_val_y: pd.DataFrame,
    dropout_test_X: pd.DataFrame,
    dropout_test_y: pd.DataFrame,
    clustering_algorithm: str = "None",
    dimensionality_reduction_algorithm: str = "None",
) -> tuple[pd.DataFrame, list[float], list[float], list[float], list[float]]:
    #####################
    ## Auction
    #####################
    auction_train_dataset = TensorDataset(
        torch.tensor(auction_train_X.values, dtype=torch.float32),
        torch.tensor(auction_train_y.values, dtype=torch.float32),
    )
    auction_validation_dataset = TensorDataset(
        torch.tensor(auction_val_X.values, dtype=torch.float32),
        torch.tensor(auction_val_y.values, dtype=torch.float32),
    )
    auction_test_dataset = TensorDataset(
        torch.tensor(auction_test_X.values, dtype=torch.float32),
        torch.tensor(auction_test_y.values, dtype=torch.float32),
    )

    auction_train_loader = DataLoader(
        auction_train_dataset, batch_size=32, shuffle=True
    )
    auction_val_loader = DataLoader(auction_validation_dataset, batch_size=32)
    auction_test_loader = DataLoader(auction_test_dataset, batch_size=32)

    input_size = auction_train_X.shape[1]
    num_epochs = 400
    (
        best_model,
        auction_training_loss_history,
        auction_validation_loss_history,
    ) = train_neural_network(
        auction_train_loader,
        auction_val_loader,
        input_size,
        num_epochs,
        num_classes=2,
        multiclass=False,
    )

    auction_test_auc, auction_test_accuracy = evaluate_model(
        best_model,
        auction_test_loader,
        num_classes=2,
    )

    #####################
    ## Dropout
    #####################
    dropout_train_dataset = TensorDataset(
        torch.tensor(dropout_train_X.values, dtype=torch.float32),
        torch.tensor(dropout_train_y.values, dtype=torch.float32),
    )
    dropout_validation_dataset = TensorDataset(
        torch.tensor(dropout_val_X.values, dtype=torch.float32),
        torch.tensor(dropout_val_y.values, dtype=torch.float32),
    )
    dropout_test_dataset = TensorDataset(
        torch.tensor(dropout_test_X.values, dtype=torch.float32),
        torch.tensor(dropout_test_y.values, dtype=torch.float32),
    )

    dropout_train_loader = DataLoader(
        dropout_train_dataset, batch_size=32, shuffle=True
    )
    dropout_val_loader = DataLoader(dropout_validation_dataset, batch_size=32)
    dropout_test_loader = DataLoader(dropout_test_dataset, batch_size=32)

    input_size = dropout_train_X.shape[1]
    num_epochs = 400
    (
        best_model,
        dropout_training_loss_history,
        dropout_validation_loss_history,
    ) = train_neural_network(
        dropout_train_loader,
        dropout_val_loader,
        input_size,
        num_epochs,
        num_classes=3,
        multiclass=True,
    )

    dropout_test_auc, dropout_test_accuracy = evaluate_model(
        best_model,
        dropout_test_loader,
        num_classes=3,
    )

    accuracy_auc_df = pd.DataFrame(
        [
            [
                dimensionality_reduction_algorithm,
                clustering_algorithm,
                "Auction",
                auction_test_accuracy,
                auction_test_auc,
            ],
            [
                dimensionality_reduction_algorithm,
                clustering_algorithm,
                "Dropout",
                dropout_test_accuracy,
                dropout_test_auc,
            ],
        ],
        columns=[
            "Dimensional Reduction Algorithm",
            "Clustering Algorithm",
            "Dataset",
            "Accuracy",
            "AUC",
        ],
    )

    return (
        accuracy_auc_df,
        auction_training_loss_history,
        auction_validation_loss_history,
        dropout_training_loss_history,
        dropout_validation_loss_history,
    )


def get_neural_network_performance_by_dimensionality_reduction_algorithm(
    # Auction
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    auction_val_X: pd.DataFrame,
    auction_val_y: pd.DataFrame,
    auction_test_X: pd.DataFrame,
    auction_test_y: pd.DataFrame,
    # Dropout
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
    dropout_val_X: pd.DataFrame,
    dropout_val_y: pd.DataFrame,
    dropout_test_X: pd.DataFrame,
    dropout_test_y: pd.DataFrame,
) -> pd.DataFrame:
    def plot_performance_curves(
        training_loss_history: list[float],
        validation_loss_history: list[float],
        filename: str = "",
        dataset_type: str = "auction",
    ) -> None:
        plt.figure(figsize=(12, 6))

        plt.plot(training_loss_history, color="red", label="Training")
        plt.plot(
            validation_loss_history,
            color="blue",
            label="Validation",
            linestyle="dotted",
        )
        plt.legend()

        plt.title(
            rf"{dataset_type.title()}: Neural Network Performance Curve per Epoch"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Metric")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(filename)

    auction_optimal_component_selection = 4
    dropout_optimal_component_selection = 10  # t-SNE takes too long on dropout; omit it

    algorithm_acronyms = ["pca", "ica", "rp", "t_sne"]

    auction_algorithms = [
        PCADimensionalityReduction(n_components=auction_optimal_component_selection),
        ICADimensionalityReduction(n_components=auction_optimal_component_selection),
        RandomProjectionDimensionalityReduction(
            n_components=auction_optimal_component_selection
        ),
        TSNEReduction(n_components=auction_optimal_component_selection),
    ]

    dropout_algorithms = [
        PCADimensionalityReduction(n_components=dropout_optimal_component_selection),
        ICADimensionalityReduction(n_components=dropout_optimal_component_selection),
        RandomProjectionDimensionalityReduction(
            n_components=dropout_optimal_component_selection
        ),
        TSNEReduction(n_components=2),
    ]

    final_accuracy_auc_df = pd.DataFrame(
        [],
        columns=[
            "Dimensional Reduction Algorithm",
            "Clustering Algorithm",
            "Dataset",
            "Accuracy",
            "AUC",
        ],
    )
    for auction_algorithm, dropout_algorithm, algorithm_acronym in zip(
        auction_algorithms, dropout_algorithms, algorithm_acronyms
    ):
        transformed_auction_train_X = pd.DataFrame(
            auction_algorithm.fit_transform(data=auction_train_X)
        )

        transformed_dropout_train_X = pd.DataFrame(
            dropout_algorithm.fit_transform(data=dropout_train_X)
        )

        if algorithm_acronym != "t_sne":
            transformed_auction_val_X = pd.DataFrame(
                auction_algorithm.transform(data=auction_val_X)
            )
            transformed_auction_test_X = pd.DataFrame(
                auction_algorithm.transform(data=auction_test_X)
            )

            transformed_dropout_val_X = pd.DataFrame(
                dropout_algorithm.transform(data=dropout_val_X)
            )
            transformed_dropout_test_X = pd.DataFrame(
                dropout_algorithm.transform(data=dropout_test_X)
            )
        else:
            transformed_auction_val_X = pd.DataFrame(
                auction_algorithm.fit_transform(data=auction_val_X)
            )
            transformed_auction_test_X = pd.DataFrame(
                auction_algorithm.fit_transform(data=auction_test_X)
            )
            transformed_dropout_val_X = pd.DataFrame(
                dropout_algorithm.fit_transform(data=dropout_val_X)
            )
            transformed_dropout_test_X = pd.DataFrame(
                dropout_algorithm.fit_transform(data=dropout_test_X)
            )

        (
            accuracy_auc_df,
            auction_training_loss_history,
            auction_validation_loss_history,
            dropout_training_loss_history,
            dropout_validation_loss_history,
        ) = get_neural_network_performance(
            transformed_auction_train_X,
            auction_train_y,
            transformed_auction_val_X,
            auction_val_y,
            transformed_auction_test_X,
            auction_test_y,
            transformed_dropout_train_X,
            dropout_train_y,
            transformed_dropout_val_X,
            dropout_val_y,
            transformed_dropout_test_X,
            dropout_test_y,
            dimensionality_reduction_algorithm=algorithm_acronym,
        )

        dataset_type = "auction"
        output_filename = rf"../output/neural_network/dimensionality_reduction/{algorithm_acronym}/{dataset_type}_performance_curve.png"
        plot_performance_curves(
            training_loss_history=auction_training_loss_history,
            validation_loss_history=auction_validation_loss_history,
            filename=output_filename,
            dataset_type=dataset_type,
        )

        dataset_type = "dropout"
        output_filename = rf"../output/neural_network/dimensionality_reduction/{algorithm_acronym}/{dataset_type}_performance_curve.png"
        plot_performance_curves(
            training_loss_history=dropout_training_loss_history,
            validation_loss_history=dropout_validation_loss_history,
            filename=output_filename,
            dataset_type=dataset_type,
        )

        final_accuracy_auc_df = pd.concat((final_accuracy_auc_df, accuracy_auc_df))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=final_accuracy_auc_df.values, colLabels=final_accuracy_auc_df.columns, loc='center')
    plt.savefig("../output/neural_network/dimensionality_reduction/dimensionality_reduction_accuracy_auc_table.png", dpi=200, bbox_inches='tight')

    return final_accuracy_auc_df


def get_neural_network_performance_by_clustering_algorithm(
    # Auction
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    auction_val_X: pd.DataFrame,
    auction_val_y: pd.DataFrame,
    auction_test_X: pd.DataFrame,
    auction_test_y: pd.DataFrame,
    # Dropout
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
    dropout_val_X: pd.DataFrame,
    dropout_val_y: pd.DataFrame,
    dropout_test_X: pd.DataFrame,
    dropout_test_y: pd.DataFrame,
) -> pd.DataFrame:
    def plot_performance_curves(
        training_loss_history: list[float],
        validation_loss_history: list[float],
        filename: str = "",
        dataset_type: str = "auction",
    ) -> None:
        plt.figure(figsize=(12, 6))

        plt.plot(training_loss_history, color="red", label="Training")
        plt.plot(
            validation_loss_history,
            color="blue",
            label="Validation",
            linestyle="dotted",
        )
        plt.legend()

        plt.title(
            rf"{dataset_type.title()}: Neural Network Performance Curve per Epoch"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Metric")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(filename)

    algorithm_acronyms = [
        "km",
        "em",
    ]

    auction_algorithms = [
        KMeans(
            X=auction_train_X,
            true_labels=auction_train_y.values.astype(int).flatten(),
            n_clusters=30,
            metric="euclidean",
        ),
        GaussianMixture(n_components=10, max_iter=100, random_state=42),
    ]

    dropout_algorithms = [
        KMeans(
            X=dropout_train_X,
            true_labels=dropout_train_y.values.astype(int).flatten(),
            n_clusters=30,
            metric="euclidean",
        ),
        GaussianMixture(n_components=10, max_iter=100, random_state=42),
    ]

    final_accuracy_auc_df = pd.DataFrame(
        [],
        columns=[
            "Dimensional Reduction Algorithm",
            "Clustering Algorithm",
            "Dataset",
            "Accuracy",
            "AUC",
        ],
    )
    for auction_algorithm, dropout_algorithm, algorithm_acronym in zip(
        auction_algorithms, dropout_algorithms, algorithm_acronyms
    ):
        if algorithm_acronym == "km":
            #######################
            ## Auction
            #######################
            # Train
            transformed_auction_train_X, _ = auction_algorithm.get_k_means(50)
            transformed_auction_train_X = pd.DataFrame(
                np.array([transformed_auction_train_X]).T, columns=["Cluster"]
            )
            transformed_auction_train_X["Cluster"] = pd.Categorical(
                transformed_auction_train_X["Cluster"],
                categories=np.arange(0, 30, 1).astype(int),
            )
            transformed_auction_train_X = pd.get_dummies(
                transformed_auction_train_X, columns=["Cluster"], drop_first=False
            )

            # Validation
            auction_algorithm.X = auction_val_X.values
            transformed_auction_val_X, _ = auction_algorithm.get_k_means(50)
            transformed_auction_val_X = pd.DataFrame(
                np.array([transformed_auction_val_X]).T, columns=["Cluster"]
            )
            transformed_auction_val_X["Cluster"] = pd.Categorical(
                transformed_auction_val_X["Cluster"],
                categories=np.arange(0, 30, 1).astype(int),
            )
            transformed_auction_val_X = pd.get_dummies(
                transformed_auction_val_X, columns=["Cluster"], drop_first=False
            )

            # Test
            auction_algorithm.X = auction_test_X.values
            transformed_auction_test_X, _ = auction_algorithm.get_k_means(50)
            transformed_auction_test_X = pd.DataFrame(
                np.array([transformed_auction_test_X]).T, columns=["Cluster"]
            )
            transformed_auction_test_X["Cluster"] = pd.Categorical(
                transformed_auction_test_X["Cluster"],
                categories=np.arange(0, 30, 1).astype(int),
            )
            transformed_auction_test_X = pd.get_dummies(
                transformed_auction_test_X, columns=["Cluster"], drop_first=False
            )

            #######################
            ## Dropout
            #######################
            # Train
            transformed_dropout_train_X, _ = dropout_algorithm.get_k_means(50)
            transformed_dropout_train_X = pd.DataFrame(
                np.array([transformed_dropout_train_X]).T, columns=["Cluster"]
            )
            transformed_dropout_train_X["Cluster"] = pd.Categorical(
                transformed_dropout_train_X["Cluster"],
                categories=np.arange(0, 30, 1).astype(int),
            )
            transformed_dropout_train_X = pd.get_dummies(
                transformed_dropout_train_X, columns=["Cluster"], drop_first=False
            )

            # Validation
            dropout_algorithm.X = dropout_val_X.values
            transformed_dropout_val_X, _ = dropout_algorithm.get_k_means(50)
            transformed_dropout_val_X = pd.DataFrame(
                np.array([transformed_dropout_val_X]).T, columns=["Cluster"]
            )
            transformed_dropout_val_X["Cluster"] = pd.Categorical(
                transformed_dropout_val_X["Cluster"],
                categories=np.arange(0, 30, 1).astype(int),
            )
            transformed_dropout_val_X = pd.get_dummies(
                transformed_dropout_val_X, columns=["Cluster"], drop_first=False
            )

            # Test
            dropout_algorithm.X = dropout_test_X.values
            transformed_dropout_test_X, _ = dropout_algorithm.get_k_means(50)
            transformed_dropout_test_X = pd.DataFrame(
                np.array([transformed_dropout_test_X]).T, columns=["Cluster"]
            )
            transformed_dropout_test_X["Cluster"] = pd.Categorical(
                transformed_dropout_test_X["Cluster"],
                categories=np.arange(0, 30, 1).astype(int),
            )
            transformed_dropout_test_X = pd.get_dummies(
                transformed_dropout_test_X, columns=["Cluster"], drop_first=False
            )

        else:
            #######################
            ## Auction
            #######################
            # Training
            auction_algorithm.fit(auction_train_X.to_numpy())
            transformed_auction_train_X = auction_algorithm.predict(
                auction_train_X.to_numpy()
            )
            transformed_auction_train_X = pd.DataFrame(
                np.array([transformed_auction_train_X]).T, columns=["Cluster"]
            )
            transformed_auction_train_X["Cluster"] = pd.Categorical(
                transformed_auction_train_X["Cluster"],
                categories=np.arange(0, 10, 1).astype(int),
            )
            transformed_auction_train_X = pd.get_dummies(
                transformed_auction_train_X, columns=["Cluster"], drop_first=False
            )

            # Validation
            auction_algorithm.fit(auction_val_X.to_numpy())
            transformed_auction_val_X = auction_algorithm.predict(
                auction_val_X.to_numpy()
            )
            transformed_auction_val_X = pd.DataFrame(
                np.array([transformed_auction_val_X]).T, columns=["Cluster"]
            )
            transformed_auction_val_X["Cluster"] = pd.Categorical(
                transformed_auction_val_X["Cluster"],
                categories=np.arange(0, 10, 1).astype(int),
            )
            transformed_auction_val_X = pd.get_dummies(
                transformed_auction_val_X, columns=["Cluster"], drop_first=False
            )

            # Test
            auction_algorithm.fit(auction_test_X.to_numpy())
            transformed_auction_test_X = auction_algorithm.predict(
                auction_test_X.to_numpy()
            )
            transformed_auction_test_X = pd.DataFrame(
                np.array([transformed_auction_test_X]).T, columns=["Cluster"]
            )
            transformed_auction_test_X["Cluster"] = pd.Categorical(
                transformed_auction_test_X["Cluster"],
                categories=np.arange(0, 10, 1).astype(int),
            )
            transformed_auction_test_X = pd.get_dummies(
                transformed_auction_test_X, columns=["Cluster"], drop_first=False
            )

            #######################
            ## Dropout
            #######################
            # Training
            dropout_algorithm.fit(dropout_train_X.to_numpy())
            transformed_dropout_train_X = dropout_algorithm.predict(
                dropout_train_X.to_numpy()
            )
            transformed_dropout_train_X = pd.DataFrame(
                np.array([transformed_dropout_train_X]).T, columns=["Cluster"]
            )
            transformed_dropout_train_X["Cluster"] = pd.Categorical(
                transformed_dropout_train_X["Cluster"],
                categories=np.arange(0, 10, 1).astype(int),
            )
            transformed_dropout_train_X = pd.get_dummies(
                transformed_dropout_train_X, columns=["Cluster"], drop_first=False
            )

            # Validation
            dropout_algorithm.fit(dropout_val_X.to_numpy())
            transformed_dropout_val_X = dropout_algorithm.predict(
                dropout_val_X.to_numpy()
            )
            transformed_dropout_val_X = pd.DataFrame(
                np.array([transformed_dropout_val_X]).T, columns=["Cluster"]
            )
            transformed_dropout_val_X["Cluster"] = pd.Categorical(
                transformed_dropout_val_X["Cluster"],
                categories=np.arange(0, 10, 1).astype(int),
            )
            transformed_dropout_val_X = pd.get_dummies(
                transformed_dropout_val_X, columns=["Cluster"], drop_first=False
            )

            # Test
            dropout_algorithm.fit(dropout_test_X.to_numpy())
            transformed_dropout_test_X = dropout_algorithm.predict(
                dropout_test_X.to_numpy()
            )
            transformed_dropout_test_X = pd.DataFrame(
                np.array([transformed_dropout_test_X]).T, columns=["Cluster"]
            )
            transformed_dropout_test_X["Cluster"] = pd.Categorical(
                transformed_dropout_test_X["Cluster"],
                categories=np.arange(0, 10, 1).astype(int),
            )
            transformed_dropout_test_X = pd.get_dummies(
                transformed_dropout_test_X, columns=["Cluster"], drop_first=False
            )

        (
            accuracy_auc_df,
            auction_training_loss_history,
            auction_validation_loss_history,
            dropout_training_loss_history,
            dropout_validation_loss_history,
        ) = get_neural_network_performance(
            transformed_auction_train_X,
            auction_train_y,
            transformed_auction_val_X,
            auction_val_y,
            transformed_auction_test_X,
            auction_test_y,
            transformed_dropout_train_X,
            dropout_train_y,
            transformed_dropout_val_X,
            dropout_val_y,
            transformed_dropout_test_X,
            dropout_test_y,
            clustering_algorithm=algorithm_acronym,
        )

        dataset_type = "auction"
        output_filename = rf"../output/neural_network/clustering/{algorithm_acronym}/{dataset_type}_performance_curve.png"
        plot_performance_curves(
            training_loss_history=auction_training_loss_history,
            validation_loss_history=auction_validation_loss_history,
            filename=output_filename,
            dataset_type=dataset_type,
        )

        dataset_type = "dropout"
        output_filename = rf"../output/neural_network/clustering/{algorithm_acronym}/{dataset_type}_performance_curve.png"
        plot_performance_curves(
            training_loss_history=dropout_training_loss_history,
            validation_loss_history=dropout_validation_loss_history,
            filename=output_filename,
            dataset_type=dataset_type,
        )

        final_accuracy_auc_df = pd.concat((final_accuracy_auc_df, accuracy_auc_df))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=final_accuracy_auc_df.values, colLabels=final_accuracy_auc_df.columns, loc='center')
    plt.savefig("../output/neural_network/clustering/clustering_accuracy_auc_table.png", dpi=200, bbox_inches='tight')

    return final_accuracy_auc_df
