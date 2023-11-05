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
import time

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)

from helper_functions import (
    fit_k_means,
    save_elbow_graph,
    get_k_means_elbow_graph,
    get_distance_metric_bar_plot,
    get_k_means_metric_vs_f1_score,
    get_expected_maximization_performance_line_charts,
    get_pca_explained_variance,
    get_pca_transformed_output,
    get_optimal_ica_components,
    get_ica_transformed_output,
    get_optimal_randomized_projection_components,
    get_randomized_projection_transformed_output,
    get_optimal_tsne_components,
    get_t_sne_transformed_output,
    get_k_means_for_all_dimensionality_reduction_techniques,
    get_expected_maximization_for_all_dimensionality_reduction_techniques,
    get_neural_network_performance_by_dimensionality_reduction_algorithm,
    get_neural_network_performance_by_clustering_algorithm,
)


def _timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        print(f"{func.__name__} executed in {execution_time} seconds")
        return result

    return wrapper


@_timer
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


@_timer
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


@_timer
def part_2_1(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    get_pca_explained_variance(
        auction_train_X,
        dropout_train_X,
    )

    get_pca_transformed_output(
        auction_train_X,
        auction_train_y,
        dataset_type="auction",
    )

    get_pca_transformed_output(
        dropout_train_X,
        dropout_train_y,
        dataset_type="dropout",
    )


@_timer
def part_2_2(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    get_optimal_ica_components(
        data=auction_train_X,
        max_components=10,
        dataset_type="auction",
    )
    get_optimal_ica_components(
        data=dropout_train_X,
        max_components=21,
        dataset_type="dropout",
    )

    get_ica_transformed_output(
        auction_train_X,
        auction_train_y,
        dataset_type="auction",
    )

    get_ica_transformed_output(
        dropout_train_X,
        dropout_train_y,
        dataset_type="dropout",
    )


@_timer
def part_2_3(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    get_optimal_randomized_projection_components(
        auction_train_X,
        max_components=10,
        dataset_type="auction",
    )
    get_optimal_randomized_projection_components(
        dropout_train_X,
        max_components=51,
        dataset_type="dropout",
    )
    get_randomized_projection_transformed_output(
        auction_train_X,
        auction_train_y,
        dataset_type="auction",
    )
    get_randomized_projection_transformed_output(
        dropout_train_X,
        dropout_train_y,
        dataset_type="dropout",
    )


@_timer
def part_2_4(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    get_optimal_tsne_components(
        auction_train_X,
        max_components=10,
        dataset_type="auction",
    )
    get_optimal_tsne_components(
        dropout_train_X,
        max_components=3,
        dataset_type="dropout",
    )
    get_t_sne_transformed_output(
        auction_train_X,
        auction_train_y,
        dataset_type="auction",
    )
    get_t_sne_transformed_output(
        dropout_train_X,
        dropout_train_y,
        dataset_type="dropout",
    )


def part_3(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
) -> None:
    get_k_means_for_all_dimensionality_reduction_techniques(
        auction_train_X,
        auction_train_y,
        dropout_train_X,
        dropout_train_y,
    )
    get_expected_maximization_for_all_dimensionality_reduction_techniques(
        auction_train_X,
        auction_train_y,
        dropout_train_X,
        dropout_train_y,
    )


def part_4(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    auction_val_X: pd.DataFrame,
    auction_val_y: pd.DataFrame,
    auction_test_X: pd.DataFrame,
    auction_test_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
    dropout_val_X: pd.DataFrame,
    dropout_val_y: pd.DataFrame,
    dropout_test_X: pd.DataFrame,
    dropout_test_y: pd.DataFrame,
) -> None:
    get_neural_network_performance_by_dimensionality_reduction_algorithm(
        auction_train_X,
        auction_train_y,
        auction_val_X,
        auction_val_y,
        auction_test_X,
        auction_test_y,
        dropout_train_X,
        dropout_train_y,
        dropout_val_X,
        dropout_val_y,
        dropout_test_X,
        dropout_test_y,
    )


def part_5(
    auction_train_X: pd.DataFrame,
    auction_train_y: pd.DataFrame,
    auction_val_X: pd.DataFrame,
    auction_val_y: pd.DataFrame,
    auction_test_X: pd.DataFrame,
    auction_test_y: pd.DataFrame,
    dropout_train_X: pd.DataFrame,
    dropout_train_y: pd.DataFrame,
    dropout_val_X: pd.DataFrame,
    dropout_val_y: pd.DataFrame,
    dropout_test_X: pd.DataFrame,
    dropout_test_y: pd.DataFrame,
) -> None:
    get_neural_network_performance_by_clustering_algorithm(
        auction_train_X,
        auction_train_y,
        auction_val_X,
        auction_val_y,
        auction_test_X,
        auction_test_y,
        dropout_train_X,
        dropout_train_y,
        dropout_val_X,
        dropout_val_y,
        dropout_test_X,
        dropout_test_y,
    )


if __name__ == "__main__":
    RUN_PART_1 = False
    RUN_PART_2 = False
    RUN_PART_3 = False
    RUN_PART_4 = False
    RUN_PART_5 = True

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
        ## TODO: Get accuracy and F1 stats for auction and dropout using expected maximization
    if RUN_PART_2:
        part_2_1(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
        part_2_2(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
        part_2_3(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
        part_2_4(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
    if RUN_PART_3:
        part_3(auction_train_X, auction_train_y, dropout_train_X, dropout_train_y)
    if RUN_PART_4:
        part_4(
            auction_train_X,
            auction_train_y,
            auction_val_X,
            auction_val_y,
            auction_test_X,
            auction_test_y,
            dropout_train_X,
            dropout_train_y,
            dropout_val_X,
            dropout_val_y,
            dropout_test_X,
            dropout_test_y,
        )
    if RUN_PART_5:
        part_5(
            auction_train_X,
            auction_train_y,
            auction_val_X,
            auction_val_y,
            auction_test_X,
            auction_test_y,
            dropout_train_X,
            dropout_train_y,
            dropout_val_X,
            dropout_val_y,
            dropout_test_X,
            dropout_test_y,
        )
