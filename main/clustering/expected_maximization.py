# Standard library imports
from itertools import permutations

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
import warnings

# Local (custom) imports
from data_preprocessing import preprocess_datasets

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)


def get_accuracy_and_f1_score(
    true_labels: np.ndarray, assignments: np.ndarray
) -> tuple[float, float]:
    unique_labels = list(set(assignments).union(set(true_labels)))
    all_permutations = list(permutations(unique_labels))

    cluster_mappings = []
    for permutation in all_permutations:
        mapping = {original: new for original, new in zip(unique_labels, permutation)}
        cluster_mapping = [
            mapping.get(assignment, assignment) for assignment in assignments
        ]
        cluster_mappings.append(cluster_mapping)

    accuracies = [
        accuracy_score(true_labels, cluster_mapping)
        for cluster_mapping in cluster_mappings
    ]

    best_accuracy_mapping_index = np.argmax(np.array(accuracies))

    accuracy = accuracies[best_accuracy_mapping_index]
    f1 = f1_score(
        true_labels,
        cluster_mappings[best_accuracy_mapping_index],
        average="macro",
    )

    return (accuracy, f1)


def get_gmm_bic_aic_accuracy_f1(
    data: pd.DataFrame, n_components: int = 2
) -> tuple[any, float, float, float, float]:
    gmm = GaussianMixture(n_components=n_components, max_iter=100, random_state=42)
    gmm.fit(auction_train_X.to_numpy())

    labels = gmm.predict(auction_train_X.to_numpy())

    _aic = gmm.aic(auction_train_X.to_numpy())
    _bic = gmm.bic(auction_train_X.to_numpy())

    (accuracy, f1) = get_accuracy_and_f1_score(
        true_labels=auction_train_y.to_numpy().flatten(), assignments=labels
    )

    return (gmm, _aic, _bic, accuracy, f1)


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
