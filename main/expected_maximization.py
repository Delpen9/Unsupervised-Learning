# Standard Libraries
import time

# Data Science Libraries
import numpy as np
import pandas as pd
from scipy.spatial import distance
from itertools import permutations

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# SKLearn Libraries
from sklearn.mixture import GaussianMixture

# Metrics
from sklearn.metrics import accuracy_score, f1_score

from data_preprocessing import (
    preprocess_datasets,
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


class ExpectedMaximizer:
    def __init__(
        self,
        X: pd.DataFrame,
        true_labels: list[int],
        n_clusters: int = 2,
        metric: str = "euclidean",
    ):
        self.X = X.values
        self.true_labels = true_labels
        self.n_clusters = n_clusters
        self.metric = metric
        self.assignments = []
        self.centroids = []

        self.inertia = None
        self.cluster_sizes = [None, None]
        self.accuracy = None
        self.f1 = None

        self.execution_time = None

    def _compute_distance(self, x: list[float], y: list[float]) -> float:
        """
        Compute the distance between two points using a specified metric.

        Args:
            x (list[float]): The first point.
            y (list[float]): The second point.

        Returns:
            float: The distance between x and y based on the specified metric.

        Raises:
            ValueError: If the specified metric is not supported.
        """
        if self.metric == "euclidean":
            return np.linalg.norm(np.array(x) - np.array(y))
        elif self.metric == "manhattan":
            return distance.cityblock(x, y)
        elif self.metric == "cosine":
            return distance.cosine(x, y)
        elif self.metric == "chebyshev":
            return distance.chebyshev(x, y)
        else:
            raise ValueError(rf"Invalid metric: {self.metric}")

    def expectation_step(
        self,
    ) -> list[int]:
        """
        Assign each data point to the closest centroid.

        Returns:
            list[int]: A list of cluster assignments for each data point.
        """
        assignments = []
        for x in self.X:
            distances = [
                self._compute_distance(x, centroid) for centroid in self.centroids
            ]
            assignments.append(np.argmin(distances))
        return assignments

    def maximization_step(self) -> list[list[float]]:
        """
        Compute the new centroids based on the current assignments.

        Returns:
            list[list[float]]: A list of the new centroids.
        """
        new_centroids = []
        for i in range(self.n_clusters):
            members = self.X[np.where(np.array(self.assignments) == i)]
            if len(members) > 0:
                new_centroids.append(members.mean(axis=0))
            else:
                new_centroids.append(np.random.randn(self.X.shape[1]))
        return new_centroids

    @_timer
    def get_expected_maximization(
        self, num_iterations: int = 10
    ) -> tuple[list[int], list[list[float]]]:
        """
        Perform the expectation-maximization algorithm.

        Args:
            num_iterations (int, optional): The number of iterations to run. Defaults to 10.

        Returns:
            tuple[list[int], list[list[float]]]: The final cluster assignments and centroids.
        """
        start_time = time.time()
        self.centroids = [self.X[i] for i in range(self.n_clusters)]

        for i in range(num_iterations):
            self.assignments = self.expectation_step()
            self.centroids = self.maximization_step()

        end_time = time.time()
        self.execution_time = round(end_time - start_time, 2)

        return (self.assignments, self.centroids)

    def compute_inertia(self) -> float:
        """
        Compute the inertia (sum of squared distances from points to their centroids).

        Returns:
            float: The inertia of the current clustering.
        """
        self.inertia = 0
        for i, x in enumerate(self.X):
            self.inertia += (
                self._compute_distance(x, self.centroids[self.assignments[i]]) ** 2
            )
        return self.inertia

    def get_cluster_sizes(self) -> list[int]:
        """
        Calculate the size of each cluster.

        Returns:
            list[int]: The number of data points in each cluster.
        """
        self.cluster_sizes = [
            np.sum(np.array(self.assignments) == i) for i in range(self.n_clusters)
        ]
        return self.cluster_sizes

    def get_accuracy_and_f1_score(self) -> tuple[float, float]:
        """
        Calculate the accuracy and F1 score for the current cluster assignments.

        Returns:
            tuple[float, float]: The accuracy and F1 score.

        Raises:
            ValueError: If the number of unique true labels does not match the number of clusters.
        """
        if len(set(self.true_labels)) != self.n_clusters:
            raise ValueError(
                "Number of unique true labels does not match the number of clusters."
            )

        unique_labels = list(set(self.assignments).union(set(self.true_labels)))
        all_permutations = list(permutations(unique_labels))

        cluster_mappings = []
        for permutation in all_permutations:
            mapping = {
                original: new for original, new in zip(unique_labels, permutation)
            }
            cluster_mapping = [
                mapping.get(assignment, assignment) for assignment in self.assignments
            ]
            cluster_mappings.append(cluster_mapping)

        accuracies = [
            accuracy_score(self.true_labels, cluster_mapping)
            for cluster_mapping in cluster_mappings
        ]

        best_accuracy_mapping_index = np.argmax(np.array(accuracies))

        self.accuracy = accuracies[best_accuracy_mapping_index]
        self.f1 = f1_score(
            self.true_labels,
            cluster_mappings[best_accuracy_mapping_index],
            average="macro",
        )

        return (self.accuracy, self.f1)

    def __str__(self):
        return (
            f"ExpectedMaximizer:\n"
            f"- n_clusters: {self.n_clusters}\n"
            f"- metric: {self.metric}\n"
            f"- centroids: {self.centroids}\n"
            f"- inertia: {self.inertia}\n"
            f"- cluster_sizes: {self.cluster_sizes}\n"
            f"- accuracy: {self.accuracy}\n"
            f"- f1: {self.f1}\n"
            f"- Fit time: {self.execution_time} seconds"
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

    n_clusters = 2
    metric = "euclidean"
    num_iterations = 50

    expected_maximizer = ExpectedMaximizer(
        X=auction_train_X,
        true_labels=auction_train_y.values.astype(int).flatten(),
        n_clusters=n_clusters,
        metric=metric,
    )
    expected_maximizer.get_expected_maximization(num_iterations)
    inertia = expected_maximizer.compute_inertia()
    cluster_sizes = expected_maximizer.get_cluster_sizes()
    (accuracy, f1) = expected_maximizer.get_accuracy_and_f1_score()

    print(expected_maximizer)
