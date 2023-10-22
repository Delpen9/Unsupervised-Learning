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

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)

from expected_maximization import (
    ExpectedMaximizer,
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
    # (accuracy, f1) = expected_maximizer.get_accuracy_and_f1_score()

    print(expected_maximizer)
