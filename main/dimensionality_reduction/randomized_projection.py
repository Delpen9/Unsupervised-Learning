import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)


class RandomProjectionDimensionalityReduction:
    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.rp = GaussianRandomProjection(
            n_components=self.n_components, random_state=random_state
        )

    def fit(self, data):
        self.rp.fit(data)

    def transform(self, data):
        return self.rp.transform(data)

    def fit_transform(self, data):
        return self.rp.fit_transform(data)

    def reconstruction_error(self, data):
        transformed = self.fit_transform(data)
        original_dim = data.shape[1]
        components = self.rp.components_
        if self.n_components is None:
            self.n_components = components.shape[0]
        reconstructed = np.dot(transformed, np.linalg.pinv(components.T))
        error = (
            np.linalg.norm(data - reconstructed, "fro") ** 2
            / np.linalg.norm(data, "fro") ** 2
        )
        return error


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

    n_components = 5
    rp_transformer = RandomProjectionDimensionalityReduction(
        n_components=n_components, random_state=42
    )
    rp_transformer.fit(auction_train_X.to_numpy())
    transformed_data = rp_transformer.transform(auction_train_X.to_numpy())

    print(transformed_data)
