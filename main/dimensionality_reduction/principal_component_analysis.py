import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)

class PCADimensionalityReduction:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, data):
        self.pca.fit(data)

    def transform(self, data):
        return self.pca.transform(data)

    def get_explained_variance(self):
        return np.array(self.pca.explained_variance_ratio_).sum()


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
    pca_transformer = PCADimensionalityReduction(n_components=n_components)
    pca_transformer.fit(auction_train_X.to_numpy())
    transformed_data = pca_transformer.transform(auction_train_X.to_numpy())

    print(transformed_data)
