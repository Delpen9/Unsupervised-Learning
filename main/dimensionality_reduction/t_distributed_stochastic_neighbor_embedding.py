import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_preprocessing import (
    preprocess_datasets,
    convert_pandas_to_dataloader,
)


class TSNEReduction:
    def __init__(self, n_components=2, random_state=None, perplexity=30, n_iter=300):
        self.n_components = n_components
        self.tsne = TSNE(
            n_components=self.n_components,
            random_state=random_state,
            perplexity=perplexity,
            n_iter=n_iter,
            method='exact' if n_components >= 4 else 'barnes_hut',
        )

    def fit_transform(self, data):
        return self.tsne.fit_transform(data)

    def plot_components(self, data, labels=None):
        transformed_data = self.fit_transform(data)

        plt.figure(figsize=(10, 8))

        if labels is not None:
            plt.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                c=labels,
                cmap="jet",
                edgecolor="k",
            )
            plt.colorbar()
        else:
            plt.scatter(transformed_data[:, 0], transformed_data[:, 1], edgecolor="k")

        plt.title("t-SNE Reduction")
        plt.show()


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
    rp_transformer = TSNEReduction(n_components=n_components, random_state=42)
    transformed_data = rp_transformer.fit_transform(auction_train_X.to_numpy())

    print(transformed_data)
