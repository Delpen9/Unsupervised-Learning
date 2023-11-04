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

import time

def _timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        print(f"{func.__name__} executed in {execution_time} seconds")
        return result

    return wrapper

class TSNEReduction:
    def __init__(self, n_components=2, random_state=None, perplexity=30, n_iter=300):
        self.n_components = n_components
        self.tsne = TSNE(
            n_components=self.n_components,
            random_state=random_state,
            perplexity=perplexity,
            n_iter=n_iter,
            method="exact" if n_components >= 4 else "barnes_hut",
        )

    @_timer
    def fit_transform(self, data):
        return self.tsne.fit_transform(data)

    @_timer
    def kl_divergence(self):
        return self.tsne.kl_divergence_


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
    tsne_transformer = TSNEReduction(n_components=n_components, random_state=42)
    transformed_data = tsne_transformer.fit_transform(auction_train_X.to_numpy())

    print(transformed_data)
