import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data Science Libraries
import numpy as np
import pandas as pd

# SKlearn Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_datasets(
    auction_dataset_location="../../data/Auction Verification/data.csv",
    dropout_dataset_location="../../data/Dropout/data.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auction_dataset = pd.read_csv(auction_dataset_location)
    dropout_dataset = pd.read_csv(dropout_dataset_location, delimiter=";")
    dropout_dataset["Target"] = dropout_dataset["Target"].replace(
        {"Graduate": 0, "Dropout": 1, "Enrolled": 2}
    )

    return (auction_dataset, dropout_dataset)


def categorize_fields() -> (
    tuple[pd.DataFrame, pd.DataFrame, list, list, list, list, list, list]
):
    (auction_dataset, dropout_dataset) = load_datasets()

    # Auction columns
    auction_categorical = []
    auction_continuous = []
    auction_targets = ["verification.result", "verification.time"]
    auction_discrete = list(
        set(auction_dataset.columns.tolist())
        - set(auction_categorical)
        - set(auction_continuous)
        - set(auction_targets)
    )

    # Dropout Columns
    dropout_categorical = [
        "Marital status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ]
    dropout_continuous = [
        "Previous qualification (grade)",
        "Admission grade",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    dropout_targets = ["Target"]
    dropout_discrete = list(
        set(dropout_dataset.columns.tolist())
        - set(dropout_categorical)
        - set(dropout_continuous)
        - set(dropout_targets)
    )

    return (
        # Dataframes
        auction_dataset,
        dropout_dataset,
        # Continuous list
        auction_continuous,
        dropout_continuous,
        # Categorical list
        auction_categorical,
        dropout_categorical,
        # Discrete list
        auction_discrete,
        dropout_discrete,
    )


def onehot_encode_categorical() -> (
    tuple[pd.DataFrame, pd.DataFrame, list, list, list, list]
):
    (
        # Dataframes
        auction_dataset,
        dropout_dataset,
        # Continuous list
        auction_continuous,
        dropout_continuous,
        # Categorical list
        auction_categorical,
        dropout_categorical,
        # Discrete list
        auction_discrete,
        dropout_discrete,
    ) = categorize_fields()

    auction_dataset = pd.get_dummies(auction_dataset, columns=auction_categorical)
    dropout_dataset = pd.get_dummies(dropout_dataset, columns=dropout_categorical)

    return (
        # Dataframes
        auction_dataset,
        dropout_dataset,
        # Continuous list
        auction_continuous,
        dropout_continuous,
        # Discrete list
        auction_discrete,
        dropout_discrete,
    )


def cluster_continuous_features() -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    (
        # Dataframes
        auction_dataset,
        dropout_dataset,
        # Continuous list
        auction_continuous,
        dropout_continuous,
        # Discrete list
        auction_discrete,
        dropout_discrete,
    ) = onehot_encode_categorical()

    def find_optimal_k(feature_values):
        distortions = []
        K = range(1, 20)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(feature_values.reshape(-1, 1))
            distortions.append(kmeanModel.inertia_)

        second_derivative = np.diff(distortions, 2)

        optimal_k = second_derivative.argmax() + 2
        return optimal_k

    for continuous_feature in auction_continuous:
        optimal_k = find_optimal_k(auction_dataset[continuous_feature].values)

        kmeans = KMeans(n_clusters=optimal_k)

        auction_dataset[continuous_feature] = kmeans.fit_predict(
            auction_dataset[[continuous_feature]]
        )

    for continuous_feature in dropout_continuous:
        optimal_k = find_optimal_k(dropout_dataset[continuous_feature].values)

        kmeans = KMeans(n_clusters=optimal_k)

        dropout_dataset[continuous_feature] = kmeans.fit_predict(
            dropout_dataset[[continuous_feature]]
        )

    auction_discrete += auction_continuous
    dropout_discrete += dropout_continuous

    return (
        # Dataframes
        auction_dataset,
        dropout_dataset,
        # Discrete list
        auction_discrete,
        dropout_discrete,
    )


def train_test_split_normalize_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    (
        # Dataframes
        auction_dataset,
        dropout_dataset,
        # Discrete list
        auction_discrete,
        dropout_discrete,
    ) = cluster_continuous_features()

    auction_temp_df, auction_test_df = train_test_split(
        auction_dataset, test_size=0.2, random_state=42
    )
    auction_train_df, auction_val_df = train_test_split(
        auction_temp_df, test_size=0.2, random_state=42
    )

    dropout_temp_df, dropout_test_df = train_test_split(
        dropout_dataset, test_size=0.2, random_state=42
    )
    dropout_train_df, dropout_val_df = train_test_split(
        dropout_temp_df, test_size=0.2, random_state=42
    )

    # Auction dataset normalization
    auction_scaler = StandardScaler()

    auction_scaler.fit(auction_train_df[auction_discrete])

    auction_train_df[auction_discrete] = auction_scaler.transform(
        auction_train_df[auction_discrete]
    )
    auction_val_df[auction_discrete] = auction_scaler.transform(
        auction_val_df[auction_discrete]
    )
    auction_test_df[auction_discrete] = auction_scaler.transform(
        auction_test_df[auction_discrete]
    )

    # Dropout dataset normalization
    dropout_scaler = StandardScaler()

    dropout_scaler.fit(dropout_train_df[dropout_discrete])

    dropout_train_df[dropout_discrete] = dropout_scaler.transform(
        dropout_train_df[dropout_discrete]
    )
    dropout_val_df[dropout_discrete] = dropout_scaler.transform(
        dropout_val_df[dropout_discrete]
    )
    dropout_test_df[dropout_discrete] = dropout_scaler.transform(
        dropout_test_df[dropout_discrete]
    )

    return (
        auction_train_df,
        auction_val_df,
        auction_test_df,
        dropout_train_df,
        dropout_val_df,
        dropout_test_df,
    )


def preprocess_datasets() -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
):
    (
        auction_train_df,
        auction_val_df,
        auction_test_df,
        dropout_train_df,
        dropout_val_df,
        dropout_test_df,
    ) = train_test_split_normalize_features()

    # Auction dataset
    auction_targets = ["verification.result", "verification.time"]

    auction_train_X = auction_train_df.drop(auction_targets, axis=1).astype(float)
    auction_train_y = auction_train_df[auction_targets].astype(float)

    auction_val_X = auction_val_df.drop(auction_targets, axis=1).astype(float)
    auction_val_y = auction_val_df[auction_targets].astype(float)

    auction_test_X = auction_test_df.drop(auction_targets, axis=1).astype(float)
    auction_test_y = auction_test_df[auction_targets].astype(float)

    # Dropout dataset
    dropout_targets = ["Target"]
    dropout_train_X = dropout_train_df.drop(dropout_targets, axis=1).astype(float)
    dropout_train_y = dropout_train_df[dropout_targets].astype(float)

    dropout_val_X = dropout_val_df.drop(dropout_targets, axis=1).astype(float)
    dropout_val_y = dropout_val_df[dropout_targets].astype(float)

    dropout_test_X = dropout_test_df.drop(dropout_targets, axis=1).astype(float)
    dropout_test_y = dropout_test_df[dropout_targets].astype(float)

    return (
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
    )


def convert_pandas_to_dataloader(
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
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    # Auction DataLoaders
    auction_train_dataset = TensorDataset(
        torch.tensor(auction_train_X.values, dtype=torch.float32),
        torch.tensor(auction_train_y.values, dtype=torch.float32),
    )

    auction_val_dataset = TensorDataset(
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
    auction_val_loader = DataLoader(auction_val_dataset, batch_size=32)
    auction_test_loader = DataLoader(auction_test_dataset, batch_size=32)

    # Dropout DataLoaders
    dropout_train_dataset = TensorDataset(
        torch.tensor(dropout_train_X.values, dtype=torch.float32),
        torch.tensor(dropout_train_y.values, dtype=torch.float32),
    )

    dropout_val_dataset = TensorDataset(
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
    dropout_val_loader = DataLoader(dropout_val_dataset, batch_size=32)
    dropout_test_loader = DataLoader(dropout_test_dataset, batch_size=32)

    return (
        auction_train_loader,
        auction_val_loader,
        auction_test_loader,
        dropout_train_loader,
        dropout_val_loader,
        dropout_test_loader,
    )
