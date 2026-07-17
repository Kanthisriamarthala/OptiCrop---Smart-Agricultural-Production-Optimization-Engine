import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATASET_PATH


def load_dataset(path=None):
    if path is None:
        path = DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def handle_missing_values(df):
    return df.dropna()


def remove_duplicates(df):
    return df.drop_duplicates()


def encode_labels(df, target_column="label"):
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])
    return df, encoder


def split_data(df, target_column="label", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(path=None, test_size=0.2, random_state=42):
    df = load_dataset(path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df_encoded, encoder = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(
        df_encoded, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, encoder, df
