"""
Unit tests for data ingestion functions.

This module contains tests for:
1. Fetching the housing dataset.
2. Loading the dataset into a Pandas DataFrame.

"""

import os

from mle_training.config import HOUSING_PATH
from mle_training.data_ingestion import fetch_housing_data, load_housing_data


def test_fetch_housing_data():
    """
    Test if the dataset is downloaded and extracted correctly.

    This function verifies whether the housing dataset exists in the expected
    directory after calling `fetch_housing_data()`.

    Raises
    ------
    AssertionError
        If the dataset file does not exist after fetching.
    """
    fetch_housing_data()
    dataset_path = os.path.join(HOUSING_PATH, "housing.csv")
    assert os.path.exists(dataset_path), "Dataset not found!"


def test_load_housing_data():
    """
    Test if the dataset loads into a Pandas DataFrame.

    This function ensures that the dataset loads successfully into a DataFrame
    and is not empty.

    Raises
    ------
    AssertionError
        If the loaded dataset is empty.
    """
    df = load_housing_data()
    assert not df.empty, "Loaded dataset is empty!"
