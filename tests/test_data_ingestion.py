import os

from mle_training.config import HOUSING_PATH
from mle_training.data_ingestion import fetch_housing_data, load_housing_data


def test_fetch_housing_data():
    """Test if the dataset is downloaded and extracted correctly."""
    fetch_housing_data()
    dataset_path = os.path.join(HOUSING_PATH, "housing.csv")
    assert os.path.exists(dataset_path), "Dataset not found!"


def test_load_housing_data():
    """Test if the dataset loads into a DataFrame."""
    df = load_housing_data()
    assert not df.empty, "Loaded dataset is empty!"
