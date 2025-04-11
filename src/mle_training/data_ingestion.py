import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Fetches the housing dataset by downloading and extracting it from the specified URL.

    Parameters
    ----------
    housing_url : str, optional
        The URL to fetch the housing dataset from (default is the URL in `HOUSING_URL`).
    housing_path : str, optional
        The path where the dataset should be stored (default is `HOUSING_PATH`).
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Loads the housing dataset from the specified directory.

    Parameters
    ----------
    housing_path : str, optional
        The directory where the dataset is stored (default is `HOUSING_PATH`).

    Returns
    -------
    pandas.DataFrame
        The loaded housing dataset.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def preprocess_data(housing):
    """
    Preprocesses the housing data by performing tasks like feature engineering,
    handling missing values, and splitting the data into training and testing sets.

    Parameters
    ----------
    housing : pandas.DataFrame
        The raw housing data to preprocess.

    Returns
    -------
    tuple
        A tuple containing:
        - `housing_prepared` : pandas.DataFrame
            The preprocessed training data with features.
        - `X_test_prepared` : pandas.DataFrame
            The preprocessed test data with features.
        - `housing_labels` : pandas.Series
            The labels for the training set.
        - `y_test` : pandas.Series
            The labels for the test set.
    """
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # If there are fewer than 2 samples per class, use a simple train-test split
    if housing["income_cat"].value_counts().min() < 2:
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            train_set = housing.loc[train_index]
            test_set = housing.loc[test_index]

    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = train_set.copy()

    # Feature Engineering
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing_labels = train_set["median_house_value"].copy()
    housing = train_set.drop("median_house_value", axis=1)

    # Handling missing values
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    housing_tr = pd.DataFrame(
        imputer.transform(housing_num), columns=housing_num.columns, index=housing.index
    )

    # Recalculate derived feature
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    # Handling categorical variables
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    # Preprocessing test data
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = pd.DataFrame(
        imputer.transform(X_test_num), columns=X_test_num.columns, index=X_test.index
    )

    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    return housing_prepared, X_test_prepared, housing_labels, y_test
