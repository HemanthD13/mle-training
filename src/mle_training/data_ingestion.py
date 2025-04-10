import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Custom Transformer for Feature Engineering
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def preprocess_data(housing):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    if housing["income_cat"].value_counts().min() < 2:
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            train_set = housing.loc[train_index]
            test_set = housing.loc[test_index]

    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing_labels = train_set["median_house_value"].copy()
    housing = train_set.drop("median_house_value", axis=1)

    # Define numerical and categorical attributes
    num_attribs = list(housing.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    # Create pipelines
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", FeatureAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    # Define expected columns after the transformations
    num_extra_features = 3 if FeatureAdder().add_bedrooms_per_room else 2
    expected_columns = (
        num_attribs
        + ["rooms_per_household", "population_per_household"]
        + (["bedrooms_per_room"] if FeatureAdder().add_bedrooms_per_room else [])
        + list(
            full_pipeline.transformers_[1][1]
            .named_steps["onehot"]
            .get_feature_names_out()
        )
    )

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)

    # Convert housing_prepared (numpy array) into a DataFrame with correct column names
    housing_prepared = pd.DataFrame(housing_prepared, columns=expected_columns)

    return housing_prepared, X_test_prepared, housing_labels, y_test
