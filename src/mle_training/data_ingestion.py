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
        # Convert to DataFrame to make it easier to reference by column names
        X_df = pd.DataFrame(X, columns=num_attribs)

        # Feature engineering
        rooms_per_household = X_df["total_rooms"] / X_df["households"]
        population_per_household = X_df["population"] / X_df["households"]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X_df["total_bedrooms"] / X_df["total_rooms"]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def preprocess_data(housing):
    # Add the 'income_cat' column required for stratification
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Stratified shuffle split for train/test data
    if housing["income_cat"].value_counts().min() < 2:
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            train_set = housing.loc[train_index]
            test_set = housing.loc[test_index]

    # Drop the 'income_cat' column
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Split the dataset into features and labels
    housing_labels = train_set["median_house_value"].copy()
    housing = train_set.drop("median_house_value", axis=1)

    # Define numerical and categorical attributes
    global num_attribs
    num_attribs = list(
        housing.drop("ocean_proximity", axis=1)
    )  # Excluding the categorical column
    cat_attribs = ["ocean_proximity"]

    # Create numerical and categorical pipelines
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

    # Combine both pipelines using a column transformer
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ]
    )

    # Preprocess the training data
    housing_prepared_array = full_pipeline.fit_transform(housing)

    # Convert the result back to DataFrame with appropriate column names
    # The columns for numerical features are generated dynamically
    room_feature_names = [
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_room" if "bedrooms_per_room" in num_attribs else "",
    ]
    room_feature_names = [
        name for name in room_feature_names if name
    ]  # Remove empty names
    cat_columns = (
        full_pipeline.transformers_[1][1]
        .named_steps["onehot"]
        .get_feature_names_out(input_features=cat_attribs)
    )

    # Create DataFrame with correct column names
    housing_prepared = pd.DataFrame(
        housing_prepared_array,
        columns=num_attribs + room_feature_names + list(cat_columns),
    )

    # Prepare the test data
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)

    return housing_prepared, X_test_prepared, housing_labels, y_test
