import argparse

import joblib
import numpy as np
from config import (
    RANDOM_SEED,
    RF_CV_FOLDS,
    RF_MAX_FEATURES_HIGH,
    RF_MAX_FEATURES_LOW,
    RF_N_ESTIMATORS_HIGH,
    RF_N_ESTIMATORS_LOW,
    RF_N_ITER,
    SCORING_METRIC,
    TEST_SIZE,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def load_data(data_path):
    """
    Loads dataset from the given file path.

    Parameters
    ----------
    data_path : str
        Path to the dataset file.

    Returns
    -------
    tuple
        A tuple containing features (X) and target values (y).
    """
    data = joblib.load(data_path)
    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y


def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model using RandomizedSearchCV.

    Parameters
    ----------
    X_train : numpy.ndarray or pandas.DataFrame
        Training feature matrix.
    y_train : numpy.ndarray or pandas.Series
        Target values for training.

    Returns
    -------
    object
        The trained RandomForestRegressor model.
    """
    param_dist = {
        "n_estimators": np.arange(RF_N_ESTIMATORS_LOW, RF_N_ESTIMATORS_HIGH),
        "max_features": np.arange(RF_MAX_FEATURES_LOW, RF_MAX_FEATURES_HIGH),
    }

    model = RandomForestRegressor(random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=RF_N_ITER,
        cv=RF_CV_FOLDS,
        scoring=SCORING_METRIC,
        random_state=RANDOM_SEED,
    )

    search.fit(X_train, y_train)
    return search.best_estimator_


def main(data_path, model_output_path):
    """
    Main function to load data, train a model, and save it.

    Parameters
    ----------
    data_path : str
        Path to the dataset file.
    model_output_path : str
        Path to save the trained model.
    """
    X, y = load_data(data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model = train_model(X_train, y_train)
    joblib.dump(model, model_output_path)
    print(f"Model saved at {model_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest model.")
    parser.add_argument("data_path", type=str, help="Path to the dataset file")
    parser.add_argument(
        "model_output_path", type=str, help="Path to save the trained model"
    )

    args = parser.parse_args()
    main(args.data_path, args.model_output_path)
