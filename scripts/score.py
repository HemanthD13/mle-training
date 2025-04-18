import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model(model_path):
    """
    Loads a trained model from a given file path.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.

    Returns
    -------
    object
        The loaded machine learning model.
    """
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model using various regression metrics.

    Parameters
    ----------
    model : object
        The trained model to be evaluated.
    X_test : numpy.ndarray or pandas.DataFrame
        Feature matrix used for testing.
    y_test : numpy.ndarray or pandas.Series
        True labels corresponding to `X_test`.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - "MSE" : Mean Squared Error
        - "RMSE" : Root Mean Squared Error
        - "MAE" : Mean Absolute Error
        - "R2 Score" : R-squared score
    """
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2 Score": r2}


def main(model_path, X_test_path, y_test_path):
    """
    Main function to load the model and test data, evaluate performance, and results.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.
    X_test_path : str
        Path to the test features file.
    y_test_path : str
        Path to the test labels file.
    """
    model = load_model(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    results = evaluate_model(model, X_test, y_test)
    print("Evaluation Results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("X_test_path", type=str, help="Path to the test features file")
    parser.add_argument("y_test_path", type=str, help="Path to the test labels file")

    args = parser.parse_args()
    main(args.model_path, args.X_test_path, args.y_test_path)
