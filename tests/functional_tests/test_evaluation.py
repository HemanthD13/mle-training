"""
Unit test for model evaluation.

This module tests:
1. The presence of evaluation metrics in the output.
2. The validity of the returned metric values.

"""

import numpy as np
from sklearn.linear_model import LinearRegression

from mle_training.model_evaluation import evaluate_model


def test_evaluate_model():
    """
    Test if model evaluation returns valid values for MSE, RMSE, MAE, and R2 Score.

    This function:
    - Generates synthetic random data.
    - Trains a Linear Regression model.
    - Evaluates the model using `evaluate_model`.
    - Checks if the returned dictionary contains all required metrics.
    - Ensures that metric values are valid.

    Raises
    ------
    AssertionError
        If any metric is missing or contains invalid values.
    """
    # Generate random data
    X = np.random.rand(100, 2)
    y = np.random.rand(100)

    # Train the model
    model = LinearRegression().fit(X, y)

    # Evaluate the model
    evaluation_metrics = evaluate_model(model, X, y)

    # Validate presence of evaluation metrics
    required_metrics = ["MSE", "RMSE", "MAE", "R2 Score"]
    for metric in required_metrics:
        assert metric in evaluation_metrics, f"{metric} is missing in evaluation result"

    # Validate metric values
    assert (
        evaluation_metrics["MSE"] >= 0
    ), f"MSE should be non-negative, but got {evaluation_metrics['MSE']}"
    assert (
        evaluation_metrics["RMSE"] >= 0
    ), f"RMSE should be non-negative, but got {evaluation_metrics['RMSE']}"
    assert (
        evaluation_metrics["MAE"] >= 0
    ), f"MAE should be non-negative, but got {evaluation_metrics['MAE']}"

    # Ensure R2 Score is a valid number (it can be negative for bad models)
    assert isinstance(
        evaluation_metrics["R2 Score"], (int, float)
    ), f"R2 Score should be a number, but got {evaluation_metrics['R2 Score']}"
