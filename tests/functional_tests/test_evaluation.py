import numpy as np
from sklearn.linear_model import LinearRegression

from mle_training.model_evaluation import evaluate_model


def test_evaluate_model():
    """Test if model evaluation returns valid values for MSE, RMSE, MAE, and R2 Score."""
    # Generate random data
    X = np.random.rand(100, 2)
    y = np.random.rand(100)

    # Train the model
    model = LinearRegression().fit(X, y)

    # Evaluate the model
    evaluation_metrics = evaluate_model(model, X, y)

    # Check that the returned dictionary contains valid metrics
    assert "MSE" in evaluation_metrics, "MSE is missing in the evaluation result"
    assert "RMSE" in evaluation_metrics, "RMSE is missing in the evaluation result"
    assert "MAE" in evaluation_metrics, "MAE is missing in the evaluation result"
    assert (
        "R2 Score" in evaluation_metrics
    ), "R2 Score is missing in the evaluation result"

    # Check that MSE, RMSE, MAE, and R2 are non-negative (RMSE should be >= 0)
    assert (
        evaluation_metrics["MSE"] >= 0
    ), f"MSE should be non-negative, but got {evaluation_metrics['MSE']}"
    assert (
        evaluation_metrics["RMSE"] >= 0
    ), f"RMSE should be non-negative, but got {evaluation_metrics['RMSE']}"
    assert (
        evaluation_metrics["MAE"] >= 0
    ), f"MAE should be non-negative, but got {evaluation_metrics['MAE']}"

    # Check that R2 Score is a valid value (it can be negative for bad models)
    assert isinstance(
        evaluation_metrics["R2 Score"], (int, float)
    ), f"R2 Score should be a number, but got {evaluation_metrics['R2 Score']}"
