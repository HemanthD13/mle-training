import numpy as np
import pytest

from mle_training.model_training import hyperparameter_tuning, train_models


# Fixture to generate random data for training
@pytest.fixture
def generate_data():
    """Fixture to generate random data for training."""
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    return X, y


def test_train_linear_regression(generate_data):
    """Test if Linear Regression model trains successfully."""
    X, y = generate_data
    model = train_models(X, y)["linear_regression"]
    assert model is not None, "Linear regression training failed!"
    assert hasattr(
        model, "predict"
    ), "Trained model does not have the 'predict' method!"


def test_train_decision_tree(generate_data):
    """Test if Decision Tree model trains successfully."""
    X, y = generate_data
    model = train_models(X, y)["decision_tree"]
    assert model is not None, "Decision tree training failed!"
    assert hasattr(
        model, "predict"
    ), "Trained model does not have the 'predict' method!"


def test_train_random_forest(generate_data):
    """Test if Random Forest model trains successfully."""
    X, y = generate_data
    model = train_models(X, y)["random_forest"]
    assert model is not None, "Random forest training failed!"
    assert hasattr(
        model, "predict"
    ), "Trained model does not have the 'predict' method!"


def test_hyperparameter_tuning(generate_data):
    """Test if hyperparameter tuning returns a valid model."""
    X, y = generate_data
    best_forest = hyperparameter_tuning(X, y)
    assert best_forest is not None, "Hyperparameter tuning failed!"
    assert hasattr(
        best_forest, "predict"
    ), "Tuned model does not have the 'predict' method!"
