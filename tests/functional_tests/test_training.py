import numpy as np
import pytest

from mle_training.model_training import hyperparameter_tuning, train_models


# Fixture to generate random data for training
@pytest.fixture
def generate_data():
    """
    Fixture to generate random data for training.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Array of shape (100, 2) with random values as input
              features.
            - y (np.ndarray): Array of shape (100,) with random values as target
              values.
    """
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    return X, y


def test_train_linear_regression(generate_data):
    """
    Test if the Linear Regression model trains successfully.

    Args:
        generate_data (tuple): Fixture that generates random training data.

    Raises:
        AssertionError: If the Linear Regression model does not train or lacks
                        the 'predict' method.
    """
    X, y = generate_data
    model = train_models(X, y)["linear_regression"]
    assert model is not None, "Linear regression training failed!"
    assert hasattr(
        model, "predict"
    ), "Trained model does not have the 'predict' method!"


def test_train_decision_tree(generate_data):
    """
    Test if the Decision Tree model trains successfully.

    Args:
        generate_data (tuple): Fixture that generates random training data.

    Raises:
        AssertionError: If the Decision Tree model does not train or lacks the
                        'predict' method.
    """
    X, y = generate_data
    model = train_models(X, y)["decision_tree"]
    assert model is not None, "Decision tree training failed!"
    assert hasattr(
        model, "predict"
    ), "Trained model does not have the 'predict' method!"


def test_train_random_forest(generate_data):
    """
    Test if the Random Forest model trains successfully.

    Args:
        generate_data (tuple): Fixture that generates random training data.

    Raises:
        AssertionError: If the Random Forest model does not train or lacks the
                        'predict' method.
    """
    X, y = generate_data
    model = train_models(X, y)["random_forest"]
    assert model is not None, "Random forest training failed!"
    assert hasattr(
        model, "predict"
    ), "Trained model does not have the 'predict' method!"


def test_hyperparameter_tuning(generate_data):
    """
    Test if hyperparameter tuning returns a valid model.

    Args:
        generate_data (tuple): Fixture that generates random training data.

    Raises:
        AssertionError: If hyperparameter tuning fails or the tuned model lacks
                        the 'predict' method.
    """
    X, y = generate_data
    best_forest = hyperparameter_tuning(X, y)
    assert best_forest is not None, "Hyperparameter tuning failed!"
    assert hasattr(
        best_forest, "predict"
    ), "Tuned model does not have the 'predict' method!"
