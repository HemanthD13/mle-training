"""
Main script for training and evaluating machine learning models.

This script performs the following steps:
1. Fetches and loads the housing dataset.
2. Preprocesses the data.
3. Trains multiple models.
4. Performs hyperparameter tuning.
5. Evaluates the best model.

"""

from mle_training.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
    preprocess_data,
)
from mle_training.model_evaluation import evaluate_model
from mle_training.model_training import hyperparameter_tuning, train_models


def main():
    """Executes the end-to-end machine learning pipeline."""
    # Step 1: Fetch & Load Data
    fetch_housing_data()
    housing = load_housing_data()

    # Step 2: Preprocessing
    train_prepared, test_prepared, train_labels, test_labels = preprocess_data(housing)

    # Step 3: Train Model
    models = train_models(train_prepared, train_labels)

    # Step 4: Hyperparameter Tuning
    best_model = hyperparameter_tuning(train_prepared, train_labels)

    # Step 5: Evaluate Model
    evaluation_metrics = evaluate_model(best_model, test_prepared, test_labels)
    print(f"Final Evaluation Metrics: {evaluation_metrics}")


if __name__ == "__main__":
    main()
