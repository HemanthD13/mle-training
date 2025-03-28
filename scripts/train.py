import argparse
import logging
import os

import joblib
import pandas as pd

from mle_training.logger import setup_logger
from mle_training.model_training import train_models


def train_model(data_folder, model_folder):
    """Loads dataset, trains models, and saves them."""
    os.makedirs(model_folder, exist_ok=True)
    logging.info("Starting model training...")

    train_path = os.path.join(data_folder, "train.csv")
    train_label_path = os.path.join(data_folder, "x_label.csv")
    try:
        X_train = pd.read_csv(train_path)
        y_train = pd.read_csv(train_label_path)
        logging.debug(
            f"Loaded training data from {train_path} and labels from {train_label_path}"
        )
    except FileNotFoundError as e:
        logging.error(f"Error loading training data: {e}")
        return  # Exit if data loading fails

    if X_train.empty or y_train.empty:
        logging.warning("Training data or labels are empty.")
        return

    try:
        models = {
            "linear_regression": train_models(X_train, y_train)["linear_regression"],
            "decision_tree": train_models(X_train, y_train)["decision_tree"],
            "random_forest": train_models(X_train, y_train)["random_forest"],
        }
        logging.debug("Models trained successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return

    for name, model in models.items():
        try:
            model_path = os.path.join(model_folder, f"{name}.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Saved {name} model to {model_path}")
            logging.debug(f"Model {name} saved successfully.")
        except Exception as e:
            logging.error(f"Error saving {name} model: {e}")
            logging.debug(f"Error details: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models.")
    parser.add_argument("data_folder", type=str, help="Folder containing train.csv")
    parser.add_argument("model_folder", type=str, help="Folder to save trained models")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="File to write logs",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )

    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)

    train_model(args.data_folder, args.model_folder)
