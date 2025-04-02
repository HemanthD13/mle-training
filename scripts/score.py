import argparse
import logging
import os

import joblib
import pandas as pd

from mle_training.logger import setup_logger
from mle_training.model_evaluation import evaluate_model


def score_models(model_folder, data_folder):
    """Scores trained models on validation dataset."""
    logging.info("Starting model scoring...")
    val_path = os.path.join(data_folder, "val.csv")
    val_label_path = os.path.join(data_folder, "val_label.csv")

    try:
        X_val = pd.read_csv(val_path)
        y_val = pd.read_csv(val_label_path)
        logging.debug(f"Validation data loaded from {val_path} and {val_label_path}")
    except FileNotFoundError as e:
        logging.error(f"Validation data file not found: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading validation data: {e}")
        return

    for model_name in [
        "linear_regression",
        "decision_tree",
        "random_forest",
    ]:
        model_path = os.path.join(model_folder, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            logging.warning(f"Model {model_name} not found, skipping.")
            continue

        try:
            model = joblib.load(model_path)
            logging.debug(f"Model {model_name} loaded from {model_path}")
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            continue
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            continue

        try:
            rmse = evaluate_model(model, X_val, y_val)
            logging.info(f"{model_name} RMSE: {rmse}")
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")
            continue
        logging.debug(f"Evaluation of {model_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score ML models.")
    parser.add_argument(
        "model_folder", type=str, help="Folder containing trained models"
    )
    parser.add_argument("data_folder", type=str, help="Folder containing val.csv")
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

    score_models(args.model_folder, args.data_folder)
