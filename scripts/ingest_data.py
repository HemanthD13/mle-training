import argparse
import logging
import os

from mle_training.data_ingestion import (
    fetch_housing_data,
    load_housing_data,
    preprocess_data,
)
from mle_training.logger import setup_logger


def ingest_data(output_folder):
    """
    Fetches dataset, processes it, and saves training/validation datasets.

    Parameters
    ----------
    output_folder : str
        Path to the folder where processed datasets will be saved.

    Returns
    -------
    None
    """
    logging.info("Starting data ingestion...")
    try:
        os.makedirs(output_folder, exist_ok=True)
        logging.debug(f"Output folder created/verified: {output_folder}")
    except OSError as e:
        logging.error(f"Error creating output folder: {e}")
        return

    try:
        logging.info("Fetching housing data...")
        fetch_housing_data()
        logging.debug("Housing data fetched successfully.")
    except Exception as e:
        logging.error(f"Error fetching housing data: {e}")
        return

    try:
        logging.info("Loading housing data...")
        data = load_housing_data()
        logging.debug("Housing data loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error loading housing data: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error loading housing data: {e}")
        return

    try:
        logging.info("Preprocessing data...")
        x_train, x_val, x_label, val_label = preprocess_data(data)
        logging.debug("Data preprocessing completed.")
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return

    train_path = os.path.join(output_folder, "train.csv")
    val_path = os.path.join(output_folder, "val.csv")
    train_label_path = os.path.join(output_folder, "x_label.csv")
    val_label_path = os.path.join(output_folder, "val_label.csv")

    try:
        logging.info(f"Saving training data to: {train_path}")
        x_train.to_csv(train_path, index=False)
        logging.debug(f"Training data saved successfully to: {train_path}")
    except Exception as e:
        logging.error(f"Error saving training data: {e}")
        return

    try:
        logging.info(f"Saving validation data to: {val_path}")
        x_val.to_csv(val_path, index=False)
        logging.debug(f"Validation data saved successfully to: {val_path}")
    except Exception as e:
        logging.error(f"Error saving validation data: {e}")
        return

    try:
        logging.info(f"Saving training labels to: {train_label_path}")
        x_label.to_csv(train_label_path, index=False)
        logging.debug(f"Training labels saved successfully to: {train_label_path}")
    except Exception as e:
        logging.error(f"Error saving training labels: {e}")
        return

    try:
        logging.info(f"Saving validation labels to: {val_label_path}")
        val_label.to_csv(val_label_path, index=False)
        logging.debug(f"Validation labels saved successfully to: {val_label_path}")
    except Exception as e:
        logging.error(f"Error saving validation labels: {e}")
        return

    logging.info("Data ingestion and processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split dataset.")
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder to save train/val datasets",
    )
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

    ingest_data(args.output_folder)
