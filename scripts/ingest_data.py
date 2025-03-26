import argparse
import os

from mle_training.data_ingestion import fetch_housing_data, load_housing_data
from mle_training.data_preprocessing import preprocess_data


def ingest_data(output_folder):
    """Fetches dataset and creates training/validation datasets."""
    os.makedirs(output_folder, exist_ok=True)

    fetch_housing_data()
    data = load_housing_data()

    x_train, x_val, x_label, val_label = preprocess_data(data)

    train_path = os.path.join(output_folder, "train.csv")
    val_path = os.path.join(output_folder, "val.csv")
    train_label_path = os.path.join(output_folder, "x_label.csv")
    val_label_path = os.path.join(output_folder, "val_label.csv")

    x_train.to_csv(train_path, index=False)
    x_val.to_csv(val_path, index=False)
    x_label.to_csv(train_label_path, index=False)
    val_label.to_csv(val_label_path, index=False)

    print(
        f"Training dataset saved at: {train_path} and shape of train is {x_train.shape}"
    )
    print(
        f"Training labels saved at: {train_label_path} and shape of train labels is {x_label.shape}"
    )
    print(
        f"Validation dataset saved at: {val_path} and shape of validation is {x_val.shape}"
    )
    print(
        f"Validation labels saved at: {val_label_path} and shape of validation labels is {val_label.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split dataset.")
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder to save train/val datasets",
    )
    args = parser.parse_args()

    ingest_data(args.output_folder)
