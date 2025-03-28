import argparse
import os

import joblib
import pandas as pd

from mle_training.model_training import train_models


def train_model(data_folder, model_folder):
    """Loads dataset, trains models, and saves them."""
    os.makedirs(model_folder, exist_ok=True)

    train_path = os.path.join(data_folder, "train.csv")
    train_label_path = os.path.join(data_folder, "x_label.csv")
    X_train = pd.read_csv(train_path)
    y_train = pd.read_csv(train_label_path)

    # X_train = preprocess_data(
    #     train_data.drop("median_house_value", axis=1)
    # )
    # y_train = train_data["median_house_value"]

    models = {
        "linear_regression": train_models(X_train, y_train)["linear_regression"],
        "decision_tree": train_models(X_train, y_train)["decision_tree"],
        "random_forest": train_models(X_train, y_train)["random_forest"],
    }

    for name, model in models.items():
        model_path = os.path.join(model_folder, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models.")
    parser.add_argument("data_folder", type=str, help="Folder containing train.csv")
    parser.add_argument("model_folder", type=str, help="Folder to save trained models")
    args = parser.parse_args()

    train_model(args.data_folder, args.model_folder)
