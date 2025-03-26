import argparse
import os

import joblib
import pandas as pd

from mle_training.model_evaluation import evaluate_model


def score_models(model_folder, data_folder):
    """Scores trained models on validation dataset."""
    val_path = os.path.join(data_folder, "val.csv")
    val_label_path = os.path.join(data_folder, "val_label.csv")
    X_val = pd.read_csv(val_path)
    y_val = pd.read_csv(val_label_path)
    # X_val = preprocess_data(
    #     val_data.drop("median_house_value", axis=1)
    # )
    # y_val = val_data["median_house_value"]

    for model_name in [
        "linear_regression",
        "decision_tree",
        "random_forest",
    ]:
        model_path = os.path.join(model_folder, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found, skipping.")
            continue

        model = joblib.load(model_path)
        rmse = evaluate_model(model, X_val, y_val)
        print(f"{model_name} RMSE: {rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score ML models.")
    parser.add_argument(
        "model_folder", type=str, help="Folder containing trained models"
    )
    parser.add_argument("data_folder", type=str, help="Folder containing val.csv")
    args = parser.parse_args()

    score_models(args.model_folder, args.data_folder)
