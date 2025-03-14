from mle_training.data_ingestion import fetch_housing_data, load_housing_data
from mle_training.data_preprocessing import preprocess_data
from mle_training.model_evaluation import evaluate_model
from mle_training.model_training import hyperparameter_tuning, train_models

# Step 1: Fetch & Load Data
fetch_housing_data()
housing = load_housing_data()

# Step 2: Pre-processing
train_prepared, test_prepared, train_labels, test_labels = preprocess_data(housing)

# Step 4: Train Models
models = train_models(train_prepared, train_labels)

# Step 5: Hyperparameter Tuning
best_model = hyperparameter_tuning(train_prepared, train_labels)

# Step 6: Evaluate Model
rmse = evaluate_model(best_model, test_prepared, test_labels)
print(f"Final RMSE: {rmse}")
