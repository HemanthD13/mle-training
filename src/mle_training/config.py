# src/mle_training/config.py

import os

# Dataset Configuration
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
METRIC = "rmse"
IMPUTATION_STRATEGY = "median"  # Options: "mean", "median", "most_frequent"
ADD_CUSTOM_FEATURES = True  # Enable/Disable custom feature engineering
ONE_HOT_ENCODING = True  # Enable/Disable one-hot encoding for categorical


# src/mle_training/config.py

# Training Configuration
RANDOM_SEED = 42  # Ensures reproducibility
TEST_SIZE = 0.2  # 20% test data

# Random Forest Hyperparameters
RF_N_ESTIMATORS_LOW = 1
RF_N_ESTIMATORS_HIGH = 200
RF_MAX_FEATURES_LOW = 1
RF_MAX_FEATURES_HIGH = 8

# Number of iterations for RandomizedSearchCV
RF_N_ITER = 10
RF_CV_FOLDS = 5  # Cross-validation folds
SCORING_METRIC = "neg_mean_squared_error"  # Optimization metric
