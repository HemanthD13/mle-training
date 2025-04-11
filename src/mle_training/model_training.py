from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def train_models(X, y):
    """
    Trains multiple regression models on the given dataset.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix used for training.
    y : numpy.ndarray or pandas.Series
        Target values corresponding to `X`.

    Returns
    -------
    dict
        A dictionary containing trained models:
        - "linear_regression" : Trained LinearRegression model.
        - "decision_tree" : Trained DecisionTreeRegressor model.
        - "random_forest" : Trained RandomForestRegressor model.
    """
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42),
    }

    for name, model in models.items():
        model.fit(X, y)
        print(f"{name} training completed.")

    return models


def hyperparameter_tuning(X, y):
    """
    Performs hyperparameter tuning for a RandomForestRegressor model.

    Uses RandomizedSearchCV to find the best hyperparameters.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix used for training.
    y : numpy.ndarray or pandas.Series
        Target values corresponding to `X`.

    Returns
    -------
    RandomForestRegressor
        The best RandomForestRegressor model found after hyperparameter tuning.
    """
    param_distribs = {
        "n_estimators": randint(1, 200),
        "max_features": randint(1, 8),
    }

    forest_reg = RandomForestRegressor(random_state=42)

    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X, y)

    best_forest = rnd_search.best_estimator_
    return best_forest
