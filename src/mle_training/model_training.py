from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def train_models(X, y):
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
    param_distribs = {"n_estimators": randint(1, 200), "max_features": randint(1, 8)}
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
