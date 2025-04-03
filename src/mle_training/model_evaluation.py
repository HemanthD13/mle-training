import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained model using various regression metrics.

    Parameters
    ----------
    model : object
        The trained model to be evaluated. It must have a `predict` method.
    X_test : pandas.DataFrame or numpy.ndarray
        The test features used for making predictions.
    y_test : pandas.Series or numpy.ndarray
        The true target values corresponding to `X_test`.

    Returns
    -------
    dict
        A dictionary containing the following evaluation metrics:
        - "MSE" : float
            Mean Squared Error of the predictions.
        - "RMSE" : float
            Root Mean Squared Error of the predictions.
        - "MAE" : float
            Mean Absolute Error of the predictions.
        - "R2 Score" : float
            R-squared score, which indicates the proportion of variance explained by
            the model.

    Notes
    -----
    - MSE: Measures the average squared difference between the predicted and actual
      values.
    - RMSE: The square root of MSE, which provides error in the same units as the
      target variable.
    - MAE: Measures the average absolute difference between predicted and actual
      values.
    - R2 Score: A measure of how well the model fits the data. A value of 1 indicates
      perfect fit, while a value of 0 indicates no explanatory power.
    """
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2 Score": r2}
