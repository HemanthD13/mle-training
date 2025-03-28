import pandas as pd

from mle_training.data_ingestion import preprocess_data


def test_preprocess_housing_data():
    """Test feature engineering in preprocess_housing_data function."""

    # Sample data simulating the housing dataset
    data = pd.DataFrame(
        {
            "total_rooms": [1000, 2000, 1500, 1800],
            "households": [500, 1000, 750, 900],
            "total_bedrooms": [200, 400, 300, 350],
            "population": [1000, 2000, 1500, 1800],
            "median_income": [3.0, 4.5, 3.5, 4.0],
            "ocean_proximity": ["NEAR BAY", "NEAR OCEAN", "NEAR BAY", "NEAR OCEAN"],
            "longitude": [
                -122.23,
                -122.25,
                -122.22,
                -122.24,
            ],  # Example longitude values
            "latitude": [37.88, 37.85, 37.86, 37.87],  # Example latitude values
            "median_house_value": [
                200000,
                300000,
                250000,
                275000,
            ],  # Added median house value (target)
        }
    )

    # Create the 'income_cat' column required for stratification
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5],
    )

    # Call preprocess_data which returns multiple datasets
    housing_prepared, X_test_prepared, housing_labels, y_test = preprocess_data(data)

    # Check that new features have been created
    assert (
        "rooms_per_household" in housing_prepared.columns
    ), "Feature 'rooms_per_household' missing!"
    assert (
        "bedrooms_per_room" in housing_prepared.columns
    ), "Feature 'bedrooms_per_room' missing!"
    assert (
        "population_per_household" in housing_prepared.columns
    ), "Feature 'population_per_household' missing!"

    # Optionally, you can add assertions to check if features are correctly calculated for the first row
    assert (
        housing_prepared["rooms_per_household"][0] == 2.0
    ), "Incorrect 'rooms_per_household' calculation!"
    assert (
        housing_prepared["bedrooms_per_room"][0] == 0.2
    ), "Incorrect 'bedrooms_per_room' calculation!"
    assert (
        housing_prepared["population_per_household"][0] == 2.0
    ), "Incorrect 'population_per_household' calculation!"
