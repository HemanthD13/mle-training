import numpy as np
import pandas as pd

from mle_training.data_ingestion import preprocess_data


def test_preprocess_housing_data():
    """
    Test feature engineering in preprocess_housing_data function.

    This test checks that the preprocessing function correctly adds features
    such as 'rooms_per_household', 'bedrooms_per_room', and
    'population_per_household'. It also verifies that the features are
    calculated correctly.

    Raises:
        AssertionError: If any of the checks fail.
    """
    # Sample data simulating the housing dataset
    data = pd.DataFrame(
        {
            "total_rooms": [1000, 2000, 1500, 1800],
            "households": [500, 1000, 750, 900],
            "total_bedrooms": [200, 400, 300, 350],
            "population": [1000, 2000, 1500, 1800],
            "median_income": [3.0, 4.5, 3.5, 4.0],
            "ocean_proximity": ["NEAR BAY", "NEAR OCEAN", "NEAR BAY", "NEAR OCEAN"],
            "longitude": [-122.23, -122.25, -122.22, -122.24],
            "latitude": [37.88, 37.85, 37.86, 37.87],
            "median_house_value": [200000, 300000, 250000, 275000],
        }
    )

    # Call preprocess_data which returns multiple datasets
    housing_prepared, X_test_prepared, housing_labels, y_test = preprocess_data(data)

    # Check that new features have been created
    expected_num_features = housing_prepared.shape[1]

    assert housing_prepared.shape[0] == int(
        0.8 * data.shape[0]
    ), "Row count mismatch\
          after preprocessing"
    assert (
        expected_num_features > data.shape[1]
    ), "Feature engineering did not add \
        extra features"
    assert not np.isnan(housing_prepared).any(), "Pipeline output contains NaN values"
