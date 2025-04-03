"""
Unit tests for verifying the modularized package imports.

This module ensures that:
1. `data_ingestion` can be imported successfully.
2. `model_training` can be imported successfully.
3. `model_evaluation` can be imported successfully.

Author: Hemanth D
"""


def test_import_data_ingestion():
    """Test if `data_ingestion` module can be imported successfully."""
    try:
        from mle_training import data_ingestion  # noqa: F401
    except ImportError as e:
        assert False, f"Importing data_ingestion failed: {e}"


def test_import_model_training():
    """Test if `model_training` module can be imported successfully."""
    try:
        from mle_training import model_training  # noqa: F401
    except ImportError as e:
        assert False, f"Importing model_training failed: {e}"


def test_import_model_evaluation():
    """Test if `model_evaluation` module can be imported successfully."""
    try:
        from mle_training import model_evaluation  # noqa: F401
    except ImportError as e:
        assert False, f"Importing model_evaluation failed: {e}"
