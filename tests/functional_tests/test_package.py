import importlib


def test_package_import():
    """
    Test if the `mle_training` package can be imported successfully.

    Raises
    ------
    AssertionError
        If the package import fails.
    """
    try:
        importlib.import_module("mle_training")
    except ImportError as e:
        assert False, f"Package installation failed: {e}"


def test_import_submodules():
    """
    Test if submodules within `mle_training` can be imported successfully.

    Raises
    ------
    AssertionError
        If any submodule import fails.
    """
    submodules = [
        "mle_training.data_ingestion",
        "mle_training.model_training",
        "mle_training.model_evaluation",
    ]

    for submodule in submodules:
        try:
            importlib.import_module(submodule)
        except ImportError as e:
            assert False, f"Importing {submodule} failed: {e}"
