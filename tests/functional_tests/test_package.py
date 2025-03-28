import importlib


def test_package_import():
    try:
        importlib.import_module("mle_training")
    except ImportError:
        assert False, "Package installation failed"
