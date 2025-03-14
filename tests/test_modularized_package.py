def test_import_data_ingestion():
    try:
        from mle_training import data_ingestion
    except ImportError as e:
        assert False, f"Importing data_ingestion failed: {e}"


def test_import_data_preprocessing():
    try:
        from mle_training import data_preprocessing
    except ImportError as e:
        assert False, f"Importing data_preprocessing failed: {e}"


def test_import_model_training():
    try:
        from mle_training import model_training
    except ImportError as e:
        assert False, f"Importing model_training failed: {e}"


def test_import_model_evaluation():
    try:
        from mle_training import model_evaluation
    except ImportError as e:
        assert False, f"Importing model_evaluation failed: {e}"
