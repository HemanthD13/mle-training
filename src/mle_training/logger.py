import logging
import sys


def setup_logger(log_level="INFO", log_path=None, no_console_log=False):
    """Configures the logger with user-specified options."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    handlers = []

    if log_path:
        handlers.append(logging.FileHandler(log_path))
    if not no_console_log:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
    )

    if not handlers:
        raise ValueError(
            "No valid logging handlers available. \
                         Check your log settings."
        )
