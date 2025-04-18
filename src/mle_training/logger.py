import logging
import sys


def setup_logger(log_level="INFO", log_path=None, no_console_log=False):
    """
    Configures the logger with user-specified options.

    Parameter
    ----------
    log_level : str, optional
        The logging level to use (default is "INFO"). Valid values are:
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    log_path : str, optional
        The file path where logs should be saved (default is None). If provided,
        logs will be written to the specified file.
    no_console_log : bool, optional
        If True, disables console logging (default is False). If False,
        logs will also be printed to the console.

    Raises
    ------
    ValueError
        If no valid logging handlers are provided (i.e., no log file or no console log).
    """
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
            "No valid logging handlers available. Check your log settings."
        )
