"""
Logging configuration.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = __name__,
    log_file: Path = None,
    level: int = logging.INFO,
    format_str: str = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_str: Log format string

    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    formatter = logging.Formatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
