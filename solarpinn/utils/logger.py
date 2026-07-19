"""
Logging utilities for SolarPINN.
"""

from __future__ import annotations

import logging
from pathlib import Path

from solarpinn.config import LOG_DIR, logging as log_config


def get_logger(
        name: str,
        level: str | int | None = None
)-> logging.Logger:
    """ 
    Create and configure a logger.

    Parameters
    ----------
    name: str
        Name of the logger.
    
    level: str | int | None, optional
        Logging level. If None, uses the project default.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    log_level = level or log_config.level
    logger.setLevel(log_level)


    LOG_DIR.mkdir(parents = True, exist_ok=True)

    formatter = logging.Formatter(
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    #Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #File handler
    if log_config.log_to_file:
        file_handler = logging.FileHandler(
            Path(LOG_DIR) / log_config.log_filename, encoding ="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
