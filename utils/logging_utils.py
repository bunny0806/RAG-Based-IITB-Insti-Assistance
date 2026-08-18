"""Centralized logging configuration for the project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import LOGS_DIR


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a project logger.

    Args:
        log_file: Optional name for the log file. If omitted, a default file is used.
        level: Logging severity level.

    Returns:
        Configured logger instance.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    file_name = log_file or "app.log"
    log_path = LOGS_DIR / file_name

    logger = logging.getLogger("iitb_insti_assist")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
