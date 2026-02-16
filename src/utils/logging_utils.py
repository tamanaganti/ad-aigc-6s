"""
Logging utilities for the multimodal ad effectiveness project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


def setup_logging(cfg: DictConfig, log_file: Optional[str] = None) -> None:
    """Setup logging configuration with console and file handlers."""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, cfg.get("log_level", "INFO").upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided or from config
    if log_file or cfg.get("log_dir"):
        log_dir = Path(cfg.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)

        if not log_file:
            log_file = log_dir / f"{cfg.get('experiment_name', 'experiment')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
