"""
Logging configuration for the whole project.

Replaces print() everywhere. Sets up:
  - a console handler (human-readable, for local dev / container stdout)
  - a rotating file handler (for the general app log)
  - a separate rotating file handler + logger for prediction request logs,
    used by the monitoring layer (request/response/latency/model version)

Usage:
    from src.utils.logging_setup import get_logger, get_prediction_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config import get_config, resolve_path

_CONFIGURED = False
PREDICTION_LOGGER_NAME = "olist.predictions"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(force: bool = False) -> None:
    """Idempotently configure root + prediction loggers from config.yaml."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    cfg = get_config()
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    fmt = cfg.logging.format
    formatter = logging.Formatter(fmt)

    root_logger = logging.getLogger("olist")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.propagate = False

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    app_log_path = resolve_path(cfg.logging.app_log_file)
    _ensure_dir(app_log_path)
    file_handler = RotatingFileHandler(
        app_log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    # Dedicated prediction-request logger: every predict call is written here,
    # independent of the app's general log level, for monitoring/evaluation later.
    pred_logger = logging.getLogger(PREDICTION_LOGGER_NAME)
    pred_logger.setLevel(logging.INFO)
    pred_logger.handlers.clear()
    pred_logger.propagate = False

    pred_log_path = resolve_path(cfg.logging.predictions_log_file)
    _ensure_dir(pred_log_path)
    pred_file_handler = RotatingFileHandler(
        pred_log_path, maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    pred_file_handler.setFormatter(logging.Formatter("%(message)s"))
    pred_logger.addHandler(pred_file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(f"olist.{name}" if not name.startswith("olist") else name)


def get_prediction_logger() -> logging.Logger:
    """Get the dedicated prediction-log logger (one JSON line per request)."""
    configure_logging()
    return logging.getLogger(PREDICTION_LOGGER_NAME)