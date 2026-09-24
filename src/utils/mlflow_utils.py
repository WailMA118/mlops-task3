"""
Small MLflow helpers shared by training and inference code, so the tracking
URI / experiment / registry names are only ever read from config.yaml.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def configure_mlflow() -> MlflowClient:
    """Point the mlflow module at the configured tracking server and experiment."""
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    logger.info(
        "MLflow configured: tracking_uri=%s, experiment=%s",
        cfg.mlflow.tracking_uri,
        cfg.mlflow.experiment_name,
    )
    return MlflowClient()


def get_production_model_uri() -> str:
    cfg = get_config()
    return f"models:/{cfg.mlflow.registered_model_name}/{cfg.mlflow.registry_stage}"
