"""
Model loading for inference.

Task 3 requirement: "The service loads the model from the registry or the
artifact store, not from a local notebook folder." This module's primary
path is the MLflow Model Registry (registered_model_name + stage from
config.yaml). If MLflow is unreachable (e.g. running tests offline), it
falls back to the local joblib artifact saved by src/pipeline/train.py's
save_local_model_artifacts, with a loud warning — this fallback is for
local dev/tests only, not the production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc

from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    model: Any
    version: str
    source: str  # "mlflow_registry" or "local_artifact_fallback"


def load_registered_model() -> LoadedModel:
    """Load the current-stage model from the MLflow Model Registry."""
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    model_name = cfg.mlflow.registered_model_name
    stage = cfg.mlflow.registry_stage
    model_uri = f"models:/{model_name}/{stage}"

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        version = versions[0].version if versions else "unknown"

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(
            "Loaded model '%s' version %s from MLflow registry (stage=%s)",
            model_name,
            version,
            stage,
        )
        return LoadedModel(model=model, version=str(version), source="mlflow_registry")
    except Exception:
        logger.exception(
            "Could not load model '%s' (stage=%s) from MLflow registry at %s. "
            "Falling back to local artifact — this fallback must NOT be used in "
            "production.",
            model_name,
            stage,
            cfg.mlflow.tracking_uri,
        )
        return _load_local_fallback()


def _load_local_fallback() -> LoadedModel:
    cfg = get_config()
    model_path = resolve_path(cfg.paths.model.final_model)
    model = joblib.load(model_path)
    logger.warning("Loaded local fallback model from %s", model_path)
    return LoadedModel(model=model, version="local-fallback", source="local_artifact_fallback")
