"""
CLI: feature tables -> trained/selected model, tracked and registered in MLflow.

Refactor of Notebook 6, runnable as:
    python -m src.pipeline.run_train

DVC pipeline stage `train` (see dvc.yaml).
"""

from __future__ import annotations

import json

import pandas as pd

from src.pipeline.train import save_local_model_artifacts, train_and_select_model
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = get_config()

    logger.info("=== Stage: train ===")
    train_df = pd.read_parquet(resolve_path(cfg.paths.data.feature_table_train))
    val_df = pd.read_parquet(resolve_path(cfg.paths.data.feature_table_val))
    test_df = pd.read_parquet(resolve_path(cfg.paths.data.feature_table_test))

    with open(resolve_path(cfg.paths.artifacts.final_feature_list), encoding="utf-8") as f:
        feature_list = json.load(f)

    final_model, test_metrics, results_summary = train_and_select_model(
        train_df, val_df, test_df, feature_list
    )

    best_params = {
        k: v
        for k, v in final_model.get_params().items()
        if k in ("n_estimators", "max_depth", "min_samples_leaf")
    }
    save_local_model_artifacts(
        final_model, results_summary, feature_list, best_params, test_metrics
    )


if __name__ == "__main__":
    main()
