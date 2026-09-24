"""
CLI: train/val/test splits -> engineered + encoded feature tables + fitted artifacts.

Refactor of Notebook 5, runnable as:
    python -m src.pipeline.run_feature_engineering

DVC pipeline stage `feature_engineering` (see dvc.yaml).
"""

from __future__ import annotations

import pandas as pd

from src.pipeline.feature_engineering import run_feature_engineering
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = get_config()

    logger.info("=== Stage: feature_engineering ===")
    train_df = pd.read_parquet(resolve_path(cfg.paths.data.train))
    val_df = pd.read_parquet(resolve_path(cfg.paths.data.val))
    test_df = pd.read_parquet(resolve_path(cfg.paths.data.test))

    run_feature_engineering(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
