"""
Feature engineering orchestrator (training time).

Ties together src/features/derived_features.py, src/features/shipping_pressure.py
and src/features/build_features.py into the same sequence as Notebook 5:
  1. add derived features to train/val/test
  2. add causal shipping-pressure features
  3. select raw feature columns
  4. fit transformers on train, transform train/val/test
  5. save feature tables + fitted artifacts + final feature list

This is training-time orchestration only. Inference reuses
src/features/build_features.py's transform_feature_table + load_feature_artifacts
directly, never this module.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.features.build_features import (
    FittedFeatureArtifacts,
    fit_feature_transformers,
    save_feature_artifacts,
    select_raw_features,
    transform_feature_table,
)
from src.features.derived_features import add_derived_features
from src.features.shipping_pressure import add_shipping_pressure_features
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def run_feature_engineering(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FittedFeatureArtifacts]:
    """
    Full training-time feature engineering pipeline. Returns the three final
    (post-encoding) feature tables and the fitted artifacts bundle.
    """
    cfg = get_config()
    target_col = cfg.target.column

    logger.info("Adding derived features (same_state, purchase_month/weekday, is_holiday)")
    train_feat = add_derived_features(train_df)
    val_feat = add_derived_features(val_df)
    test_feat = add_derived_features(test_df)

    logger.info("Adding causal shipping-pressure features")
    train_feat_p, val_feat_p, test_feat_p = add_shipping_pressure_features(
        train_feat, val_feat, test_feat
    )
    train_feat["customer_state_pressure_5d"] = train_feat_p["customer_state_pressure_5d"]
    train_feat["seller_pressure_5d"] = train_feat_p["seller_pressure_5d"]
    val_feat["customer_state_pressure_5d"] = val_feat_p["customer_state_pressure_5d"]
    val_feat["seller_pressure_5d"] = val_feat_p["seller_pressure_5d"]
    test_feat["customer_state_pressure_5d"] = test_feat_p["customer_state_pressure_5d"]
    test_feat["seller_pressure_5d"] = test_feat_p["seller_pressure_5d"]

    logger.info("Selecting raw feature columns")
    train_table = select_raw_features(train_feat, target_col)
    val_table = select_raw_features(val_feat, target_col)
    test_table = select_raw_features(test_feat, target_col)

    logger.info("Fitting transformers on train only")
    artifacts = fit_feature_transformers(train_table)

    final_train = transform_feature_table(train_table, artifacts, target_col)
    artifacts.final_feature_list = [c for c in final_train.columns if c != target_col]

    final_val = transform_feature_table(val_table, artifacts, target_col)
    final_test = transform_feature_table(test_table, artifacts, target_col)

    assert list(final_val.columns) == list(final_train.columns), (
        "val columns must match train exactly"
    )
    assert list(final_test.columns) == list(final_train.columns), (
        "test columns must match train exactly"
    )

    logger.info(
        "Feature tables built -- train: %s, val: %s, test: %s",
        final_train.shape,
        final_val.shape,
        final_test.shape,
    )

    save_feature_artifacts(artifacts)

    final_train.to_parquet(resolve_path(cfg.paths.data.feature_table_train), index=False)
    final_val.to_parquet(resolve_path(cfg.paths.data.feature_table_val), index=False)
    final_test.to_parquet(resolve_path(cfg.paths.data.feature_table_test), index=False)
    logger.info("Saved feature tables to %s", resolve_path(cfg.paths.data.processed_dir))

    return final_train, final_val, final_test, artifacts
