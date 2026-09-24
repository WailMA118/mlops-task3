"""
Feature selection, missing-value handling, and encoding.

Refactor of the remainder of Notebook 5 (task2nb5.ipynb): "Select features",
"Handle missing values, encode categories", "Assemble the final feature tables".

Split deliberately into FIT functions (training-time only, produce fitted
transformers) and TRANSFORM functions (used both at training time on
train/val/test, and at inference time on new orders — loading the already
-fitted transformers, never re-fitting them). This is the boundary Task 3
requires: "Load the saved fitted objects ... never fit again at inference".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class FittedFeatureArtifacts:
    """Everything the transform step needs, loaded once at inference start."""

    numeric_imputer: SimpleImputer
    onehot_encoder: OneHotEncoder
    payment_encoder: MultiLabelBinarizer
    same_state_fill_value: Any
    final_feature_list: list[str]
    numeric_features: list[str]
    categorical_single_features: list[str]
    multi_value_feature: str
    boolean_features: list[str]


def select_raw_features(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    Slice out exactly the raw columns the feature pipeline needs (plus the
    target if present and requested). Mirrors Notebook 5's "Select features"
    + "Build the raw feature table" cells.
    """
    cfg = get_config()
    numeric_features = list(cfg.features.numeric_features)
    categorical_single_features = list(cfg.features.categorical_single_features)
    multi_value_feature = cfg.features.multi_value_feature
    boolean_features = list(cfg.features.boolean_features)

    selected = (
        numeric_features
        + categorical_single_features
        + [multi_value_feature]
        + boolean_features
    )
    cols = selected + ([target_col] if target_col and target_col in df.columns else [])
    return df[cols].copy()


def fit_feature_transformers(train_table: pd.DataFrame) -> FittedFeatureArtifacts:
    """
    TRAINING-TIME ONLY. Fit the numeric imputer, one-hot encoder, payment
    multi-label binarizer, and same_state fill value on the training split.
    Returns a FittedFeatureArtifacts bundle (not yet the final feature list,
    which is only known after transforming — see assemble_feature_table).
    """
    cfg = get_config()
    numeric_features = list(cfg.features.numeric_features)
    categorical_single_features = list(cfg.features.categorical_single_features)
    multi_value_feature = cfg.features.multi_value_feature
    boolean_features = list(cfg.features.boolean_features)

    table = train_table.copy()

    numeric_imputer = SimpleImputer(strategy="median")
    numeric_imputer.fit(table[numeric_features])

    same_state_fill_value = table["same_state"].mode(dropna=True)[0]

    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot_encoder.fit(table[categorical_single_features].fillna("missing"))

    def _to_payment_lists(t: pd.DataFrame) -> pd.Series:
        return t[multi_value_feature].fillna("missing").apply(
            lambda x: [p.strip() for p in str(x).split(",")]
        )

    payment_encoder = MultiLabelBinarizer()
    payment_encoder.fit(_to_payment_lists(table))

    logger.info("Fitted numeric imputer, one-hot encoder, payment encoder")

    return FittedFeatureArtifacts(
        numeric_imputer=numeric_imputer,
        onehot_encoder=onehot_encoder,
        payment_encoder=payment_encoder,
        same_state_fill_value=same_state_fill_value,
        final_feature_list=[],  # filled in after the first transform, see train pipeline
        numeric_features=numeric_features,
        categorical_single_features=categorical_single_features,
        multi_value_feature=multi_value_feature,
        boolean_features=boolean_features,
    )


def transform_feature_table(
    raw_table: pd.DataFrame,
    artifacts: FittedFeatureArtifacts,
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    Apply already-fitted transformers to a raw feature table (train, val,
    test, or a batch of new orders at inference time). Never fits anything.

    Mirrors Notebook 5's transform cells exactly: median-impute numerics,
    fill same_state/categoricals, one-hot encode, multi-hot encode payment
    types, assemble booleans, concatenate into the final feature table.
    """
    table = raw_table.copy()

    # ---- numeric: impute with the already-fitted median imputer ----
    table[artifacts.numeric_features] = artifacts.numeric_imputer.transform(
        table[artifacts.numeric_features]
    )

    # ---- same_state: missing-flag + fill with the training-time mode ----
    table["same_state_missing"] = table["same_state"].isna().astype(int)
    table["same_state"] = table["same_state"].fillna(artifacts.same_state_fill_value)

    # ---- categorical / multi-value: constant fill ----
    for col in artifacts.categorical_single_features:
        table[col] = table[col].fillna("missing")
    table[artifacts.multi_value_feature] = table[artifacts.multi_value_feature].fillna(
        "missing"
    )

    # ---- one-hot encode ----
    onehot_array = artifacts.onehot_encoder.transform(
        table[artifacts.categorical_single_features]
    )
    onehot_cols = artifacts.onehot_encoder.get_feature_names_out(
        artifacts.categorical_single_features
    )
    onehot_df = pd.DataFrame(onehot_array, columns=onehot_cols, index=table.index)

    # ---- multi-hot encode payment_types ----
    payment_lists = table[artifacts.multi_value_feature].apply(
        lambda x: [p.strip() for p in str(x).split(",")]
    )
    payment_array = artifacts.payment_encoder.transform(payment_lists)
    payment_cols = [f"payment_type_{c}" for c in artifacts.payment_encoder.classes_]
    payment_df = pd.DataFrame(payment_array, columns=payment_cols, index=table.index)

    # ---- booleans ----
    bool_features_all = artifacts.boolean_features + ["same_state_missing"]
    bool_df = table[bool_features_all].astype(int)

    parts = [
        table[artifacts.numeric_features].reset_index(drop=True),
        onehot_df.reset_index(drop=True),
        payment_df.reset_index(drop=True),
        bool_df.reset_index(drop=True),
    ]
    if target_col and target_col in table.columns:
        parts.append(table[[target_col]].reset_index(drop=True))

    result = pd.concat(parts, axis=1)
    return result


def save_feature_artifacts(artifacts: FittedFeatureArtifacts) -> None:
    """TRAINING-TIME ONLY. Persist fitted transformers + feature lists to disk."""
    cfg = get_config()
    out_dir = resolve_path(cfg.paths.artifacts.feature_engineering_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.numeric_imputer, resolve_path(cfg.paths.artifacts.numeric_imputer))
    joblib.dump(artifacts.onehot_encoder, resolve_path(cfg.paths.artifacts.onehot_encoder))
    joblib.dump(artifacts.payment_encoder, resolve_path(cfg.paths.artifacts.payment_encoder))
    joblib.dump(
        artifacts.same_state_fill_value,
        resolve_path(cfg.paths.artifacts.same_state_fill_value),
    )

    raw_feature_list = (
        artifacts.numeric_features
        + artifacts.categorical_single_features
        + [artifacts.multi_value_feature]
        + artifacts.boolean_features
    )
    with open(resolve_path(cfg.paths.artifacts.feature_list_raw), "w", encoding="utf-8") as f:
        json.dump(raw_feature_list, f, indent=2)

    with open(resolve_path(cfg.paths.artifacts.final_feature_list), "w", encoding="utf-8") as f:
        json.dump(artifacts.final_feature_list, f, indent=2)

    logger.info("Saved feature engineering artifacts to %s", out_dir)


def load_feature_artifacts() -> FittedFeatureArtifacts:
    """
    INFERENCE-TIME. Load the already-fitted transformers and feature lists
    from disk. Never fits anything — this is the only way inference obtains
    these objects.
    """
    cfg = get_config()

    numeric_imputer = joblib.load(resolve_path(cfg.paths.artifacts.numeric_imputer))
    onehot_encoder = joblib.load(resolve_path(cfg.paths.artifacts.onehot_encoder))
    payment_encoder = joblib.load(resolve_path(cfg.paths.artifacts.payment_encoder))
    same_state_fill_value = joblib.load(
        resolve_path(cfg.paths.artifacts.same_state_fill_value)
    )

    with open(resolve_path(cfg.paths.artifacts.final_feature_list), encoding="utf-8") as f:
        final_feature_list = json.load(f)

    artifacts = FittedFeatureArtifacts(
        numeric_imputer=numeric_imputer,
        onehot_encoder=onehot_encoder,
        payment_encoder=payment_encoder,
        same_state_fill_value=same_state_fill_value,
        final_feature_list=final_feature_list,
        numeric_features=list(cfg.features.numeric_features),
        categorical_single_features=list(cfg.features.categorical_single_features),
        multi_value_feature=cfg.features.multi_value_feature,
        boolean_features=list(cfg.features.boolean_features),
    )
    logger.info("Loaded feature engineering artifacts (never re-fit)")
    return artifacts
