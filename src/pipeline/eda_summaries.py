"""
EDA summary functions: schema mapping, missing values, numeric and
categorical profiling. Refactor of the first half of Notebook 4
(task2nb4.ipynb): "Data types, shape, and memory", "Missing values",
"Numerical analysis", "Categorical analysis".

Every function here is a pure function of a DataFrame (+ config-driven
parameters) that returns a DataFrame/dict — no file I/O, no plotting. Plot
generation and file writes live in src/pipeline/eda.py's orchestrator, so
these functions stay unit-testable without a filesystem or a display backend.

This module documents and reproduces exploratory analysis; it is not part of
the inference path and src/inference never imports it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_column_type_mapping(column_types: Dict[str, str]) -> pd.DataFrame:
    """Turn the config-driven column -> semantic-type map into a DataFrame."""
    return pd.DataFrame(
        list(column_types.items()), columns=["Column", "Semantic_Type"]
    )


def columns_of_type(column_types: Dict[str, str], semantic_type: str, df: pd.DataFrame) -> List[str]:
    """Columns of a given semantic type that are actually present in df."""
    return [
        column
        for column, stype in column_types.items()
        if stype == semantic_type and column in df.columns
    ]


def dataset_shape_and_memory(df: pd.DataFrame) -> Dict[str, Any]:
    """Row/column counts and memory footprint, mirroring the notebook's print cell."""
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "memory_usage_mb": round(float(memory_mb), 2),
    }


def missing_value_report(
    df: pd.DataFrame, meaningful_missing_columns: Dict[str, str]
) -> pd.DataFrame:
    """
    Missing count/percentage per column, annotated with a known cause where
    one is configured (meaningful_missing_columns), else "Not meaningful".
    """
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing_Count"]
    missing_df["Missing_Percentage"] = missing_df["Missing_Count"] / len(df) * 100
    missing_df["Meaningful_Missing"] = (
        missing_df["Column"].map(meaningful_missing_columns).fillna("Not meaningful")
    )
    return missing_df


def numeric_summary_report(
    df: pd.DataFrame,
    numeric_columns: List[str],
    outlier_iqr_multiplier: float,
) -> pd.DataFrame:
    """
    describe() + median/skewness/range/IQR-outlier count per numeric column.
    """
    summary = df[numeric_columns].describe().T
    summary["median"] = df[numeric_columns].median()
    summary["skewness"] = df[numeric_columns].skew()
    summary["range"] = summary["max"] - summary["min"]
    summary["outlier_count"] = 0

    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_iqr_multiplier * iqr
        upper_bound = q3 + outlier_iqr_multiplier * iqr
        summary.loc[column, "outlier_count"] = int(
            ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        )

    summary["outlier_percentage"] = summary["outlier_count"] / len(df) * 100
    return summary


def invalid_negative_values(
    df: pd.DataFrame, numeric_columns: List[str], non_negative_keywords: List[str]
) -> Dict[str, int]:
    """
    Count negative values in numeric columns that should conceptually never
    be negative (matched by keyword in the column name, e.g. 'price', 'weight').
    """
    non_negative_columns = [
        column
        for column in numeric_columns
        if any(keyword in column.lower() for keyword in non_negative_keywords)
    ]
    return {
        column: int((df[column] < 0).sum()) for column in non_negative_columns
    }


def categorical_summary_report(
    df: pd.DataFrame,
    categorical_columns: List[str],
    rare_category_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per categorical column: cardinality, missing count, rare-category count,
    and normalized-duplicate groups (e.g. "SP" vs " sp " vs "Sp" colliding
    after strip+lowercase). Returns (summary_df, messy_values_df).
    """
    categorical_summary: List[Dict[str, Any]] = []
    messy_values: List[Dict[str, Any]] = []

    for column in categorical_columns:
        values = df[column].dropna().astype(str)
        counts = values.value_counts()
        normalized = values.str.strip().str.lower()
        normalized_counts = normalized.value_counts()

        categorical_summary.append(
            {
                "column": column,
                "cardinality": int(values.nunique()),
                "missing_count": int(df[column].isna().sum()),
                "rare_categories_lt_1pct": int(
                    (counts / len(df) < rare_category_threshold).sum()
                ),
                "normalized_duplicate_groups": int((normalized_counts > 1).sum()),
            }
        )

        for normalized_value in normalized_counts[normalized_counts > 1].index:
            original_values = sorted(values[normalized == normalized_value].unique())
            if len(original_values) > 1:
                messy_values.append(
                    {
                        "column": column,
                        "normalized_value": normalized_value,
                        "original_values": original_values,
                    }
                )

    return pd.DataFrame(categorical_summary), pd.DataFrame(messy_values)