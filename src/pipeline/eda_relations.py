"""
EDA relationship functions: label relations, date-based analysis, geography
analysis. Refactor of the second half of Notebook 4 (task2nb4.ipynb):
"Relations with the label", "Date analysis", "Geography analysis".

Same rules as eda_summaries.py: pure functions of a DataFrame, no file I/O
or plotting. Not part of the inference path.

Note on is_holiday: Notebook 4 originally hardcoded a fixed list of
Brazilian holiday dates. This refactor uses the `holidays` library's
holidays.Brazil(years=...) instead (the same source already used by
src/features/derived_features.py for the production feature pipeline), so
the EDA's holiday definition can never silently drift from the one the model
actually trains and predicts on.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import holidays
import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def label_relation_reports(
    df: pd.DataFrame, numeric_columns: List[str], target_col: str
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Late rate by order_status and by payment_types, a status/label cross-tab,
    and the correlation of every numeric column with the target.
    """
    status_relation = (
        df.groupby("order_status", dropna=False)[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    status_relation["late_rate_percent"] = status_relation["mean"] * 100

    payment_relation = (
        df.groupby("payment_types", dropna=False)[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    payment_relation["late_rate_percent"] = payment_relation["mean"] * 100

    cross_tab = pd.crosstab(df["order_status"], df[target_col], normalize="index") * 100
    correlation_matrix = df[numeric_columns + [target_col]].corr()

    return {
        "status_relation": status_relation,
        "payment_relation": payment_relation,
        "cross_tab": cross_tab,
        "correlation_matrix": correlation_matrix,
    }


def add_date_derived_columns(
    df: pd.DataFrame, date_columns: List[str], years: List[int] | None = None
) -> pd.DataFrame:
    """
    Parse date columns and add purchase_month, purchase_weekday, delivery_days,
    delivery_delay_days, is_holiday. delivery_delay_days is for EDA reference
    only (it is how is_late is built) and must never be used as a model input.
    """
    date_df = df.copy()
    for column in date_columns:
        date_df[column] = pd.to_datetime(date_df[column], errors="coerce")

    date_df["purchase_month"] = date_df["order_purchase_timestamp"].dt.month
    date_df["purchase_weekday"] = date_df["order_purchase_timestamp"].dt.day_name()
    date_df["delivery_days"] = (
        date_df["order_delivered_customer_date"] - date_df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    date_df["delivery_delay_days"] = (
        date_df["order_delivered_customer_date"] - date_df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    years = years or date_df["order_purchase_timestamp"].dt.year.dropna().unique().tolist()
    br_holidays = holidays.Brazil(years=years)
    date_df["is_holiday"] = date_df["order_purchase_timestamp"].dt.normalize().isin(
        br_holidays
    )

    return date_df


def date_relation_reports(
    date_df: pd.DataFrame, target_col: str
) -> Dict[str, pd.Series | float]:
    """Late rate by weekday / month / holiday flag, plus delivery-time medians."""
    weekday_late_rate = (
        date_df.groupby("purchase_weekday")[target_col].mean().reindex(WEEKDAY_ORDER) * 100
    )
    monthly_late_rate = date_df.groupby("purchase_month")[target_col].mean() * 100
    holiday_late_rate = date_df.groupby("is_holiday")[target_col].mean() * 100

    return {
        "weekday_late_rate": weekday_late_rate,
        "monthly_late_rate": monthly_late_rate,
        "holiday_late_rate": holiday_late_rate,
        "purchase_date_min": date_df["order_purchase_timestamp"].min(),
        "purchase_date_max": date_df["order_purchase_timestamp"].max(),
        "median_delivery_days": float(date_df["delivery_days"].median()),
        "median_delivery_delay_days": float(date_df["delivery_delay_days"].median()),
    }


def add_same_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """Whether at least one seller is in the same state as the customer."""
    geo_df = df.copy()
    geo_df["same_state"] = geo_df.apply(
        lambda row: row["customer_state"] in str(row["seller_states"]).split(", "),
        axis=1,
    )
    return geo_df


def geography_reports(
    geo_df: pd.DataFrame,
    target_col: str,
    min_state_orders: int,
    distance_bucket_edges: List[float],
    distance_bucket_labels: List[str],
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Same-state vs different-state late rate, per-state late rate (ranked),
    distance-bucket late rate, and distance/is_late correlations.
    """
    same_state_late_rate = geo_df.groupby("same_state")[target_col].mean() * 100

    state_late_rate = (
        geo_df.groupby("customer_state")[target_col]
        .agg(["count", "mean"])
        .rename(columns={"mean": "late_rate"})
        .sort_values("late_rate", ascending=False)
    )
    state_late_rate["late_rate_percent"] = state_late_rate["late_rate"] * 100

    geo_df = geo_df.copy()
    geo_df["distance_bucket"] = pd.cut(
        geo_df["avg_distance_km"],
        bins=distance_bucket_edges,
        labels=distance_bucket_labels,
    )
    distance_late_rate = geo_df.groupby("distance_bucket", observed=True)[target_col].agg(
        ["count", "mean"]
    )
    distance_late_rate["late_rate_percent"] = distance_late_rate["mean"] * 100

    distance_corr = geo_df[
        ["avg_distance_km", "max_distance_km", "min_distance_km", target_col]
    ].corr()[target_col]

    top_states = state_late_rate[state_late_rate["count"] >= min_state_orders]

    return {
        "same_state_late_rate": same_state_late_rate,
        "state_late_rate": state_late_rate,
        "top_states_by_late_rate": top_states,
        "distance_late_rate": distance_late_rate,
        "distance_corr": distance_corr,
        "missing_avg_distance_km": int(geo_df["avg_distance_km"].isna().sum()),
        "customer_zip_cardinality": int(geo_df["customer_zip_code_prefix"].nunique()),
        "seller_zip_cardinality": int(geo_df["seller_zip_codes"].nunique()),
    }
