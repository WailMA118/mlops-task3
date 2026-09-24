"""
EDA plotting functions. Refactor of every `plt.subplots(...) / sns.xxx(...) /
plt.savefig(...)` cell in Notebook 4 (task2nb4.ipynb).

Kept separate from src/pipeline/eda_summaries.py and eda_relations.py so the
analysis logic is testable without a display backend. Every function here
takes already-computed data (never re-derives it) and writes exactly one PNG
to the given path; none of them call plt.show(), since this runs headless in
CI/containers.

matplotlib is forced to the non-interactive "Agg" backend at import time,
before pyplot is imported, which is required for headless environments
(containers, CI) that have no X display.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_numerical_distributions_and_outliers(
    df: pd.DataFrame, numeric_columns: List[str], out_path: Path
) -> None:
    """Histogram + boxplot per numeric column, two rows x N columns."""
    fig, axes = plt.subplots(
        2, len(numeric_columns), figsize=(4 * len(numeric_columns), 7), squeeze=False
    )
    for index, column in enumerate(numeric_columns):
        sns.histplot(df[column].dropna(), kde=True, ax=axes[0, index])
        axes[0, index].set_title(f"{column} distribution")
        sns.boxplot(x=df[column], ax=axes[1, index])
        axes[1, index].set_title(f"{column} outliers")
    _save(fig, out_path)


def plot_categorical_top_values(
    df: pd.DataFrame, column: str, top_n: int, out_path: Path
) -> None:
    """Horizontal bar chart of the top-N category values for one column."""
    counts = df[column].fillna("<MISSING>").astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, palette="viridis", ax=ax)
    ax.set_title(f"Top Categories: {column}")
    _save(fig, out_path)


def plot_relations_and_correlations(
    status_relation: pd.DataFrame,
    payment_relation: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    out_path: Path,
) -> None:
    """Late rate by status, late rate by payment type, and the correlation heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(
        data=status_relation, x="order_status", y="late_rate_percent",
        hue="order_status", legend=False, ax=axes[0], palette="Reds",
    )
    sns.barplot(
        data=payment_relation, x="payment_types", y="late_rate_percent",
        hue="payment_types", legend=False, ax=axes[1], palette="Oranges",
    )
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2])
    axes[0].set_title("Late Rate by Order Status")
    axes[1].set_title("Late Rate by Payment Type")
    axes[2].set_title("Numerical Correlations")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=45)
    _save(fig, out_path)


def plot_date_analysis(
    monthly_late_rate: pd.Series,
    weekday_late_rate: pd.Series,
    holiday_late_rate: pd.Series,
    delivery_days: pd.Series,
    out_path: Path,
) -> None:
    """Monthly trend, weekday effect, holiday effect, delivery-time distribution."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    sns.lineplot(x=monthly_late_rate.index, y=monthly_late_rate.values, marker="o", ax=axes[0])
    axes[0].set_title("Monthly Late Rate")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Late Rate (%)")

    sns.barplot(x=weekday_late_rate.index, y=weekday_late_rate.values, ax=axes[1], color="#2ecc71")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Late Rate by Weekday")

    sns.barplot(
        x=holiday_late_rate.index.astype(str), y=holiday_late_rate.values,
        ax=axes[2], color="#9b59b6",
    )
    axes[2].set_title("Holiday Effect")
    axes[2].set_xlabel("Is Holiday")
    axes[2].set_ylabel("Late Rate (%)")

    sns.histplot(delivery_days.dropna(), kde=True, ax=axes[3], color="#3498db")
    axes[3].set_title("Delivery Time Distribution")
    axes[3].set_xlabel("Days")

    _save(fig, out_path)


def plot_geography_analysis(
    avg_distance_km: pd.Series,
    distance_late_rate: pd.DataFrame,
    top_states: pd.DataFrame,
    out_path: Path,
) -> None:
    """Distance distribution, late rate by distance bucket, late rate by state."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(avg_distance_km.dropna(), kde=True, ax=axes[0], color="#e67e22")
    axes[0].set_title("Average Customer-Seller Distance (km)")

    sns.barplot(
        x=distance_late_rate.index.astype(str), y=distance_late_rate["late_rate_percent"],
        ax=axes[1], color="#c0392b",
    )
    axes[1].set_title("Late Rate by Distance Bucket")
    axes[1].tick_params(axis="x", rotation=45)

    sns.barplot(x=top_states.index, y=top_states["late_rate_percent"], ax=axes[2], color="#16a085")
    axes[2].set_title("Late Rate by Customer State")
    axes[2].tick_params(axis="x", rotation=45)

    _save(fig, out_path)
