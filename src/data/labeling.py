"""
Labeling logic: drop undelivered orders, build the `is_late` target.

Refactor of Notebook 2 (task2nb2.ipynb). Training-time only — inference never
computes or needs this label; it exists here for reproducibility and for the
data tests (schema/leakage checks required by Task 3, section 6).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

DELIVERED_COL = "order_delivered_customer_date"
ESTIMATED_COL = "order_estimated_delivery_date"
TARGET_COL = "is_late"


def drop_undelivered_orders(ml_table: pd.DataFrame) -> pd.DataFrame:
    """
    Remove orders with no delivered date. "Late" vs "on time" is undefined
    for an order that never arrived, and NaT comparisons silently return
    False in pandas, which would mislabel undelivered orders as on-time.
    """
    before_rows = len(ml_table)
    out = ml_table[ml_table[DELIVERED_COL].notna()].copy()
    logger.info(
        "Dropped undelivered orders: %d -> %d (removed %d)",
        before_rows,
        len(out),
        before_rows - len(out),
    )
    return out


def add_is_late_label(ml_table: pd.DataFrame) -> pd.DataFrame:
    """Label is 1 if delivered after the estimated date, else 0."""
    out = ml_table.copy()
    out[TARGET_COL] = (out[DELIVERED_COL] > out[ESTIMATED_COL]).astype(int)
    return out


def build_labeled_table(ml_table: pd.DataFrame) -> pd.DataFrame:
    """Full Notebook 2 pipeline: drop undelivered, then label."""
    out = drop_undelivered_orders(ml_table)
    out = add_is_late_label(out)
    rate = out[TARGET_COL].mean()
    logger.info("Label built. Positive rate: %.4f", rate)
    return out
