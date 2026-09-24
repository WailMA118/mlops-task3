"""
Shipping-pressure features: trailing 5-day causal order counts.

Refactor of Notebook 5's shipping-pressure cell.

Two features, both CAUSAL (only counts orders strictly before the current
order's timestamp, within a trailing window — never look-ahead):
  - customer_state_pressure_5d: trailing order count at customer-state level
  - seller_pressure_5d: trailing order count at seller-state level (orders
    spanning multiple seller states get the mean pressure across those states)

At training time the trailing count is computed over the full combined
train+val+test chronological log (since the split is random, not
time-based), then mapped back to each split. At inference time for a single
new order, this module also exposes `compute_pressure_for_new_order`, which
computes the same trailing count against a reference historical log (e.g.
the full training-time order log persisted as an artifact), never against
future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["order_purchase_timestamp", "customer_state", "seller_states"]


def _trailing_causal_count(times: np.ndarray, window_days: int) -> np.ndarray:
    """Count earlier timestamps within [t - window_days, t) for each t."""
    order = np.argsort(times)
    sorted_times = times[order]
    window = np.timedelta64(window_days, "D")

    upper = np.searchsorted(sorted_times, sorted_times, side="left")
    lower = np.searchsorted(sorted_times, sorted_times - window, side="left")
    sorted_counts = upper - lower

    counts = np.empty_like(sorted_counts)
    counts[order] = sorted_counts
    return counts


def add_shipping_pressure_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Training-time version: compute pressure features over the combined
    train+val+test chronological log, then attach them back to each split.
    Returns copies of train_df/val_df/test_df with the two new columns added.
    """
    cfg = get_config()
    window_days = cfg.features.window_days if window_days is None else window_days

    def prep(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        tmp = df[REQUIRED_COLUMNS].copy()
        tmp["order_purchase_timestamp"] = pd.to_datetime(tmp["order_purchase_timestamp"])
        tmp["_split"] = split_name
        tmp["_orig_index"] = df.index.values
        return tmp

    combined = pd.concat(
        [prep(train_df, "train"), prep(val_df, "val"), prep(test_df, "test")],
        ignore_index=True,
    )
    combined["_row_id"] = combined.index

    combined["customer_state_pressure_5d"] = np.nan
    for _, group in combined.dropna(subset=["customer_state"]).groupby("customer_state"):
        counts = _trailing_causal_count(group["order_purchase_timestamp"].values, window_days)
        combined.loc[group.index, "customer_state_pressure_5d"] = counts

    exploded = combined[["_row_id", "order_purchase_timestamp", "seller_states"]].copy()
    exploded["seller_states"] = exploded["seller_states"].fillna("")
    exploded = exploded.assign(
        seller_state=exploded["seller_states"].astype(str).str.split(",")
    ).explode("seller_state")
    exploded["seller_state"] = exploded["seller_state"].str.strip()
    exploded = exploded[exploded["seller_state"] != ""].reset_index(drop=True)

    exploded["seller_pressure_5d"] = np.nan
    for _, group in exploded.groupby("seller_state"):
        counts = _trailing_causal_count(group["order_purchase_timestamp"].values, window_days)
        exploded.loc[group.index, "seller_pressure_5d"] = counts

    seller_pressure_per_order = exploded.groupby("_row_id")["seller_pressure_5d"].mean()
    combined["seller_pressure_5d"] = combined["_row_id"].map(seller_pressure_per_order)

    def extract(split_name: str, orig_df: pd.DataFrame) -> pd.DataFrame:
        sub = combined[combined["_split"] == split_name].set_index("_orig_index")
        out = orig_df.copy()
        out["customer_state_pressure_5d"] = sub["customer_state_pressure_5d"]
        out["seller_pressure_5d"] = sub["seller_pressure_5d"]
        return out

    logger.info("Computed shipping pressure features (window=%d days)", window_days)
    return extract("train", train_df), extract("val", val_df), extract("test", test_df)


def compute_pressure_for_new_order(
    new_order: pd.Series,
    reference_log: pd.DataFrame,
    window_days: int | None = None,
) -> dict:
    """
    Inference-time pressure computation for a single new order.

    Counts, against `reference_log` (a historical order log with the same
    REQUIRED_COLUMNS, e.g. persisted from training data), how many orders
    fall strictly within [new_order_time - window_days, new_order_time) at
    the customer-state and seller-state level. This never looks at future
    data relative to the new order, and never mutates the reference log.

    Returns a dict: {"customer_state_pressure_5d": float, "seller_pressure_5d": float}
    """
    cfg = get_config()
    window_days = cfg.features.window_days if window_days is None else window_days
    window = np.timedelta64(window_days, "D")

    order_time = pd.to_datetime(new_order["order_purchase_timestamp"])
    lower_bound = order_time - window

    customer_state = new_order.get("customer_state")
    customer_pressure = np.nan
    if pd.notna(customer_state):
        mask = (
            (reference_log["customer_state"] == customer_state)
            & (reference_log["order_purchase_timestamp"] >= lower_bound)
            & (reference_log["order_purchase_timestamp"] < order_time)
        )
        customer_pressure = float(mask.sum())

    seller_states_raw = new_order.get("seller_states")
    seller_pressure = np.nan
    if pd.notna(seller_states_raw) and str(seller_states_raw).strip():
        seller_states = [s.strip() for s in str(seller_states_raw).split(",") if s.strip()]
        per_state_counts = []
        for state in seller_states:
            exploded_states = (
                reference_log["seller_states"]
                .fillna("")
                .astype(str)
                .str.split(",")
            )
            has_state = exploded_states.apply(
                lambda states, s=state: s in [x.strip() for x in states]
            )
            mask = (
                has_state
                & (reference_log["order_purchase_timestamp"] >= lower_bound)
                & (reference_log["order_purchase_timestamp"] < order_time)
            )
            per_state_counts.append(float(mask.sum()))
        if per_state_counts:
            seller_pressure = float(np.mean(per_state_counts))

    return {
        "customer_state_pressure_5d": customer_pressure,
        "seller_pressure_5d": seller_pressure,
    }
