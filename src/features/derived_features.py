"""
Derived feature creation: same_state, purchase_month, purchase_weekday, is_holiday.

Refactor of Notebook 5's `add_derived_features` cell. Pure function of a
DataFrame's own columns — no fitting involved, so it's identical at training
and inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _same_state_row(row: pd.Series) -> bool | float:
    if pd.isna(row["seller_states"]) or pd.isna(row["customer_state"]):
        return np.nan
    seller_state_list = [s.strip() for s in str(row["seller_states"]).split(",")]
    return row["customer_state"] in seller_state_list


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add same_state, purchase_month, purchase_weekday, is_holiday to a copy of df.

    Requires columns: order_purchase_timestamp, seller_states, customer_state.
    """
    feat = df.copy()
    feat["order_purchase_timestamp"] = pd.to_datetime(feat["order_purchase_timestamp"])

    feat["same_state"] = feat.apply(_same_state_row, axis=1)

    feat["purchase_month"] = feat["order_purchase_timestamp"].dt.month
    feat["purchase_weekday"] = feat["order_purchase_timestamp"].dt.day_name()

    years = feat["order_purchase_timestamp"].dt.year.unique().tolist()
    br_holidays = holidays.Brazil(years=years)
    feat["is_holiday"] = feat["order_purchase_timestamp"].dt.normalize().isin(br_holidays)
    feat["is_holiday"] = feat["is_holiday"].astype(bool)

    return feat
