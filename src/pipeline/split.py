"""
Stratified random train/val/test split.

Refactor of Notebook 3 (task2nb3.ipynb). Time-based splitting was rejected in
the notebook (unstable monthly late-rate trend); stratified random split on
`is_late` was chosen instead. Training-time only.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def stratified_split(
    ml_table: pd.DataFrame,
    test_size: float | None = None,
    val_ratio_of_temp: float | None = None,
    stratify_column: str | None = None,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test with the same two-step stratified approach
    as Notebook 3: first carve off `test_size` as a temp set, then split that
    temp set evenly (by default) into val and test, both stratified on the
    target column.

    All defaults are pulled from config.yaml if not given explicitly.
    """
    cfg = get_config()
    test_size = cfg.split.test_size if test_size is None else test_size
    val_ratio_of_temp = (
        cfg.split.val_ratio_of_temp if val_ratio_of_temp is None else val_ratio_of_temp
    )
    stratify_column = (
        cfg.split.stratify_column if stratify_column is None else stratify_column
    )
    random_state = cfg.project.random_state if random_state is None else random_state

    train_df, temp_df = train_test_split(
        ml_table,
        test_size=test_size,
        random_state=random_state,
        stratify=ml_table[stratify_column],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_ratio_of_temp,
        random_state=random_state,
        stratify=temp_df[stratify_column],
    )

    logger.info(
        "Split sizes -- train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df