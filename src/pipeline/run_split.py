"""
CLI: labeled table -> stratified train/val/test parquet files.

Refactor of Notebook 3, runnable as:
    python -m src.pipeline.run_split

DVC pipeline stage `split` (see dvc.yaml).
"""

from __future__ import annotations

import pandas as pd

from src.pipeline.split import stratified_split
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = get_config()

    logger.info("=== Stage: split ===")
    labeled_path = resolve_path(cfg.paths.data.ml_table_labeled)
    ml_table = pd.read_parquet(labeled_path)

    train_df, val_df, test_df = stratified_split(ml_table)

    for name, df, path_key in (
        ("train", train_df, "train"),
        ("val", val_df, "val"),
        ("test", test_df, "test"),
    ):
        out_path = resolve_path(cfg.paths.data[path_key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s split to %s (%d rows)", name, out_path, len(df))


if __name__ == "__main__":
    main()
