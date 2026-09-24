"""
CLI: raw DB tables -> ml_orders_dataset.parquet -> ml_orders_labeled.parquet.

Refactor of Notebooks 1 and 2, runnable as:
    python -m src.pipeline.run_data_prep

This is a DVC pipeline stage (see dvc.yaml, stage `data_prep`).
"""

from __future__ import annotations

from src.data.build_ml_table import build_ml_table
from src.data.db import read_raw_tables
from src.data.labeling import build_labeled_table
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = get_config()

    logger.info("=== Stage: data_prep ===")
    tables = read_raw_tables()

    ml_table = build_ml_table(tables)
    ml_table_path = resolve_path(cfg.paths.data.ml_table)
    ml_table_path.parent.mkdir(parents=True, exist_ok=True)
    ml_table.to_parquet(ml_table_path, index=False)
    logger.info("Saved ML table to %s", ml_table_path)

    labeled_table = build_labeled_table(ml_table)
    labeled_path = resolve_path(cfg.paths.data.ml_table_labeled)
    labeled_table.to_parquet(labeled_path, index=False)
    logger.info("Saved labeled table to %s", labeled_path)


if __name__ == "__main__":
    main()
