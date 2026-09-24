"""
CLI: run exploratory data analysis on the train split, writing summary
reports, plots, and a feature-selection rationale document.

Refactor of Notebook 4 (task2nb4.ipynb), runnable as:
    python -m src.pipeline.run_eda

Optional DVC pipeline stage `eda` (see dvc.yaml) — not a dependency of the
`train` stage, since EDA is diagnostic output, not a model input.
"""

from __future__ import annotations

import pandas as pd

from src.pipeline.eda import run_eda
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = get_config()

    logger.info("=== Stage: eda ===")
    input_path = resolve_path(cfg.paths.data[cfg.eda.input_split])
    df = pd.read_parquet(input_path)
    logger.info("Loaded %s: %d rows, %d columns", input_path, len(df), df.shape[1])

    run_eda(df)


if __name__ == "__main__":
    main()