"""
Database access layer.

Refactor of Notebook 1's "Database Connection" and "Read all tables" cells.
No hardcoded connection string — read from config.yaml / DATABASE_URL env var.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

RAW_TABLE_NAMES = [
    "category_name",
    "geolocation",
    "sellers",
    "products",
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
]


def get_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine from config (or an explicit override)."""
    cfg = get_config()
    url = database_url or cfg.database.url
    logger.info("Creating DB engine")
    return create_engine(url)


def read_raw_tables(engine: Engine | None = None) -> Dict[str, pd.DataFrame]:
    """
    Read every raw Olist table into a dict of DataFrames, keyed by table name.
    Equivalent to Notebook 1's individual `pd.read_sql(...)` calls.
    """
    engine = engine or get_engine()
    tables: Dict[str, pd.DataFrame] = {}
    for name in RAW_TABLE_NAMES:
        logger.info("Reading table '%s'", name)
        tables[name] = pd.read_sql(f"SELECT * FROM {name};", engine)
        logger.info("Table '%s' loaded: %d rows", name, len(tables[name]))
    return tables
