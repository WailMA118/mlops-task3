"""
Build the one-row-per-order ML table from the raw Olist tables.

Refactor of Notebook 1 (task2nb1.ipynb), STEP 1 through STEP 5:
  - item-level merge (order_items + sellers + products + customer zip)
  - per-item haversine distance, aggregated to order level
  - order_items aggregated to one row per order_id
  - order_payments aggregated to one row per order_id
  - orders + customers + the two aggregates joined into the final ML table

This module is used both:
  (a) at training time, to rebuild ml_orders_dataset.parquet from the DB, and
  (b) conceptually at inference time as documentation of how a single new
      order's raw fields map to the aggregated columns the feature pipeline
      expects (see src/inference/predict.py for the actual single-order path,
      which accepts already-aggregated order-level input rather than raw
      order_items/order_payments rows).
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.data.geo import build_zip_geo_lookup, haversine_km
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _merge_item_level(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """STEP 1: item-level merge with seller/product/customer-zip + distance."""
    order_items = tables["order_items"]
    sellers = tables["sellers"]
    products = tables["products"]
    orders = tables["orders"]
    customers = tables["customers"]
    geolocation = tables["geolocation"]

    zip_geo = build_zip_geo_lookup(geolocation)

    items_full = order_items.merge(
        sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
    items_full = items_full.merge(
        products[["product_id", "product_weight_g", "product_photos_qty"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    order_customer_zip = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_zip_code_prefix"]],
        on="customer_id",
        how="left",
    )
    items_full = items_full.merge(order_customer_zip, on="order_id", how="left")

    items_full = items_full.merge(
        zip_geo.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "lat": "seller_lat",
                "lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )
    items_full = items_full.merge(
        zip_geo.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "lat": "customer_lat",
                "lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    items_full["distance_km"] = haversine_km(
        items_full["customer_lat"],
        items_full["customer_lng"],
        items_full["seller_lat"],
        items_full["seller_lng"],
    )
    return items_full


def _aggregate_order_items(items_full: pd.DataFrame) -> pd.DataFrame:
    """STEP 2: collapse item-level rows to one row per order_id."""
    return (
        items_full.groupby("order_id", as_index=False).agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            num_items=("order_item_id", "count"),
            num_sellers=("seller_id", "nunique"),
            seller_states=(
                "seller_state",
                lambda x: ", ".join(sorted(x.dropna().astype(str).unique())),
            ),
            seller_zip_codes=(
                "seller_zip_code_prefix",
                lambda x: ", ".join(sorted(x.dropna().astype(str).unique())),
            ),
            num_products=("product_id", "nunique"),
            total_weight=("product_weight_g", "sum"),
            num_photos=("product_photos_qty", "sum"),
            avg_distance_km=("distance_km", "mean"),
            max_distance_km=("distance_km", "max"),
            min_distance_km=("distance_km", "min"),
        )
    )


def _aggregate_order_payments(order_payments: pd.DataFrame) -> pd.DataFrame:
    """STEP 3: collapse order_payments (one-to-many with orders) to one row/order."""
    return (
        order_payments.groupby("order_id", as_index=False).agg(
            total_payment_value=("payment_value", "sum"),
            num_payment_sequential=("payment_sequential", "max"),
            payment_types=(
                "payment_type",
                lambda x: ", ".join(x.dropna().astype(str).unique()),
            ),
        )
    )


def build_ml_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Full Notebook 1 pipeline: raw tables -> one row per order_id ML table.

    Parameters
    ----------
    tables : dict of raw DataFrames, as returned by src.data.db.read_raw_tables

    Returns
    -------
    pd.DataFrame with exactly one row per order_id (asserted).
    """
    orders = tables["orders"]
    customers = tables["customers"]

    logger.info("Building item-level merge with distances")
    items_full = _merge_item_level(tables)

    logger.info("Aggregating order_items to order level")
    agg_order_items = _aggregate_order_items(items_full)

    logger.info("Aggregating order_payments to order level")
    agg_order_payments = _aggregate_order_payments(tables["order_payments"])

    logger.info("Joining orders + customers + aggregates")
    ml_table = orders.copy()
    assert ml_table["order_id"].is_unique, "orders must contain one row per order_id"

    customer_info = customers.drop_duplicates("customer_id")
    ml_table = ml_table.merge(
        customer_info, on="customer_id", how="left", validate="many_to_one"
    )
    ml_table = ml_table.merge(
        agg_order_items, on="order_id", how="left", validate="one_to_one"
    )
    ml_table = ml_table.merge(
        agg_order_payments, on="order_id", how="left", validate="one_to_one"
    )

    assert ml_table["order_id"].is_unique, "Each order_id must appear exactly once"
    assert len(ml_table) == orders["order_id"].nunique(), (
        "Final ML table must have exactly one row per order"
    )

    logger.info(
        "ML table built: %d rows, %d columns, missing avg_distance_km: %d",
        len(ml_table),
        ml_table.shape[1],
        ml_table["avg_distance_km"].isna().sum(),
    )
    return ml_table