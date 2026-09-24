"""
Feature-selection rationale table. Refactor of Notebook 4's "features
selection" cell: the documented reasoning connecting each EDA finding to a
feature carried into src/features (Notebook 5's production feature set).

This is a documentation artifact, not a computation step — the actual
feature list the model trains on lives in config.yaml (features.*), and this
table exists purely to explain, for each of those features, what it
represents, why the EDA above justified including it, and how it is derived.
Keeping this in code (vs. only prose in a notebook) means it is versioned,
diffable, and regenerated as a CSV report artifact alongside the rest of the
EDA outputs.

IMPORTANT: delivery_delay_days is listed here for reference only, flagged as
excluded. It is how `is_late` itself is computed (see src/data/labeling.py)
and must never appear in config.yaml's features.* lists — including it as a
model input would leak the label.
"""

from __future__ import annotations

import pandas as pd

FEATURE_SELECTION_RATIONALE: list[dict[str, object]] = [
    {
        "feature": "total_price",
        "represents": "Sum of item prices for the order.",
        "why_selected": (
            "Right-skewed with real outliers (see numeric summary); "
            "higher-value orders may ship differently."
        ),
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(sum of order_items.price per order_id)."
        ),
    },
    {
        "feature": "total_freight",
        "represents": "Sum of freight/shipping cost for the order.",
        "why_selected": (
            "Freight cost is a proxy for shipping method/distance and showed a "
            "non-trivial correlation with is_late."
        ),
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(sum of order_items.freight_value per order_id)."
        ),
    },
    {
        "feature": "num_items",
        "represents": "Number of items in the order.",
        "why_selected": "More items can mean more picking/packing time and more chances for delay.",
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(count of order_items rows per order_id)."
        ),
    },
    {
        "feature": "num_sellers",
        "represents": "Number of distinct sellers fulfilling the order.",
        "why_selected": (
            "Multi-seller orders depend on every seller shipping on time, so this "
            "is a plausible risk factor for delay."
        ),
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(nunique of seller_id per order_id)."
        ),
    },
    {
        "feature": "num_products",
        "represents": "Number of distinct products in the order.",
        "why_selected": (
            "Similar reasoning to num_items -- more distinct products can mean "
            "more sourcing/packing complexity."
        ),
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(nunique of product_id per order_id)."
        ),
    },
    {
        "feature": "total_weight",
        "represents": "Total weight (grams) of all items in the order.",
        "why_selected": (
            "Heavier shipments can require different carriers/handling; "
            "distribution and outliers were checked in the numeric summary."
        ),
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(sum of product_weight_g per order_id)."
        ),
    },
    {
        "feature": "avg_distance_km",
        "represents": "Average real-world distance between the customer and the order's seller(s).",
        "why_selected": (
            "Geography analysis showed a higher late rate at greater distance "
            "buckets and a positive correlation with is_late."
        ),
        "derived": True,
        "how_to_derive": (
            "src/data/geo.py: haversine distance per item (seller zip vs customer "
            "zip coordinates), averaged per order_id in "
            "src/data/build_ml_table.py."
        ),
    },
    {
        "feature": "max_distance_km",
        "represents": "Distance to the farthest seller in the order.",
        "why_selected": (
            "An order is only complete once its farthest seller's items arrive, "
            "so this can matter more than the average."
        ),
        "derived": True,
        "how_to_derive": (
            "Same per-item haversine distance as avg_distance_km "
            "(src/data/geo.py), aggregated with max instead of mean, per order_id."
        ),
    },
    {
        "feature": "same_state",
        "represents": "Whether at least one seller is in the same state as the customer.",
        "why_selected": (
            "Geography analysis showed a different late rate for same-state vs "
            "different-state shipments; a cheap, low-cardinality complement to "
            "the distance features."
        ),
        "derived": True,
        "how_to_derive": (
            "src/features/derived_features.py: customer_state in "
            "seller_states.split(', ') -- boolean, computed from two existing "
            "columns, no new data needed."
        ),
    },
    {
        "feature": "delivery_delay_days (reference only, NOT a model input)",
        "represents": (
            "Difference in days between the actual and estimated delivery date "
            "-- this is how is_late itself is built."
        ),
        "why_selected": (
            "Not selected as a feature: it leaks the label. Listed here only to "
            "flag it must stay excluded from config.yaml's features.* lists."
        ),
        "derived": True,
        "how_to_derive": "N/A -- excluded on purpose. See src/data/labeling.py.",
    },
    {
        "feature": "purchase_month",
        "represents": "Calendar month the order was purchased in.",
        "why_selected": (
            "Date analysis showed the late rate moves across months "
            "(seasonality, e.g. holiday shopping peaks)."
        ),
        "derived": True,
        "how_to_derive": "src/features/derived_features.py: order_purchase_timestamp.dt.month.",
    },
    {
        "feature": "purchase_weekday",
        "represents": "Day of week the order was purchased.",
        "why_selected": "Date analysis showed a weekday effect on the late rate.",
        "derived": True,
        "how_to_derive": (
            "src/features/derived_features.py: "
            "order_purchase_timestamp.dt.day_name()."
        ),
    },
    {
        "feature": "is_holiday",
        "represents": "Whether the order was purchased on a Brazilian public holiday.",
        "why_selected": "Date analysis showed a different late rate on holidays vs regular days.",
        "derived": True,
        "how_to_derive": (
            "src/features/derived_features.py: "
            "order_purchase_timestamp.dt.normalize().isin(holidays.Brazil(...))."
        ),
    },
    {
        "feature": "order_status",
        "represents": "Order status at the time the data was extracted (e.g. delivered, canceled).",
        "why_selected": (
            "Relations analysis showed a different late rate by status; almost "
            "all rows are 'delivered' after the Notebook 2 filtering, but this "
            "column is dropped from the final feature set for label-leakage "
            "reasons -- see src/data/labeling.py (undelivered orders are "
            "dropped before is_late is computed, so order_status becomes "
            "near-constant and post-outcome for the rows that remain)."
        ),
        "derived": False,
        "how_to_derive": "Raw column from orders. NOT included in config.yaml's features.*.",
    },
    {
        "feature": "payment_types",
        "represents": "Payment method(s) used for the order (e.g. credit_card, boleto, voucher).",
        "why_selected": "Relations analysis showed the late rate varies by payment type.",
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(aggregated unique payment_type values per order_id); multi-hot "
            "encoded in src/features/build_features.py since an order can have "
            "more than one payment type."
        ),
    },
    {
        "feature": "num_payment_sequential",
        "represents": "Number of payment installments/methods used for the order.",
        "why_selected": (
            "A simple numeric complement to payment_types; more installments "
            "could reflect order value or customer behavior linked to delay."
        ),
        "derived": False,
        "how_to_derive": (
            "Raw column from Notebook 1 / src/data/build_ml_table.py "
            "(count of order_payments rows per order_id)."
        ),
    },
    {
        "feature": "customer_state",
        "represents": "State of the customer.",
        "why_selected": "Geography analysis showed the late rate differs meaningfully across states.",
        "derived": False,
        "how_to_derive": (
            "Raw column from customers. High-cardinality-safe (27 Brazilian "
            "states), unlike zip prefix."
        ),
    },
    {
        "feature": "customer_state_pressure_5d",
        "represents": "Trailing 5-day causal order count at the customer-state level.",
        "why_selected": (
            "Added after the original Notebook 4 EDA, in response to the F1 "
            "plateau diagnosed in Notebook 6 -- see approach-and-tools notes: "
            "existing features had weak correlation with is_late (max 0.072), "
            "so shipping-pressure features were introduced as a proxy for "
            "operational load."
        ),
        "derived": True,
        "how_to_derive": "src/features/shipping_pressure.py (causal, via np.searchsorted).",
    },
    {
        "feature": "seller_pressure_5d",
        "represents": "Trailing 5-day causal order count at the seller-state level.",
        "why_selected": "Same rationale as customer_state_pressure_5d, at the seller side.",
        "derived": True,
        "how_to_derive": "src/features/shipping_pressure.py (causal, via np.searchsorted).",
    },
]


def build_feature_selection_table() -> pd.DataFrame:
    """Return the feature-selection rationale as a DataFrame for the EDA report."""
    return pd.DataFrame(FEATURE_SELECTION_RATIONALE)