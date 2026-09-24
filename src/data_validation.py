"""Great Expectations validation for order-level inference data.

The inference contract is the order-level table produced by
``src.data.build_ml_table``.  Schema and key violations are rejected.  Data
quality anomalies in recoverable feature columns are logged and replaced with
training medians so that a single malformed value cannot poison a prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.core import ExpectationConfiguration, ExpectationSuite
from src.utils.config import get_config, resolve_path

logger = logging.getLogger(__name__)

SUITE_NAME = "olist_order_inference"
STATE_CODES = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
}
PAYMENT_TYPES = {"boleto", "credit_card", "debit_card", "voucher", "not_defined"}

IDENTIFIER_COLUMNS = ("order_id", "customer_id")
DATETIME_COLUMNS = ("order_purchase_timestamp",)
OPTIONAL_DATETIME_COLUMNS = (
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)
STRING_COLUMNS = ("customer_state", "seller_states", "payment_types")
NUMERIC_COLUMNS = (
    "total_price",
    "total_freight",
    "num_items",
    "num_sellers",
    "num_products",
    "total_weight",
    "num_payment_sequential",
    "avg_distance_km",
    "max_distance_km",
)
NON_NEGATIVE_COLUMNS = NUMERIC_COLUMNS
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + DATETIME_COLUMNS + STRING_COLUMNS + NUMERIC_COLUMNS


class CriticalValidationError(ValueError):
    """Raised when a payload cannot safely enter the inference pipeline."""


@dataclass
class ValidationResult:
    """Validation outcome and the safe DataFrame to pass to inference."""

    dataframe: pd.DataFrame
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.critical_failures


def _expectation(expectation_type: str, **kwargs: Any) -> ExpectationConfiguration:
    meta = kwargs.pop("meta", {})
    return ExpectationConfiguration(expectation_type=expectation_type, kwargs=kwargs, meta=meta)


def build_expectation_suite() -> ExpectationSuite:
    """Build the versioned GX suite for the raw order-level inference contract."""
    expectations: list[ExpectationConfiguration] = []
    for column in REQUIRED_COLUMNS:
        expectations.append(
            _expectation("expect_column_to_exist", column=column, meta={"severity": "critical"})
        )
    for column in IDENTIFIER_COLUMNS + DATETIME_COLUMNS + STRING_COLUMNS:
        expectations.append(
            _expectation(
                "expect_column_values_to_not_be_null",
                column=column,
                meta={"severity": "critical"},
            )
        )
    for column in NUMERIC_COLUMNS:
        expectations.extend(
            [
                _expectation(
                    "expect_column_values_to_be_of_type",
                    column=column,
                    type_="float64",
                    meta={"severity": "warning"},
                ),
                _expectation(
                    "expect_column_values_to_be_between",
                    column=column,
                    min_value=0,
                    mostly=0.99,
                    meta={"severity": "warning"},
                ),
            ]
        )
    expectations.extend(
        [
            _expectation(
                "expect_column_values_to_be_in_set",
                column="customer_state",
                value_set=sorted(STATE_CODES),
                mostly=0.99,
                meta={"severity": "warning"},
            ),
        ]
    )
    return ExpectationSuite(expectation_suite_name=SUITE_NAME, expectations=expectations)


def save_expectation_suite(path: str | Path | None = None) -> Path:
    """Persist the suite as JSON so CI and GX tooling can inspect it."""
    if path is None:
        path = resolve_path(f"great_expectations/{SUITE_NAME}.json")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_expectation_suite().to_json_dict(), indent=2), encoding="utf-8")
    return output


def _load_or_build_suite() -> ExpectationSuite:
    path = resolve_path(f"great_expectations/{SUITE_NAME}.json")
    if not path.exists():
        return build_expectation_suite()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExpectationSuite(**payload)


def _training_medians() -> dict[str, float]:
    cfg = get_config()
    train_path = resolve_path(cfg.paths.data.train)
    if not train_path.exists():
        return {column: 0.0 for column in NUMERIC_COLUMNS}
    train = pd.read_parquet(train_path, columns=list(NUMERIC_COLUMNS))
    return {column: float(train[column].median()) for column in NUMERIC_COLUMNS}


def _run_gx(df: pd.DataFrame, suite: ExpectationSuite) -> list[str]:
    dataset = gx.from_pandas(df, expectation_suite=suite)
    result = dataset.validate()
    return [
        str(item.expectation_config.kwargs.get("column", item.expectation_config.expectation_type))
        for item in result.results
        if not item.success
    ]


def _critical_checks(df: pd.DataFrame) -> list[str]:
    failures = [f"missing required column: {column}" for column in REQUIRED_COLUMNS if column not in df]
    if failures:
        return failures
    for column in IDENTIFIER_COLUMNS:
        if df[column].isna().any() or df[column].duplicated().any() and column == "order_id":
            failures.append(f"invalid primary key column: {column}")
    for column in DATETIME_COLUMNS:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.isna().any():
            failures.append(f"invalid datetime values: {column}")
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            failures.append(f"invalid numeric dtype: {column}")
    return failures


def _warning_checks(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for column in NON_NEGATIVE_COLUMNS:
        invalid = pd.to_numeric(df[column], errors="coerce").lt(0) | df[column].isna()
        if invalid.any():
            warnings.append(f"invalid or missing non-negative values: {column} ({int(invalid.sum())})")
    for column in ("customer_state", "payment_types"):
        if column not in df:
            continue
        values = df[column].dropna().astype(str)
        if column == "customer_state":
            invalid = ~values.isin(STATE_CODES)
        else:
            invalid = ~values.map(lambda value: set(value.split(", ")).issubset(PAYMENT_TYPES))
        if invalid.any():
            warnings.append(f"unseen categorical values: {column} ({int(invalid.sum())})")
    if {"max_distance_km", "avg_distance_km"}.issubset(df):
        invalid = pd.to_numeric(df["max_distance_km"], errors="coerce") < pd.to_numeric(
            df["avg_distance_km"], errors="coerce"
        )
        if invalid.any():
            warnings.append(f"inconsistent distance ordering ({int(invalid.sum())})")
    date_columns = [column for column in DATETIME_COLUMNS + OPTIONAL_DATETIME_COLUMNS if column in df]
    parsed_dates = {column: pd.to_datetime(df[column], errors="coerce") for column in date_columns}
    for column, parsed in parsed_dates.items():
        invalid = df[column].notna() & parsed.isna()
        if invalid.any():
            warnings.append(f"invalid datetime values: {column} ({int(invalid.sum())})")
    ordering = (
        ("order_purchase_timestamp", "order_approved_at"),
        ("order_purchase_timestamp", "order_delivered_carrier_date"),
        ("order_delivered_carrier_date", "order_delivered_customer_date"),
        ("order_purchase_timestamp", "order_delivered_customer_date"),
        ("order_delivered_customer_date", "order_estimated_delivery_date"),
    )
    for earlier, later in ordering:
        if earlier not in parsed_dates or later not in parsed_dates:
            continue
        invalid = parsed_dates[earlier].notna() & parsed_dates[later].notna() & (
            parsed_dates[later] < parsed_dates[earlier]
        )
        if invalid.any():
            warnings.append(f"invalid datetime ordering: {later} before {earlier} ({int(invalid.sum())})")
    return warnings


def _apply_fallbacks(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    clean = df.copy()
    medians = _training_medians()
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = values.where(values.ge(0), medians[column]).fillna(medians[column])
    clean["customer_state"] = clean["customer_state"].where(
        clean["customer_state"].isin(STATE_CODES), "missing"
    )
    payment_values = clean["payment_types"].fillna("missing").astype(str)
    clean["payment_types"] = payment_values.map(
        lambda value: ", ".join(p for p in value.split(", ") if p in PAYMENT_TYPES) or "missing"
    )
    for warning in warnings:
        logger.warning("Input validation warning: %s", warning)
    return clean


def validate_dataframe(df: pd.DataFrame, *, reject_on_critical: bool = True) -> ValidationResult:
    """Validate and sanitize an inference DataFrame.

    Critical schema/key/type failures raise ``CriticalValidationError`` by
    default. Warning-level anomalies are logged and repaired in the returned
    copy using training medians and the encoder's ``missing`` category.
    """
    if not isinstance(df, pd.DataFrame):
        raise CriticalValidationError("input must be a pandas DataFrame")
    critical = _critical_checks(df)
    if critical and reject_on_critical:
        raise CriticalValidationError("; ".join(critical))
    if critical:
        return ValidationResult(df.copy(), critical_failures=critical)
    warnings = _warning_checks(df)
    gx_failures = _run_gx(df, _load_or_build_suite())
    warnings.extend(f"Great Expectations failure: {failure}" for failure in gx_failures)
    return ValidationResult(_apply_fallbacks(df, warnings), warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    """Validate a Parquet file and print a concise human-readable result."""
    parser = argparse.ArgumentParser(description="Validate an Olist order-level Parquet file.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Parquet path; defaults to paths.data.ml_table from config.yaml.",
    )
    args = parser.parse_args(argv)
    default_path = resolve_path(get_config().paths.data.ml_table)
    path = Path(args.path) if args.path else default_path
    if not path.is_absolute():
        path = resolve_path(str(path))

    try:
        result = validate_dataframe(pd.read_parquet(path))
    except CriticalValidationError as exc:
        print(f"REJECTED: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI should report input failures cleanly
        print(f"ERROR: {exc}")
        return 2

    status = "PASSED" if not result.warnings else "PASSED WITH WARNINGS"
    print(f"{status}: {path}")
    print(f"Rows checked: {len(result.dataframe)}")
    print(f"Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"- {warning}")
    return 0


__all__ = [
    "CriticalValidationError",
    "ValidationResult",
    "build_expectation_suite",
    "save_expectation_suite",
    "validate_dataframe",
]


if __name__ == "__main__":
    sys.exit(main())
