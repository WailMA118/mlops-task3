"""
EDA orchestrator. Ties src/pipeline/eda_summaries.py, eda_relations.py,
eda_feature_selection.py, and eda_plots.py together into the same sequence
as Notebook 4 (task2nb4.ipynb): schema/shape -> missing values -> numeric
analysis -> categorical analysis -> label relations -> date analysis ->
geography analysis -> feature-selection rationale -> findings summary.

This module writes every report/plot artifact under config.yaml's
paths.eda.*, exactly mirroring what Notebook 4 wrote to eda_artifacts/, but
path-driven instead of a relative cwd-dependent folder.

EDA is diagnostic/documentation output, not part of the inference path. It
runs against paths.data.train only (EDA never opens val/test, same as the
notebook), and src/inference never imports anything from this module.
"""

from __future__ import annotations

import pandas as pd

from src.pipeline.eda_feature_selection import build_feature_selection_table
from src.pipeline.eda_plots import (
    plot_categorical_top_values,
    plot_date_analysis,
    plot_geography_analysis,
    plot_numerical_distributions_and_outliers,
    plot_relations_and_correlations,
)
from src.pipeline.eda_relations import (
    add_date_derived_columns,
    add_same_state_column,
    date_relation_reports,
    geography_reports,
    label_relation_reports,
)
from src.pipeline.eda_summaries import (
    categorical_summary_report,
    columns_of_type,
    dataset_shape_and_memory,
    invalid_negative_values,
    missing_value_report,
    numeric_summary_report,
)
from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def run_eda(df: pd.DataFrame) -> dict:
    """
    Full Notebook 4 pipeline against an already-loaded DataFrame (the train
    split). Writes every report/plot artifact to disk and returns a dict of
    the key summary objects, mainly for tests/inspection.
    """
    cfg = get_config()
    eda_cfg = cfg.eda
    target_col = cfg.target.column
    column_types = dict(eda_cfg.column_types)

    report_dir = resolve_path(eda_cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---- schema, shape, memory ----
    shape_info = dataset_shape_and_memory(df)
    logger.info(
        "Training data shape: %d rows, %d columns, memory: %.2f MB",
        shape_info["n_rows"],
        shape_info["n_columns"],
        shape_info["memory_usage_mb"],
    )

    numeric_columns = columns_of_type(column_types, "numerical", df)
    numeric_columns = [c for c in numeric_columns if c != target_col]
    categorical_columns = columns_of_type(column_types, "categorical", df)
    date_columns = columns_of_type(column_types, "date", df)

    # ---- missing values ----
    missing_df = missing_value_report(df, dict(eda_cfg.meaningful_missing_columns))
    missing_df.to_csv(resolve_path(cfg.paths.eda.missing_summary), index=False)
    logger.info("Missing-value report written (%d columns)", len(missing_df))

    # ---- numeric analysis ----
    numeric_summary = numeric_summary_report(
        df, numeric_columns, float(eda_cfg.outlier_iqr_multiplier)
    )
    numeric_summary.to_csv(resolve_path(cfg.paths.eda.numeric_summary))

    invalid_ranges = invalid_negative_values(
        df, numeric_columns, list(eda_cfg.non_negative_keywords)
    )
    for column, invalid_count in invalid_ranges.items():
        if invalid_count:
            logger.warning("%s: %d negative values found", column, invalid_count)

    plot_numerical_distributions_and_outliers(
        df, numeric_columns, resolve_path(cfg.paths.eda.numerical_distributions_plot)
    )

    # ---- categorical analysis ----
    categorical_summary_df, messy_values_df = categorical_summary_report(
        df, categorical_columns, float(eda_cfg.rare_category_threshold)
    )
    categorical_summary_df.to_csv(resolve_path(cfg.paths.eda.categorical_summary), index=False)
    messy_values_df.to_csv(resolve_path(cfg.paths.eda.messy_values), index=False)

    for column in categorical_columns:
        plot_categorical_top_values(
            df,
            column,
            int(eda_cfg.top_categories_to_plot),
            resolve_path(eda_cfg.report_dir) / f"categorical_{column}.png",
        )

    # ---- relations with the label ----
    relations = label_relation_reports(df, numeric_columns, target_col)
    relations["status_relation"].to_csv(resolve_path(cfg.paths.eda.status_relation), index=False)
    relations["payment_relation"].to_csv(resolve_path(cfg.paths.eda.payment_relation), index=False)
    relations["cross_tab"].to_csv(resolve_path(cfg.paths.eda.status_label_crosstab))
    relations["correlation_matrix"].to_csv(resolve_path(cfg.paths.eda.correlation_matrix))

    plot_relations_and_correlations(
        relations["status_relation"],
        relations["payment_relation"],
        relations["correlation_matrix"],
        resolve_path(cfg.paths.eda.relations_and_correlations_plot),
    )

    # ---- date analysis ----
    date_df = add_date_derived_columns(df, date_columns)
    date_reports = date_relation_reports(date_df, target_col)
    logger.info(
        "Purchase date range: %s to %s | median delivery: %.2f days | median delay vs estimate: %.2f days",
        date_reports["purchase_date_min"],
        date_reports["purchase_date_max"],
        date_reports["median_delivery_days"],
        date_reports["median_delivery_delay_days"],
    )

    plot_date_analysis(
        date_reports["monthly_late_rate"],
        date_reports["weekday_late_rate"],
        date_reports["holiday_late_rate"],
        date_df["delivery_days"],
        resolve_path(cfg.paths.eda.date_analysis_plot),
    )

    # ---- geography analysis ----
    geo_df = add_same_state_column(df)
    geo_reports = geography_reports(
        geo_df,
        target_col,
        int(eda_cfg.min_state_orders_for_ranking),
        list(eda_cfg.distance_bucket_edges),
        list(eda_cfg.distance_bucket_labels),
    )
    geo_reports["distance_late_rate"].to_csv(resolve_path(cfg.paths.eda.distance_late_rate))
    geo_reports["distance_corr"].to_csv(resolve_path(cfg.paths.eda.distance_corr))
    logger.info(
        "Missing avg_distance_km: %d / %d | customer zip cardinality: %d | seller zip cardinality: %d",
        geo_reports["missing_avg_distance_km"],
        len(geo_df),
        geo_reports["customer_zip_cardinality"],
        geo_reports["seller_zip_cardinality"],
    )

    plot_geography_analysis(
        geo_df["avg_distance_km"],
        geo_reports["distance_late_rate"],
        geo_reports["top_states_by_late_rate"],
        resolve_path(cfg.paths.eda.geography_analysis_plot),
    )

    # ---- feature-selection rationale (documentation artifact) ----
    feature_selection_df = build_feature_selection_table()
    feature_selection_df.to_csv(resolve_path(cfg.paths.eda.feature_selection), index=False)

    # ---- findings summary ----
    holiday_rate_true = date_reports["holiday_late_rate"].get(True, float("nan"))
    holiday_rate_false = date_reports["holiday_late_rate"].get(False, float("nan"))
    findings = [
        f"EDA uses {cfg.eda.input_split}.parquet only; other splits were not opened.",
        f"Training shape: {shape_info['n_rows']:,} rows and {shape_info['n_columns']} columns.",
        "Customer city and ZIP prefixes have high cardinality and need careful encoding.",
        f"Median delivery time: {date_reports['median_delivery_days']:.2f} days.",
        "Numerical features are right-skewed and contain IQR outliers.",
        f"Holiday late rate: {holiday_rate_true:.2f}% versus {holiday_rate_false:.2f}% on non-holidays.",
        "Longer customer-seller distance is associated with a higher late rate.",
        "Use these findings to guide imputation, encoding, outlier treatment, and model selection.",
    ]
    resolve_path(cfg.paths.eda.findings_summary).write_text(
        "\n".join(findings), encoding="utf-8"
    )
    logger.info("Saved EDA artifacts to %s", report_dir)

    return {
        "shape_info": shape_info,
        "missing_df": missing_df,
        "numeric_summary": numeric_summary,
        "categorical_summary_df": categorical_summary_df,
        "messy_values_df": messy_values_df,
        "relations": relations,
        "date_reports": date_reports,
        "geo_reports": geo_reports,
        "feature_selection_df": feature_selection_df,
        "findings": findings,
    }