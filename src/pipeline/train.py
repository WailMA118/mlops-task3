"""
Model training pipeline: baseline, Random Forest grid search, final test
evaluation. Refactor of Notebook 6 (task2nb6.ipynb).

Task 3 requirement #5 (experiment tracking & model registry): every run
(baseline + each grid point + the final model) is logged to MLflow with its
parameters, metrics, and artifacts. The selected final model is registered
in the MLflow Model Registry with a version and a stage, so the inference
service loads it from the registry / artifact store rather than a local
notebook folder.

Training-time only. src/inference/predict.py never imports this module.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import get_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def evaluate(model: Any, X: pd.DataFrame, y: pd.Series, label: str = "") -> Dict[str, Any]:
    """Metric set suited to an imbalanced problem — not accuracy alone."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "set": label,
        "f1": round(f1_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "pr_auc": round(average_precision_score(y, y_proba), 4),
    }


def _setup_mlflow() -> None:
    cfg = get_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def train_and_select_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Full Notebook 6 pipeline with MLflow tracking:
      1. baseline DummyClassifier (logged as its own run)
      2. grid search over RandomForestClassifier, tuned on validation F1
         (each combination logged as its own run)
      3. final model evaluated once on test (logged as its own run, and
         registered in the MLflow Model Registry)

    Returns (final_model, test_metrics, results_summary_df).
    """
    cfg = get_config()
    target = cfg.target.column
    random_state = cfg.project.random_state

    X_train, y_train = train_df[feature_list], train_df[target].astype(int)
    X_val, y_val = val_df[feature_list], val_df[target].astype(int)
    X_test, y_test = test_df[feature_list], test_df[target].astype(int)

    logger.info(
        "Train: %s (pos rate %.4f) | Val: %s (pos rate %.4f) | Test: %s (pos rate %.4f)",
        X_train.shape, y_train.mean(),
        X_val.shape, y_val.mean(),
        X_test.shape, y_test.mean(),
    )

    _setup_mlflow()
    results_log = []

    # ---- baseline ----
    with mlflow.start_run(run_name="baseline_dummy"):
        baseline = DummyClassifier(strategy="stratified", random_state=random_state)
        baseline.fit(X_train, y_train)
        baseline_metrics = evaluate(baseline, X_val, y_val, label="baseline_dummy (val)")
        results_log.append(baseline_metrics)
        mlflow.log_param("strategy", "stratified")
        mlflow.log_metrics({k: v for k, v in baseline_metrics.items() if k != "set"})
        logger.info("Baseline (val): %s", baseline_metrics)

    # ---- grid search, tuned on validation F1 ----
    param_grid = {
        "n_estimators": list(cfg.model.param_grid.n_estimators),
        "max_depth": list(cfg.model.param_grid.max_depth),
        "min_samples_leaf": list(cfg.model.param_grid.min_samples_leaf),
    }
    grid_keys = list(param_grid.keys())
    grid_values = list(param_grid.values())

    best_f1 = -1.0
    best_model = None
    best_params: Dict[str, Any] | None = None

    for combo in itertools.product(*grid_values):
        params = dict(zip(grid_keys, combo))
        with mlflow.start_run(run_name=f"rf_{params}"):
            rf = RandomForestClassifier(
                **params,
                class_weight=cfg.model.class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            metrics = evaluate(rf, X_val, y_val, label=f"rf {params} (val)")
            results_log.append(metrics)

            mlflow.log_params(params)
            mlflow.log_param("class_weight", cfg.model.class_weight)
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "set"})
            logger.info("%s", metrics)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = rf
                best_params = params

    logger.info("Best params (by F1 on validation): %s (F1=%.4f)", best_params, best_f1)
    final_model = best_model

    # ---- final test evaluation + registration ----
    with mlflow.start_run(run_name="FINAL_MODEL") as run:
        test_metrics = evaluate(final_model, X_test, y_test, label="FINAL MODEL (test)")
        results_log.append(test_metrics)

        mlflow.log_params(best_params or {})
        mlflow.log_param("class_weight", cfg.model.class_weight)
        mlflow.log_param("tuning_metric", cfg.model.tuning_metric)
        mlflow.log_param("decision_threshold", cfg.model.decision_threshold)
        mlflow.log_metrics({k: v for k, v in test_metrics.items() if k != "set"})

        report = classification_report(y_test, final_model.predict(X_test), digits=4)
        cm = confusion_matrix(y_test, final_model.predict(X_test))
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        logger.info("=== FINAL TEST RESULTS (touched once) ===")
        logger.info("%s", test_metrics)
        logger.info("\n%s", report)

        # serialization_format="pickle": mlflow's default skops serializer rejects
        # RandomForest's internal tree structures as "untrusted types" to load back
        # (a safety check against malicious files). Since we trust our own
        # training output, pickle is the standard, documented way around this.
        model_info = mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        logger.info(
            "Registered model '%s' from run %s",
            cfg.mlflow.registered_model_name,
            run.info.run_id,
        )

        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.search_model_versions(
                f"name='{cfg.mlflow.registered_model_name}'"
            )
            newest = max(latest_versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name=cfg.mlflow.registered_model_name,
                version=newest.version,
                stage=cfg.mlflow.registry_stage,
                archive_existing_versions=True,
            )
            logger.info(
                "Transitioned model version %s to stage '%s'",
                newest.version,
                cfg.mlflow.registry_stage,
            )
        except Exception:
            logger.exception(
                "Could not transition model stage automatically; "
                "set it manually in the MLflow UI/registry."
            )

    results_summary = pd.DataFrame(results_log)
    return final_model, test_metrics, results_summary


def save_local_model_artifacts(
    final_model: Any,
    results_summary: pd.DataFrame,
    feature_list: list[str],
    best_params: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> None:
    """
    Also persist local copies of the model + summary, mirroring Notebook 6's
    'model_artifacts/' output. MLflow's registry remains the source of truth
    for what the API loads at inference (see src/inference/predict.py); these
    local files are kept for parity with the notebooks and for quick local
    debugging.
    """
    cfg = get_config()
    model_dir = resolve_path(cfg.paths.model.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, resolve_path(cfg.paths.model.final_model))
    results_summary.to_csv(resolve_path(cfg.paths.model.results_summary), index=False)

    feature_importances = (
        pd.Series(final_model.feature_importances_, index=feature_list)
        .sort_values(ascending=False)
    )
    feature_importances.to_csv(
        resolve_path(cfg.paths.model.feature_importances), header=["importance"]
    )

    with open(resolve_path(cfg.paths.model.model_config), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": best_params,
                "class_weight": cfg.model.class_weight,
                "tuning_metric": cfg.model.tuning_metric,
                "decision_threshold": cfg.model.decision_threshold,
                "final_test_metrics": test_metrics,
            },
            f,
            indent=2,
        )

    logger.info("Saved local model artifacts to %s", model_dir)
