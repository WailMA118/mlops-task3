"""
Inference pipeline. This is the module Task 3 is actually about.

Rules enforced here:
  - Nothing is fitted. Every transformer and the model are loaded from disk
    (feature artifacts) / the MLflow registry (model), exactly as saved by
    the training notebooks/pipeline.
  - The same derived-feature and encoding logic as training is reused via
    src/features/derived_features.py and src/features/build_features.py, so
    the pipeline reproduces notebook output exactly on the same input.
  - Bad input / missing values are handled without crashing the service:
    validation happens upstream (src/validation) before this module is
    called, and this module additionally guards against missing optional
    columns.

Input contract: a single order (dict/Series) or a batch (DataFrame) of
already order-aggregated rows with the same raw columns produced by
Notebook 1 / src/data/build_ml_table.py (e.g. total_price, total_freight,
num_items, customer_state, seller_states, order_purchase_timestamp,
payment_types, ...). This module does not re-run the raw SQL joins; it
starts from order-level rows, which is what a "new order" request to the
API supplies (see app/schemas.py for the exact request schema).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from src.features.build_features import (
    FittedFeatureArtifacts,
    load_feature_artifacts,
    select_raw_features,
    transform_feature_table,
)
from src.features.derived_features import add_derived_features
from src.features.shipping_pressure import compute_pressure_for_new_order
from src.inference.model_loader import LoadedModel, load_registered_model
from src.utils.config import get_config
from src.utils.logging_setup import get_logger, get_prediction_logger

logger = get_logger(__name__)
prediction_logger = get_prediction_logger()


@dataclass
class PredictionResult:
    is_late: int
    probability_late: float
    model_version: str
    latency_ms: float


class InferencePipeline:
    """
    Holds the loaded (never-fitted-here) feature artifacts and model in
    memory so a FastAPI app can build one instance at startup and reuse it
    across requests, instead of hitting disk/registry on every call.
    """

    def __init__(
        self,
        feature_artifacts: FittedFeatureArtifacts | None = None,
        loaded_model: LoadedModel | None = None,
        reference_log: pd.DataFrame | None = None,
    ) -> None:
        self.feature_artifacts = feature_artifacts or load_feature_artifacts()
        self.loaded_model = loaded_model or load_registered_model()
        self.reference_log = (
            reference_log if reference_log is not None else _load_reference_log()
        )
        cfg = get_config()
        self.target_col = cfg.target.column
        self.threshold = float(cfg.model.decision_threshold)
        logger.info(
            "InferencePipeline ready (model_version=%s)", self.loaded_model.version
        )

    def _prepare_single_order(self, order: Dict[str, Any]) -> pd.DataFrame:
        """
        Turn one raw order dict into a one-row DataFrame with derived +
        shipping-pressure features, ready for select_raw_features/transform.
        """
        row = pd.Series(order)
        df = pd.DataFrame([row])

        df = add_derived_features(df)

        pressure = compute_pressure_for_new_order(
            new_order=df.iloc[0], reference_log=self.reference_log
        )
        df["customer_state_pressure_5d"] = pressure["customer_state_pressure_5d"]
        df["seller_pressure_5d"] = pressure["seller_pressure_5d"]

        return df

    def predict_one(self, order: Dict[str, Any]) -> PredictionResult:
        """Predict late/on-time + probability for a single new order."""
        start = time.perf_counter()
        try:
            df = self._prepare_single_order(order)
            raw_table = select_raw_features(df)
            feature_table = transform_feature_table(raw_table, self.feature_artifacts)
            feature_table = feature_table[self.feature_artifacts.final_feature_list]

            proba = float(self.loaded_model.model.predict_proba(feature_table)[0, 1])
            pred = int(proba >= self.threshold)
            latency_ms = (time.perf_counter() - start) * 1000

            result = PredictionResult(
                is_late=pred,
                probability_late=proba,
                model_version=self.loaded_model.version,
                latency_ms=round(latency_ms, 3),
            )
            self._log_prediction(order, result, error=None)
            return result
        except Exception as exc:  # noqa: BLE001 - log then re-raise for the API layer
            latency_ms = (time.perf_counter() - start) * 1000
            self._log_prediction(order, None, error=str(exc), latency_ms=latency_ms)
            logger.exception("Prediction failed")
            raise

    def predict_batch(self, orders: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Predict for a batch of new orders. Errors on one row don't abort the batch."""
        results: List[PredictionResult] = []
        for order in orders:
            try:
                results.append(self.predict_one(order))
            except Exception:  # noqa: BLE001 - already logged in predict_one
                results.append(
                    PredictionResult(
                        is_late=-1,
                        probability_late=float("nan"),
                        model_version=self.loaded_model.version,
                        latency_ms=0.0,
                    )
                )
        return results

    def _log_prediction(
        self,
        order: Dict[str, Any],
        result: PredictionResult | None,
        error: str | None,
        latency_ms: float | None = None,
    ) -> None:
        """Structured one-line log per prediction request, for monitoring."""
        payload = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "input": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                      for k, v in order.items()},
            "model_version": self.loaded_model.version,
            "error": error,
        }
        if result is not None:
            payload.update(
                {
                    "prediction": result.is_late,
                    "probability_late": result.probability_late,
                    "latency_ms": result.latency_ms,
                }
            )
        elif latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 3)

        import json as _json

        prediction_logger.info(_json.dumps(payload, default=str))


def _load_reference_log() -> pd.DataFrame:
    """
    Load the historical order log (timestamp + customer_state + seller_states)
    used to compute causal shipping-pressure features for new orders at
    inference time. Backed by the training feature table's source data so
    inference never needs live DB access for this.
    """
    cfg = get_config()
    from src.utils.config import resolve_path

    frames = []
    for path_key in ("train", "val", "test"):
        path = resolve_path(cfg.paths.data[path_key])
        if path.exists():
            cols = ["order_purchase_timestamp", "customer_state", "seller_states"]
            frames.append(pd.read_parquet(path, columns=cols))
    if not frames:
        logger.warning(
            "No reference log parquet files found; shipping-pressure features "
            "will be NaN and median-imputed at inference time."
        )
        return pd.DataFrame(
            columns=["order_purchase_timestamp", "customer_state", "seller_states"]
        )
    reference_log = pd.concat(frames, ignore_index=True)
    reference_log["order_purchase_timestamp"] = pd.to_datetime(
        reference_log["order_purchase_timestamp"]
    )
    return reference_log
