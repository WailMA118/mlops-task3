from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderRequest,
    PredictionResponse,
)
from src.inference.predict import InferencePipeline, PredictionResult
from src.utils.config import get_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the fitted feature artifacts and registered model once at startup."""
    app.state.pipeline = None
    try:
        app.state.pipeline = InferencePipeline()
        logger.info("FastAPI inference service started successfully")
    except Exception:
        logger.exception("FastAPI startup could not load the inference pipeline")
        # Keep the process alive so /health can report an unhealthy service.
        # Prediction endpoints return a clear 503 until the dependency is fixed.
    yield


cfg = get_config()

app = FastAPI(
    title=cfg.api.title,
    version=cfg.api.version,
    description=(
        "Inference API for the Olist late-delivery Random Forest model. "
        "The API loads fitted feature transformers and the registered model; "
        "it never fits a model during inference."
    ),
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a stable, explicit validation error shape for bad payloads."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
        },
    )


def _pipeline(request: Request) -> InferencePipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not loaded. Check MLflow and model artifacts.",
        )
    return pipeline


def _to_response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        prediction=result.is_late,
        probability=result.probability_late,
        model_version=result.model_version,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
)
def health(request: Request) -> HealthResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        model_loaded=pipeline is not None,
        model_version=(pipeline.loaded_model.version if pipeline else None),
    )


@app.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["model"],
    summary="Model information and version",
)
def model_info(request: Request) -> ModelInfoResponse:
    pipeline = _pipeline(request)
    return ModelInfoResponse(
        model_name=cfg.mlflow.registered_model_name,
        model_version=pipeline.loaded_model.version,
        model_source=pipeline.loaded_model.source,
        registry_stage=cfg.mlflow.registry_stage,
        decision_threshold=pipeline.threshold,
        feature_count=len(pipeline.feature_artifacts.final_feature_list),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["model"],
    summary="Model information and version (alias)",
    include_in_schema=True,
)
def model_info_alias(request: Request) -> ModelInfoResponse:
    return model_info(request)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    summary="Predict one order",
)
def predict(order: OrderRequest, request: Request) -> PredictionResponse:
    pipeline = _pipeline(request)
    try:
        return _to_response(pipeline.predict_one(order.model_dump(mode="python")))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction input could not be processed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected single-order prediction failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed unexpectedly.",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    summary="Predict a batch of orders",
)
def predict_batch(
    payload: BatchPredictRequest, request: Request
) -> BatchPredictionResponse:
    pipeline = _pipeline(request)
    predictions: list[PredictionResponse] = []

    for index, order in enumerate(payload.orders):
        try:
            result = pipeline.predict_one(order.model_dump(mode="python"))
            predictions.append(_to_response(result))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Order at index {index} could not be processed: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected batch prediction failure at index %d", index)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed for order at index {index}.",
            ) from exc

    return BatchPredictionResponse(predictions=predictions)
