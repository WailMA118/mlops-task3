from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


NonNegativeFloat = Annotated[float | None, Field(ge=0)]


class OrderRequest(BaseModel):
    """Validated order-level payload expected by the inference pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "order_id": "00010242-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "customer_id": "3ce4360f-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "order_purchase_timestamp": "2017-01-01T10:30:00",
                "customer_state": "SP",
                "seller_states": "SP",
                "payment_types": "credit_card",
                "total_price": 58.90,
                "total_freight": 13.29,
                "num_items": 1,
                "num_sellers": 1,
                "num_products": 1,
                "total_weight": 500.0,
                "num_payment_sequential": 1,
                "avg_distance_km": 10.5,
                "max_distance_km": 10.5,
            }
        },
    )

    order_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    order_purchase_timestamp: datetime
    customer_state: str = Field(min_length=2, max_length=2)
    seller_states: str = Field(min_length=1)
    payment_types: str = Field(min_length=1)

    total_price: NonNegativeFloat
    total_freight: NonNegativeFloat
    num_items: NonNegativeFloat
    num_sellers: NonNegativeFloat
    num_products: NonNegativeFloat
    total_weight: NonNegativeFloat
    num_payment_sequential: NonNegativeFloat
    avg_distance_km: NonNegativeFloat
    max_distance_km: NonNegativeFloat

    @field_validator("customer_state")
    @classmethod
    def normalize_customer_state(cls, value: str) -> str:
        return value.strip().upper()


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 0,
                "probability": 0.1734,
                "model_version": "7",
            }
        }
    )

    prediction: int = Field(ge=0, le=1, description="0 = on time, 1 = late")
    probability: float = Field(ge=0, le=1)
    model_version: str


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "orders": [
                    {
                        "order_id": "order-001",
                        "customer_id": "customer-001",
                        "order_purchase_timestamp": "2017-01-01T10:30:00",
                        "customer_state": "SP",
                        "seller_states": "SP",
                        "payment_types": "credit_card",
                        "total_price": 58.9,
                        "total_freight": 13.29,
                        "num_items": 1,
                        "num_sellers": 1,
                        "num_products": 1,
                        "total_weight": 500,
                        "num_payment_sequential": 1,
                        "avg_distance_km": 10.5,
                        "max_distance_km": 10.5,
                    }
                ]
            }
        },
    )

    orders: list[OrderRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "prediction": 0,
                        "probability": 0.1734,
                        "model_version": "7",
                    }
                ]
            }
        }
    )

    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_source: str
    registry_stage: str
    decision_threshold: float
    feature_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
