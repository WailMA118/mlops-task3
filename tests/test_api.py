from fastapi.testclient import TestClient

from app.main import app


class FakeModel:
    version = "test-1"
    source = "test"


class FakeArtifacts:
    final_feature_list = ["f1", "f2"]


class FakePipeline:
    loaded_model = FakeModel()
    feature_artifacts = FakeArtifacts()
    threshold = 0.5

    def predict_one(self, order):
        from src.inference.predict import PredictionResult

        return PredictionResult(
            is_late=1,
            probability_late=0.82,
            model_version="test-1",
            latency_ms=1.0,
        )


def test_health_and_model_info():
    with TestClient(app) as client:
        app.state.pipeline = FakePipeline()
        health = client.get("/health")
        info = client.get("/model")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert info.status_code == 200
        assert info.json()["model_version"] == "test-1"


def test_predict_rejects_unknown_fields():
    with TestClient(app) as client:
        app.state.pipeline = FakePipeline()
        payload = {
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
            "unexpected": "reject me",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"] == "Request validation failed"


def test_predict_response_contract():
    with TestClient(app) as client:
        app.state.pipeline = FakePipeline()
        payload = {
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
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "prediction": 1,
            "probability": 0.82,
            "model_version": "test-1",
        }


def test_batch_response_contract():
    with TestClient(app) as client:
        app.state.pipeline = FakePipeline()
        payload = {
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
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 1
