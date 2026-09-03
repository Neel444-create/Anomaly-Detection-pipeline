"""Tests for FastAPI serving behavior and input validation."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app
from src.data.validate import FEATURE_COLUMNS


class PredictableModel:
    """Small deterministic stand-in for API endpoint tests."""

    def predict_proba(self, features):
        assert list(features.columns)[-2:] == ["Amount_Log", "Time_Of_Day"]
        return np.array([[0.88, 0.12]] * len(features))

    def predict(self, features):
        return np.zeros(len(features), dtype=int)


def valid_payload() -> dict[str, float]:
    """Return one complete valid raw transaction request."""
    return {column: 1.0 for column in FEATURE_COLUMNS}


def test_health_endpoint_is_healthy() -> None:
    client = TestClient(create_app(model=PredictableModel()))
    assert client.get("/health").json() == {"status": "healthy"}


def test_predict_returns_prediction_and_probability() -> None:
    client = TestClient(create_app(model=PredictableModel()))
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    assert response.json() == {"prediction": 0, "anomaly_score": 0.12}


def test_predict_rejects_invalid_request() -> None:
    client = TestClient(create_app(model=PredictableModel()))
    payload = valid_payload()
    payload.pop("Amount")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
