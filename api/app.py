"""FastAPI service for production financial-fraud predictions."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.data.preprocess import MODEL_FEATURE_COLUMNS, transform_features
from src.data.validate import FEATURE_COLUMNS
from src.models.train import DEFAULT_MODEL_PATH


def _request_fields() -> dict[str, tuple[type, Field]]:
    """Build a strict Pydantic schema from the shared raw feature contract."""
    fields: dict[str, tuple[type, Field]] = {}
    for feature in FEATURE_COLUMNS:
        minimum = 0.0 if feature in {"Time", "Amount"} else None
        fields[feature] = (float, Field(..., ge=minimum))
    return fields


TransactionRequest = create_model(
    "TransactionRequest",
    __base__=BaseModel,
    __config__=ConfigDict(extra="forbid"),
    **_request_fields(),
)


class PredictionResponse(BaseModel):
    """Prediction returned by the production fraud model."""

    prediction: int
    anomaly_score: float


@lru_cache(maxsize=1)
def load_production_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load and cache the production artifact saved by the training pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Production model not found: {path}. Run training before serving predictions."
        )
    return joblib.load(path)


def create_app(
    model: Any | None = None, model_path: str | Path | None = None
) -> FastAPI:
    """Create an API instance; injectable model support keeps endpoint tests fast."""
    app = FastAPI(title="Financial Anomaly Detection API", version="0.1.0")
    configured_path = Path(
        model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    )

    def get_model() -> Any:
        return model if model is not None else load_production_model(configured_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            get_model()
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error
        return {"status": "healthy"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(transaction: TransactionRequest) -> PredictionResponse:
        try:
            production_model = get_model()
            raw_features = pd.DataFrame([transaction.model_dump()])
            model_features = transform_features(raw_features).loc[
                :, MODEL_FEATURE_COLUMNS
            ]
            probability = float(production_model.predict_proba(model_features)[:, 1][0])
            prediction = int(production_model.predict(model_features)[0])
            return PredictionResponse(prediction=prediction, anomaly_score=probability)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error

    return app


app = create_app()
