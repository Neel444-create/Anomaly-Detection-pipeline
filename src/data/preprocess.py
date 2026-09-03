"""Reproducible preprocessing pipeline for credit-card transactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow the documented direct command (`python src/data/preprocess.py`) while
# retaining normal package imports when run with `python -m src.data.preprocess`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import DEFAULT_RAW_DATA_PATH, PROJECT_ROOT, load_csv
from src.data.validate import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ValidationResult,
    validate_dataframe,
)


DEFAULT_PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "creditcard_processed.csv"
)
MODEL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "Amount_Log", "Time_Of_Day"]


@dataclass(frozen=True)
class PreparedData:
    """Validated model inputs, target, and persisted processed dataset."""

    features: pd.DataFrame
    target: pd.Series
    processed_data: pd.DataFrame
    validation: ValidationResult


def preprocess_dataframe(data: pd.DataFrame) -> PreparedData:
    """Validate and transform raw records into a model-ready tabular dataset.

    Scaling is deliberately deferred until model training, where it can be fit
    only on the training split and avoid data leakage. This step is limited to
    deterministic row cleanup and transformations that do not learn from data.
    """
    validation = validate_dataframe(data)
    processed = data.drop_duplicates().copy()
    processed[TARGET_COLUMN] = processed[TARGET_COLUMN].astype("int8")
    features = transform_features(processed)
    # Persist deterministic derived features as part of the processed dataset.
    # The API calls the same function at inference, keeping its schema aligned.
    processed["Amount_Log"] = features["Amount_Log"]
    processed["Time_Of_Day"] = features["Time_Of_Day"]
    target = processed.loc[:, TARGET_COLUMN].copy()
    return PreparedData(features, target, processed, validation)


def transform_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic training-time feature transformations for inference.

    This shared function does not require the target column, so the API and
    training pipeline cannot silently diverge in their feature calculations.
    """
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")
    features = data.loc[:, FEATURE_COLUMNS].copy()
    features["Amount_Log"] = np.log1p(features["Amount"])
    features["Time_Of_Day"] = (features["Time"] % 86_400) / 86_400
    return features.loc[:, MODEL_FEATURE_COLUMNS]


def save_processed_data(
    prepared: PreparedData, output_path: str | Path = DEFAULT_PROCESSED_DATA_PATH
) -> Path:
    """Persist processed data separately from the immutable raw source file."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.processed_data.to_csv(destination, index=False)
    return destination


def run_preprocessing(
    input_path: str | Path = DEFAULT_RAW_DATA_PATH,
    output_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
) -> PreparedData:
    """Load, validate, preprocess, and save a dataset with explicit paths."""
    prepared = preprocess_dataframe(load_csv(input_path))
    save_processed_data(prepared, output_path)
    return prepared


if __name__ == "__main__":
    prepared_data = run_preprocessing()
    print(
        f"Saved {len(prepared_data.processed_data)} processed rows to {DEFAULT_PROCESSED_DATA_PATH}"
    )
