"""Lightweight numeric feature-drift detection using Population Stability Index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import DEFAULT_PROCESSED_DATA_PATH
from src.data.validate import TARGET_COLUMN


@dataclass(frozen=True)
class DriftConfig:
    """Configuration for PSI-based feature distribution comparisons."""

    threshold: float = 0.20
    bins: int = 10
    epsilon: float = 1e-6


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift result for one numeric feature."""

    feature: str
    score: float
    status: str


def population_stability_index(
    reference: pd.Series, production: pd.Series, config: DriftConfig = DriftConfig()
) -> float:
    """Calculate PSI; higher values indicate larger distribution shifts.

    Quantile bins are derived only from reference data, which makes it a stable
    baseline for comparing later production samples. Small probabilities are
    clipped to avoid undefined logarithms when a bin is empty.
    """
    reference_values = reference.dropna().to_numpy(dtype=float)
    production_values = production.dropna().to_numpy(dtype=float)
    if not len(reference_values) or not len(production_values):
        raise ValueError(
            "PSI requires at least one non-missing value in both datasets."
        )

    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, config.bins + 1)))
    if len(edges) < 2:
        edges = np.array([reference_values[0] - 0.5, reference_values[0] + 0.5])
    else:
        edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference_values, bins=edges)[0] / len(
        reference_values
    )
    production_share = np.histogram(production_values, bins=edges)[0] / len(
        production_values
    )
    reference_share = np.clip(reference_share, config.epsilon, None)
    production_share = np.clip(production_share, config.epsilon, None)
    return float(
        np.sum(
            (production_share - reference_share)
            * np.log(production_share / reference_share)
        )
    )


def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
    config: DriftConfig = DriftConfig(),
) -> list[FeatureDriftResult]:
    """Compare common numeric features and return one PSI result per feature."""
    if features is None:
        features = [
            column for column in reference_data.columns if column != TARGET_COLUMN
        ]
    selected_features = list(features)
    missing = [
        column
        for column in selected_features
        if column not in reference_data.columns or column not in production_data.columns
    ]
    if missing:
        raise ValueError(
            f"Features missing from reference or production data: {', '.join(missing)}"
        )

    results = []
    for feature in selected_features:
        if not pd.api.types.is_numeric_dtype(
            reference_data[feature]
        ) or not pd.api.types.is_numeric_dtype(production_data[feature]):
            raise TypeError(f"PSI monitoring supports numeric features only: {feature}")
        score = population_stability_index(
            reference_data[feature], production_data[feature], config
        )
        results.append(
            FeatureDriftResult(
                feature, score, "DRIFT" if score >= config.threshold else "OK"
            )
        )
    return results


def drift_report(results: Iterable[FeatureDriftResult]) -> pd.DataFrame:
    """Return a tabular monitoring report sorted by highest drift score."""
    return pd.DataFrame(
        [
            {"Feature": item.feature, "Drift Score": item.score, "Status": item.status}
            for item in results
        ]
    ).sort_values("Drift Score", ascending=False, ignore_index=True)


def has_drift(results: Iterable[FeatureDriftResult]) -> bool:
    """Return whether any monitored feature crossed its configured threshold."""
    return any(result.status == "DRIFT" for result in results)


def print_report(results: Iterable[FeatureDriftResult]) -> None:
    """Print the compact monitoring table for command-line use."""
    print(
        drift_report(results).to_string(
            index=False, formatters={"Drift Score": "{:.4f}".format}
        )
    )


if __name__ == "__main__":
    data = pd.read_csv(DEFAULT_PROCESSED_DATA_PATH)
    split_index = int(len(data) * 0.8)
    print_report(detect_drift(data.iloc[:split_index], data.iloc[split_index:]))
