"""Evaluation utilities for binary fraud-detection classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable binary-classification metrics and a confusion matrix."""

    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Return metrics in a JSON-compatible structure."""
        return asdict(self)


def evaluate_classifier(
    model: Any, features: pd.DataFrame, target: pd.Series
) -> EvaluationResult:
    """Calculate threshold and ranking metrics for a fitted binary classifier."""
    if target.nunique() != 2:
        raise ValueError(
            "Evaluation requires both classes to be present in the target."
        )
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model must provide predict_proba for ROC-AUC and PR-AUC evaluation."
        )

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return EvaluationResult(
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(target, probabilities)),
        pr_auc=float(average_precision_score(target, probabilities)),
        confusion_matrix=confusion_matrix(target, predictions, labels=[0, 1])
        .astype(int)
        .tolist(),
    )
