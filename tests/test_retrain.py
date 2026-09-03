"""Tests for drift-gated candidate retraining decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.drift import DriftConfig
from src.models.evaluate import EvaluationResult
from src.models.retrain import (
    RetrainingConfig,
    candidate_meets_quality_gate,
    run_retraining_workflow,
)


def metrics(pr_auc: float, recall: float = 0.8) -> EvaluationResult:
    """Build concise metrics fixtures for acceptance-gate tests."""
    return EvaluationResult(0.8, recall, 0.8, 0.9, pr_auc, [[9, 1], [1, 9]])


def test_quality_gate_requires_all_explicit_thresholds() -> None:
    config = RetrainingConfig(
        minimum_pr_auc=0.7, minimum_pr_auc_improvement=0.02, minimum_recall=0.75
    )
    accepted, _ = candidate_meets_quality_gate(metrics(0.75), metrics(0.72), config)
    assert accepted
    accepted, reason = candidate_meets_quality_gate(
        metrics(0.74, recall=0.7), metrics(0.72), config
    )
    assert not accepted
    assert "recall" in reason


def test_workflow_skips_training_when_no_drift() -> None:
    data = pd.DataFrame({"Amount": np.linspace(0, 10, 40), "Class": [0, 1] * 20})
    decision = run_retraining_workflow(
        data,
        data.copy(),
        production_model=object(),
        retraining_config=RetrainingConfig(drift=DriftConfig(threshold=0.1)),
    )
    assert decision.status == "skipped"
    assert decision.registered_model_version is None
