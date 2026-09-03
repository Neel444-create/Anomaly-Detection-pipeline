"""Drift-gated candidate retraining without automatic production deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monitoring.drift import DriftConfig, FeatureDriftResult, detect_drift, has_drift
from src.data.preprocess import DEFAULT_PROCESSED_DATA_PATH
from src.models.evaluate import EvaluationResult, evaluate_classifier
from src.models.train import (
    TrainingConfig,
    build_baseline_models,
    configure_mlflow,
    load_processed_data,
    split_dataset,
)


CANDIDATE_MODEL_NAME = "financial-anomaly-detection-candidates"


@dataclass(frozen=True)
class RetrainingConfig:
    """Explicit criteria for retraining and candidate acceptance."""

    drift: DriftConfig = DriftConfig()
    minimum_pr_auc: float = 0.50
    minimum_pr_auc_improvement: float = 0.00
    minimum_recall: float = 0.50


@dataclass(frozen=True)
class RetrainingDecision:
    """Outcome of the retraining workflow, including reasons for a rejection."""

    status: str
    reason: str
    drift_results: list[FeatureDriftResult]
    candidate_model_name: str | None = None
    candidate_metrics: EvaluationResult | None = None
    production_metrics: EvaluationResult | None = None
    registered_model_version: str | None = None


def candidate_meets_quality_gate(
    candidate: EvaluationResult, production: EvaluationResult, config: RetrainingConfig
) -> tuple[bool, str]:
    """Compare a candidate against transparent minimum quality criteria."""
    if candidate.pr_auc < config.minimum_pr_auc:
        return (
            False,
            f"Candidate PR-AUC {candidate.pr_auc:.4f} is below minimum {config.minimum_pr_auc:.4f}.",
        )
    if candidate.recall < config.minimum_recall:
        return (
            False,
            f"Candidate recall {candidate.recall:.4f} is below minimum {config.minimum_recall:.4f}.",
        )
    improvement = candidate.pr_auc - production.pr_auc
    if improvement < config.minimum_pr_auc_improvement:
        return (
            False,
            f"Candidate PR-AUC improvement {improvement:.4f} is below required {config.minimum_pr_auc_improvement:.4f}.",
        )
    return True, "Candidate meets PR-AUC, recall, and improvement thresholds."


def _train_candidate(
    data: pd.DataFrame, config: TrainingConfig
) -> tuple[str, Any, EvaluationResult]:
    """Select a baseline candidate by validation PR-AUC and evaluate it once on test data."""
    splits = split_dataset(data, config)
    validation_metrics: dict[str, EvaluationResult] = {}
    fitted_models: dict[str, Any] = {}
    for name, model in build_baseline_models(config).items():
        model.fit(splits.x_train, splits.y_train)
        fitted_models[name] = model
        validation_metrics[name] = evaluate_classifier(
            model, splits.x_validation, splits.y_validation
        )
    selected_name = max(
        validation_metrics, key=lambda name: validation_metrics[name].pr_auc
    )
    selected = clone(fitted_models[selected_name])
    selected.fit(
        pd.concat([splits.x_train, splits.x_validation]),
        pd.concat([splits.y_train, splits.y_validation]),
    )
    return (
        selected_name,
        selected,
        evaluate_classifier(selected, splits.x_test, splits.y_test),
    )


def _register_candidate(model: Any, model_name: str, tracking_uri: str | None) -> str:
    """Log and register an accepted candidate; this deliberately does not deploy it."""
    configure_mlflow(tracking_uri)
    with mlflow.start_run(run_name=f"candidate-retrain-{model_name}"):
        mlflow.set_tags(
            {
                "run_role": "candidate",
                "deployment_status": "not_deployed",
                "model_type": model_name,
            }
        )
        model_info = mlflow.sklearn.log_model(model, name="model")
        registered = mlflow.register_model(model_info.model_uri, CANDIDATE_MODEL_NAME)
    return str(registered.version)


def run_retraining_workflow(
    reference_data: pd.DataFrame,
    candidate_data: pd.DataFrame,
    production_model: Any,
    *,
    retraining_config: RetrainingConfig = RetrainingConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    tracking_uri: str | None = None,
    register_model: Callable[[Any, str, str | None], str] = _register_candidate,
) -> RetrainingDecision:
    """Drift-gated candidate workflow; only MLflow registration can follow acceptance.

    The passed production model is evaluated for comparison only. This function
    never writes to `models/best_model.joblib`, so production deployment remains
    a manual promotion decision.
    """
    drift_results = detect_drift(
        reference_data, candidate_data, config=retraining_config.drift
    )
    if not has_drift(drift_results):
        return RetrainingDecision(
            "skipped", "No feature exceeded the drift threshold.", drift_results
        )

    candidate_name, candidate_model, candidate_metrics = _train_candidate(
        candidate_data, training_config
    )
    candidate_splits = split_dataset(candidate_data, training_config)
    production_metrics = evaluate_classifier(
        production_model, candidate_splits.x_test, candidate_splits.y_test
    )
    accepted, reason = candidate_meets_quality_gate(
        candidate_metrics, production_metrics, retraining_config
    )
    if not accepted:
        return RetrainingDecision(
            "rejected",
            reason,
            drift_results,
            candidate_name,
            candidate_metrics,
            production_metrics,
        )

    version = register_model(candidate_model, candidate_name, tracking_uri)
    return RetrainingDecision(
        "accepted_candidate",
        f"{reason} Registered as candidate only; production was not changed.",
        drift_results,
        candidate_name,
        candidate_metrics,
        production_metrics,
        version,
    )


def run_local_retraining_check(
    reference_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    candidate_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    production_model_path: str | Path = "models/best_model.joblib",
) -> RetrainingDecision:
    """Run the workflow locally using configured files; suitable for manual review."""
    reference_data = load_processed_data(reference_path)
    candidate_data = load_processed_data(candidate_path)
    production_model = joblib.load(production_model_path)
    return run_retraining_workflow(reference_data, candidate_data, production_model)


if __name__ == "__main__":
    decision = run_local_retraining_check()
    print(f"Retraining status: {decision.status}")
    print(f"Reason: {decision.reason}")
