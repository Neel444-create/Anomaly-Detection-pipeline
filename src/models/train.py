"""Reproducible baseline training for financial anomaly detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Support both `python src/models/train.py` and module-style execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import DEFAULT_PROCESSED_DATA_PATH
from src.data.validate import TARGET_COLUMN
from src.models.evaluate import EvaluationResult, evaluate_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
DEFAULT_MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_PATH = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "financial-anomaly-detection"
REGISTERED_MODEL_NAME = "financial-anomaly-detection-best-model"


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters controlling split reproducibility and lightweight baselines."""

    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.25  # 25% of remaining data = 20% of all data.
    logistic_max_iter: int = 1_000
    random_forest_estimators: int = 100
    random_forest_max_depth: int = 12
    random_forest_min_samples_leaf: int = 5


@dataclass(frozen=True)
class DataSplits:
    """Stratified train, validation, and test partitions."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class TrainingResult:
    """Details of a completed model-selection and test-evaluation run."""

    best_model_name: str
    validation_metrics: dict[str, EvaluationResult]
    test_metrics: EvaluationResult
    model_path: Path
    metadata_path: Path
    split_sizes: dict[str, int]
    candidate_run_ids: dict[str, str]
    best_model_run_id: str
    registered_model_version: str


def load_processed_data(path: str | Path = DEFAULT_PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed data and ensure the target required for training exists."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run `python src/data/preprocess.py` first."
        )
    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Processed dataset must contain target column '{TARGET_COLUMN}'."
        )
    return data


def split_dataset(
    data: pd.DataFrame, config: TrainingConfig = TrainingConfig()
) -> DataSplits:
    """Create reproducible 60/20/20 stratified train/validation/test splits."""
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    if target.nunique() != 2:
        raise ValueError("Training requires both fraud and non-fraud classes.")

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y_train_validation,
    )
    return DataSplits(x_train, x_validation, x_test, y_train, y_validation, y_test)


def build_baseline_models(config: TrainingConfig = TrainingConfig()) -> dict[str, Any]:
    """Build two class-weighted baseline models.

    Class weights counter the 0.17% fraud rate. The scaler is held inside the
    logistic-regression pipeline so it is fit only on training data.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=config.logistic_max_iter,
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.random_forest_estimators,
            max_depth=config.random_forest_max_depth,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state,
        ),
    }


def get_tracking_uri(tracking_uri: str | None = None) -> str:
    """Return an explicit URI, environment override, or local SQLite default."""
    if tracking_uri:
        return tracking_uri
    return os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_MLFLOW_DB_PATH.as_posix()}"
    )


def configure_mlflow(tracking_uri: str | None = None) -> str:
    """Configure a local tracking store and select the project experiment."""
    uri = get_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite:///"):
        DEFAULT_MLFLOW_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(
                EXPERIMENT_NAME, artifact_location=DEFAULT_MLFLOW_ARTIFACT_PATH.as_uri()
            )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def dataset_information(
    data: pd.DataFrame, data_path: str | Path
) -> dict[str, str | int | float]:
    """Return lightweight dataset lineage metadata without logging data itself."""
    resolved_path = Path(data_path).resolve()
    digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    return {
        "dataset_path": str(resolved_path),
        "dataset_sha256": digest,
        "dataset_rows": len(data),
        "dataset_columns": len(data.columns),
        "positive_class_rate": float(data[TARGET_COLUMN].mean()),
    }


def _log_metrics(prefix: str, metrics: EvaluationResult) -> None:
    """Log scalar metrics and the confusion matrix as a JSON artifact."""
    mlflow.log_metrics(
        {
            f"{prefix}_{name}": value
            for name, value in metrics.to_dict().items()
            if name != "confusion_matrix"
        }
    )
    mlflow.log_dict(
        {"confusion_matrix": metrics.confusion_matrix},
        f"{prefix}_confusion_matrix.json",
    )


def _model_parameters(model: Any) -> dict[str, str | int | float | bool]:
    """Extract MLflow-compatible model parameters from a sklearn estimator."""
    parameters = model.get_params(deep=False)
    return {
        f"model_{name}": value
        for name, value in parameters.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _save_artifacts(
    model: Any,
    model_path: Path,
    metadata_path: Path,
    *,
    best_model_name: str,
    validation_metrics: dict[str, EvaluationResult],
    test_metrics: EvaluationResult,
    split_sizes: dict[str, int],
    feature_names: list[str],
    config: TrainingConfig,
    tracking_uri: str,
    best_model_run_id: str,
    registered_model_version: str,
) -> None:
    """Save the selected estimator and enough metadata to reproduce the run."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "split_sizes": split_sizes,
        "training_config": asdict(config),
        "validation_metrics": {
            name: result.to_dict() for name, result in validation_metrics.items()
        },
        "test_metrics": test_metrics.to_dict(),
        "mlflow": {
            "experiment_name": EXPERIMENT_NAME,
            "tracking_uri": tracking_uri,
            "best_model_run_id": best_model_run_id,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": registered_model_version,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_baselines(
    data_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    config: TrainingConfig = TrainingConfig(),
    tracking_uri: str | None = None,
) -> TrainingResult:
    """Train, track, select by validation PR-AUC, register, and save the best model."""
    data = load_processed_data(data_path)
    tracking_uri = configure_mlflow(tracking_uri)
    data_info = dataset_information(data, data_path)
    splits = split_dataset(data, config)
    validation_metrics: dict[str, EvaluationResult] = {}
    fitted_models: dict[str, Any] = {}
    candidate_run_ids: dict[str, str] = {}

    for name, model in build_baseline_models(config).items():
        with mlflow.start_run(run_name=f"candidate-{name}") as run:
            mlflow.set_tag("model_type", name)
            mlflow.log_params(
                {**asdict(config), **data_info, **_model_parameters(model)}
            )
            model.fit(splits.x_train, splits.y_train)
            fitted_models[name] = model
            validation_metrics[name] = evaluate_classifier(
                model, splits.x_validation, splits.y_validation
            )
            _log_metrics("validation", validation_metrics[name])
            mlflow.sklearn.log_model(model, name="model")
            candidate_run_ids[name] = run.info.run_id

    # PR-AUC is appropriate for selecting among candidates on highly imbalanced data.
    best_model_name = max(
        validation_metrics,
        key=lambda name: (validation_metrics[name].pr_auc, validation_metrics[name].f1),
    )
    best_model = clone(fitted_models[best_model_name])
    x_train_validation = pd.concat([splits.x_train, splits.x_validation])
    y_train_validation = pd.concat([splits.y_train, splits.y_validation])
    best_model.fit(x_train_validation, y_train_validation)
    test_metrics = evaluate_classifier(best_model, splits.x_test, splits.y_test)

    split_sizes = {
        "train": len(splits.x_train),
        "validation": len(splits.x_validation),
        "test": len(splits.x_test),
    }
    resolved_model_path = Path(model_path)
    resolved_metadata_path = Path(metadata_path)
    with mlflow.start_run(run_name="best-model-refit") as best_run:
        mlflow.set_tags(
            {
                "model_type": best_model_name,
                "selection_metric": "validation_pr_auc",
                "run_role": "champion",
            }
        )
        mlflow.log_params(
            {
                **asdict(config),
                **data_info,
                "selected_from_run_id": candidate_run_ids[best_model_name],
            }
        )
        _log_metrics("validation", validation_metrics[best_model_name])
        _log_metrics("test", test_metrics)
        model_info = mlflow.sklearn.log_model(best_model, name="model")
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri, name=REGISTERED_MODEL_NAME
        )
        best_model_run_id = best_run.info.run_id

    _save_artifacts(
        best_model,
        resolved_model_path,
        resolved_metadata_path,
        best_model_name=best_model_name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        split_sizes=split_sizes,
        feature_names=splits.x_train.columns.tolist(),
        config=config,
        tracking_uri=tracking_uri,
        best_model_run_id=best_model_run_id,
        registered_model_version=str(registered_model.version),
    )
    return TrainingResult(
        best_model_name=best_model_name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model_path=resolved_model_path,
        metadata_path=resolved_metadata_path,
        split_sizes=split_sizes,
        candidate_run_ids=candidate_run_ids,
        best_model_run_id=best_model_run_id,
        registered_model_version=str(registered_model.version),
    )


def _print_metrics(label: str, metrics: EvaluationResult) -> None:
    """Print metrics produced by this run for human inspection."""
    print(f"\n{label}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1:        {metrics.f1:.4f}")
    print(f"ROC-AUC:   {metrics.roc_auc:.4f}")
    print(f"PR-AUC:    {metrics.pr_auc:.4f}")
    print(f"Confusion matrix: {metrics.confusion_matrix}")


if __name__ == "__main__":
    result = train_baselines()
    print(f"Selected model: {result.best_model_name}")
    print(f"Split sizes: {result.split_sizes}")
    for model_name, metrics in result.validation_metrics.items():
        _print_metrics(f"Validation metrics — {model_name}", metrics)
    _print_metrics("Test metrics", result.test_metrics)
    print(f"Saved model: {result.model_path}")
    print(f"MLflow run ID: {result.best_model_run_id}")
    print(
        f"Registered model: {REGISTERED_MODEL_NAME} v{result.registered_model_version}"
    )
