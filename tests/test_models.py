"""Basic behavior tests for splitting, metrics, and baseline model setup."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import evaluate_classifier
from src.models.train import TrainingConfig, build_baseline_models, split_dataset


def make_training_data() -> pd.DataFrame:
    """Return a small, balanced, deterministic dataset for fast unit tests."""
    feature_a = list(range(20)) + list(range(100, 120))
    feature_b = [value % 5 for value in feature_a]
    target = [0] * 20 + [1] * 20
    return pd.DataFrame(
        {"feature_a": feature_a, "feature_b": feature_b, "Class": target}
    )


def test_split_dataset_preserves_both_classes() -> None:
    splits = split_dataset(make_training_data())
    assert len(splits.x_train) == 24
    assert len(splits.x_validation) == 8
    assert len(splits.x_test) == 8
    assert (
        splits.y_train.nunique()
        == splits.y_validation.nunique()
        == splits.y_test.nunique()
        == 2
    )


def test_baseline_models_include_required_estimators() -> None:
    models = build_baseline_models(TrainingConfig(random_forest_estimators=2))
    assert set(models) == {"logistic_regression", "random_forest"}
    assert (
        models["logistic_regression"].named_steps["classifier"].class_weight
        == "balanced"
    )
    assert models["random_forest"].class_weight == "balanced_subsample"


def test_evaluation_returns_requested_metrics() -> None:
    data = make_training_data()
    features = data.drop(columns="Class")
    model = LogisticRegression().fit(features, data["Class"])
    result = evaluate_classifier(model, features, data["Class"])
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.f1 <= 1
    assert 0 <= result.roc_auc <= 1
    assert 0 <= result.pr_auc <= 1
    assert len(result.confusion_matrix) == 2
