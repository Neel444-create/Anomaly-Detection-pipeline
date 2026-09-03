"""Tests for PSI feature-distribution monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.drift import (
    DriftConfig,
    detect_drift,
    drift_report,
    has_drift,
    population_stability_index,
)


def test_psi_is_near_zero_for_matching_distributions() -> None:
    reference = pd.Series(np.linspace(0, 1, 1_000))
    score = population_stability_index(reference, reference.copy())
    assert score < 1e-9


def test_detect_drift_identifies_shifted_feature() -> None:
    reference = pd.DataFrame(
        {"Amount": np.linspace(0, 10, 1_000), "V1": np.linspace(-1, 1, 1_000)}
    )
    production = pd.DataFrame(
        {"Amount": np.linspace(100, 110, 1_000), "V1": np.linspace(-1, 1, 1_000)}
    )
    results = detect_drift(reference, production, config=DriftConfig(threshold=0.2))
    by_feature = {result.feature: result for result in results}
    assert by_feature["Amount"].status == "DRIFT"
    assert by_feature["V1"].status == "OK"
    assert has_drift(results)


def test_report_has_clear_monitoring_columns() -> None:
    data = pd.DataFrame({"Time": [0.0, 1.0], "Amount": [10.0, 20.0]})
    report = drift_report(detect_drift(data, data, config=DriftConfig(threshold=0.1)))
    assert report.columns.tolist() == ["Feature", "Drift Score", "Status"]
    assert set(report["Status"]) == {"OK"}
