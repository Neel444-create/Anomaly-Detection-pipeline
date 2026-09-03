"""Tests for the read-only raw-data pipeline and processed output."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import run_preprocessing
from src.data.validate import DataValidationError, REQUIRED_COLUMNS, validate_dataframe


def make_valid_data() -> pd.DataFrame:
    """Create a minimal, valid ULB-style dataset for isolated tests."""
    rows = []
    for offset, transaction_class in enumerate((0, 1)):
        row = {column: float(offset + 1) for column in REQUIRED_COLUMNS}
        row["Time"] = float(offset)
        row["Amount"] = float(offset + 10)
        row["Class"] = transaction_class
        rows.append(row)
    return pd.DataFrame(rows)


def test_required_columns_are_enforced() -> None:
    data = make_valid_data().drop(columns="Class")
    with pytest.raises(DataValidationError, match="Missing required columns: Class"):
        validate_dataframe(data)


def test_missing_values_raise_meaningful_error() -> None:
    data = make_valid_data()
    data.loc[0, "Amount"] = None
    with pytest.raises(
        DataValidationError, match="Missing values in required columns: Amount=1"
    ):
        validate_dataframe(data)


def test_validation_reports_duplicate_records() -> None:
    data = pd.concat(
        [make_valid_data(), make_valid_data().iloc[[0]]], ignore_index=True
    )
    result = validate_dataframe(data)
    assert result.is_valid
    assert result.duplicate_count == 1
    assert result.warnings == ["Found 1 duplicate transaction records."]


def test_preprocessing_separates_target_and_creates_expected_output(tmp_path) -> None:
    data = pd.concat(
        [make_valid_data(), make_valid_data().iloc[[0]]], ignore_index=True
    )
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    data.to_csv(raw_path, index=False)

    prepared = run_preprocessing(raw_path, processed_path)

    assert processed_path.is_file()
    assert len(prepared.processed_data) == 2
    assert "Class" not in prepared.features.columns
    assert {"Amount_Log", "Time_Of_Day"}.issubset(prepared.features.columns)
    assert {"Amount_Log", "Time_Of_Day"}.issubset(prepared.processed_data.columns)
    assert prepared.target.name == "Class"
    assert (
        pd.read_csv(processed_path).columns.tolist()
        == prepared.processed_data.columns.tolist()
    )
