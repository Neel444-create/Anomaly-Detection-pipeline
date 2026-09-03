"""Schema and data-quality validation for credit-card transaction data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["Time", *[f"V{index}" for index in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


class DataValidationError(ValueError):
    """Raised when a dataset violates one or more required quality rules."""


@dataclass(frozen=True)
class ValidationResult:
    """Structured outcome of a data-validation run."""

    row_count: int
    duplicate_count: int
    missing_values: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether blocking data-quality errors were found."""
        return not self.errors


def validate_dataframe(
    data: pd.DataFrame,
    *,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
    raise_on_error: bool = True,
) -> ValidationResult:
    """Validate schema, missingness, numeric data, ranges, and duplicates.

    Exact duplicate records are reported as a warning rather than an error.
    They can be safely removed in the processed output while preserving the raw
    source file and its original lineage.
    """
    required = list(required_columns)
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [column for column in required if column not in data.columns]

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        result = ValidationResult(
            row_count=len(data),
            duplicate_count=int(data.duplicated().sum()),
            missing_values={},
            errors=errors,
            warnings=warnings,
        )
        if raise_on_error:
            raise DataValidationError("; ".join(result.errors))
        return result

    missing_values = data[required].isna().sum()
    missing_summary = {
        column: int(count) for column, count in missing_values.items() if count
    }
    if missing_summary:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_summary.items()
        )
        errors.append(f"Missing values in required columns: {details}")

    non_numeric = [
        column for column in required if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        errors.append(f"Expected numeric columns: {', '.join(non_numeric)}")
    else:
        finite_columns = [
            column for column in FEATURE_COLUMNS if column in data.columns
        ]
        invalid_finite = [
            column
            for column in finite_columns
            if not np.isfinite(data[column].dropna().to_numpy()).all()
        ]
        if invalid_finite:
            errors.append(
                f"Non-finite numeric values found in: {', '.join(invalid_finite)}"
            )

        if (data["Time"].dropna() < 0).any():
            errors.append("Time values must be non-negative.")
        if (data["Amount"].dropna() < 0).any():
            errors.append("Amount values must be non-negative.")
        invalid_classes = sorted(set(data[TARGET_COLUMN].dropna().unique()) - {0, 1})
        if invalid_classes:
            errors.append(
                f"Class values must be binary (0 or 1); found: {invalid_classes}"
            )

    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicate transaction records.")

    result = ValidationResult(
        row_count=len(data),
        duplicate_count=duplicate_count,
        missing_values=missing_summary,
        errors=errors,
        warnings=warnings,
    )
    if raise_on_error and errors:
        raise DataValidationError("; ".join(errors))
    return result
