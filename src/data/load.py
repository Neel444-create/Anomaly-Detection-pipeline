"""Data-loading utilities for the raw credit-card transaction dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def load_csv(path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Load a CSV file without altering its contents.

    Parameters are intentionally exposed so tests and later pipeline runs can
    supply another data source without editing application code.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)
