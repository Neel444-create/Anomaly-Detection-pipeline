"""Command-line exploration for the raw credit-card fraud dataset.

This module is intentionally read-only: it reports dataset quality and summary
statistics without modifying the source CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
TARGET_COLUMN = "Class"


def load_data(data_path: Path) -> pd.DataFrame:
    """Load a CSV dataset, with a helpful error if it is unavailable."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def explore_data(data: pd.DataFrame) -> None:
    """Print essential, reproducible exploratory statistics for *data*."""
    print("\n=== Dataset shape ===")
    print(data.shape)

    print("\n=== Column information ===")
    data.info()

    print("\n=== Missing values by column ===")
    print(data.isna().sum().to_string())

    print("\n=== Duplicate records ===")
    print(data.duplicated().sum())

    if TARGET_COLUMN not in data.columns:
        print(
            f"\n=== Class distribution ===\nSkipped: '{TARGET_COLUMN}' column not found."
        )
    else:
        class_counts = data[TARGET_COLUMN].value_counts().sort_index()
        class_percentages = (
            data[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
        )

        print("\n=== Class distribution ===")
        print(class_counts.to_string())

        print("\n=== Class percentages ===")
        print(class_percentages.round(4).to_string())

    print("\n=== Basic numerical statistics ===")
    print(data.describe().T.to_string())


def parse_args() -> argparse.Namespace:
    """Parse optional data-path override for reproducible local execution."""
    parser = argparse.ArgumentParser(
        description="Explore the credit-card fraud dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV to explore (default: {DEFAULT_DATA_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configured CSV and print the exploration report."""
    args = parse_args()
    explore_data(load_data(args.data_path))


if __name__ == "__main__":
    main()
