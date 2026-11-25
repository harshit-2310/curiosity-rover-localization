"""Lesson 1: load and inspect Curiosity's public NASA localization CSV."""

import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).parent / "localized_pos.csv"


def main() -> None:
    # Path(__file__) makes this work no matter which folder runs the script from.
    rover = pd.read_csv(DATA_PATH)

    print("Dataset size (rows, columns):")
    print(rover.shape)

    print("\nColumn names:")
    print(rover.columns.tolist())

    print("\nFirst five rows:")
    print(rover.head())

    print("\nMissing values per column:")
    print(rover.isna().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
