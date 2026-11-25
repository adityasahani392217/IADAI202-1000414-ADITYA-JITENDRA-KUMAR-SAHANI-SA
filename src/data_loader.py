"""Memory-efficient data loading utilities for MediScope AI."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import (
    AGGREGATES_DIR,
    CLOUD_SAMPLE_PARQUET_PATH,
    PROCESSED_PARQUET_PATH,
    resolve_raw_csv_path,
)


def iter_raw_chunks(
    csv_path: Path | str | None = None,
    usecols: list[str] | None = None,
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Yield chunks of the raw SPARCS CSV as string DataFrames to prevent parse errors."""
    resolved = Path(csv_path) if csv_path is not None else resolve_raw_csv_path()
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            "Raw SPARCS CSV file not found. Place the CSV in data/raw/ or project root, "
            "or set the MEDISCOPE_DATA_PATH environment variable."
        )
    yield from pd.read_csv(
        resolved,
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=True,
        low_memory=False,
    )


def load_raw_dataset(
    csv_path: Path | str | None = None,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load the raw dataset (optionally a column subset or row limit) cleanly."""
    resolved = Path(csv_path) if csv_path is not None else resolve_raw_csv_path()
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            "Raw SPARCS CSV file not found. Place the CSV in data/raw/ or project root, "
            "or set MEDISCOPE_DATA_PATH."
        )

    if nrows is not None:
        return pd.read_csv(
            resolved,
            usecols=usecols,
            nrows=nrows,
            dtype=str,
            keep_default_na=True,
            low_memory=False,
        )

    chunks = list(iter_raw_chunks(resolved, usecols=usecols, chunksize=chunksize))
    return pd.concat(chunks, ignore_index=True)


def get_data_mode() -> str:
    """Return active dataset mode: 'full_local', 'raw_local', or 'cloud_sample'."""
    if PROCESSED_PARQUET_PATH.exists():
        return "full_local"
    if resolve_raw_csv_path() is not None:
        return "raw_local"
    if CLOUD_SAMPLE_PARQUET_PATH.exists():
        return "cloud_sample"
    return "unavailable"


def load_analysis_dataset(
    columns: list[str] | None = None,
    prefer_full: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load the best available cleaned dataset and return (DataFrame, mode).

    Priority:
      1. Cleaned full Parquet (`data/processed/discharges_cleaned.parquet`)
      2. Stratified cloud sample Parquet (`data/aggregates/discharges_sample.parquet`)
    """
    if prefer_full and PROCESSED_PARQUET_PATH.exists():
        df = pd.read_parquet(PROCESSED_PARQUET_PATH, columns=columns)
        return df, "full_local"

    if CLOUD_SAMPLE_PARQUET_PATH.exists():
        df = pd.read_parquet(CLOUD_SAMPLE_PARQUET_PATH, columns=columns)
        return df, "cloud_sample"

    raise FileNotFoundError(
        "No processed dataset found. Run `python -m src.preprocessing` locally with the raw CSV."
    )


def load_aggregate_table(filename: str) -> pd.DataFrame | None:
    """Load a pre-aggregated parquet table from data/aggregates if available."""
    path = AGGREGATES_DIR / filename
    if path.exists():
        return pd.read_parquet(path)
    return None
