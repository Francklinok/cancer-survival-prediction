"""
data_loading.py — Data ingestion and validation service.

Features:
  - Multi-format loading: CSV, Excel, Parquet, JSON
  - Automatic CSV separator detection
  - Schema validation (columns, types, minimum size)
  - Basic safety cleanup (column strip, obvious type conversion)
  - Raw data protection (immutable deep copy)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.data_loading")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def load_data(
    file_path: str | os.PathLike,
    target_column: Optional[str] = None,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
    sheet_name: int | str = 0,
    validate: bool = True,
    min_rows: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a data file and return (df_raw, df_work).

    - ``df_raw``  : immutable reference copy — never modify it.
    - ``df_work`` : working copy used by the pipeline.

    Parameters
    ----------
    file_path : str | Path
        Path to the data file.
    target_column : str, optional
        Name of the target column — used for validation only.
    encoding : str
        File encoding (CSV / JSON).
    sep : str, optional
        CSV separator. If None, auto-detection is used.
    sheet_name : int | str
        Excel sheet to load.
    validate : bool
        Enable schema validation.
    min_rows : int
        Minimum number of rows required.
    verbose : bool
        Print the initial quality report.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DatsaFrame]
        (df_raw, df_work)

    Raises
    -------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is not supported or validation fails.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    suffix = path.suffix.lower()
    supported = {".csv", ".xlsx", ".xls", ".parquet", ".json"}

    if suffix not in supported:
        raise ValueError(
            f"Format '{suffix}' not supported. Accepted formats: {supported}"
        )

    logger.info("Loading file: %s", path.name)

    try:
        if suffix == ".csv":
            df = _load_csv(path, encoding, sep)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=sheet_name)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".json":
            df = pd.read_json(path, encoding=encoding)
    except Exception as exc:
        raise ValueError(f"Error loading '{path.name}': {exc}") from exc

    df = _basic_cleanup(df)

    if validate:
        _validate(df, target_column=target_column, min_rows=min_rows)

    df_raw = df.copy(deep=True)
    df_work = df.copy(deep=True)

    logger.info(
        "Data loaded successfully — %d rows × %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df_raw, df_work

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _load_csv(
    path: Path,
    encoding: str,
    sep: Optional[str],
) -> pd.DataFrame:
    """Load a CSV file with automatic separator detection if needed."""
    if sep is not None:
        return pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False)

    # Auto-detect among common separators
    for candidate in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                path, encoding=encoding, sep=candidate, nrows=5, low_memory=False
            )
            if df.shape[1] > 1:
                logger.debug("Detected separator: '%s'", candidate)
                return pd.read_csv(
                    path, encoding=encoding, sep=candidate, low_memory=False
                )
        except Exception:
            continue

    # Last resort: pandas auto-detection
    return pd.read_csv(path, encoding=encoding, sep=None, engine="python", low_memory=False)

def _basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal non-destructive cleanup:
      - Strip whitespace from column names
      - Drop completely empty columns
      - Drop completely empty rows
    """
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.reset_index(drop=True)
    return df


def _validate(
    df: pd.DataFrame,
    target_column: Optional[str],
    min_rows: int,
) -> None:
    """Validate the loaded DataFrame. Raises ValueError if a constraint is violated."""
    errors: list[str] = []

    if df.empty:
        errors.append("DataFrame is empty.")

    if len(df) < min_rows:
        errors.append(
            f"Insufficient row count: {len(df)} < {min_rows} (minimum required)."
        )

    if target_column and target_column not in df.columns:
        errors.append(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    if errors:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(errors))

    logger.info("Validation passed ✓")

