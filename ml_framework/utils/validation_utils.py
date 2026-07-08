"""
validation_utils.py — Input validation helpers for the ml_framework.

All functions raise ValueError or TypeError with informative messages.
They are pure guards — no logging, no printing.
"""

from __future__ import annotations

import difflib
from typing import Any, Collection, Optional

import numpy as np
import pandas as pd

# =============================================================================
# COLUMN EXISTENCE
# =============================================================================

def validate_column_exists(
    df: pd.DataFrame,
    col: str,
    param_name: str = "column",
) -> None:
    """Raise ValueError if *col* is not in *df*."""
    if col not in df.columns:
        close = _close_matches(col, df.columns)
        hint = f"  Did you mean: {close}" if close else ""
        raise ValueError(f"Column '{col}' (param '{param_name}') not found in DataFrame.{hint}")

def validate_columns_exist(
    df: pd.DataFrame,
    cols: Collection[str],
    param_name: str = "columns",
) -> None:
    """Raise ValueError listing every missing column."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from DataFrame (param '{param_name}'): {missing}"
        )

# =============================================================================
# TYPE CHECKS
# =============================================================================

def validate_dataframe(obj: Any, param_name: str = "df") -> None:
    """Raise TypeError if *obj* is not a pd.DataFrame."""
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame for '{param_name}', got {type(obj).__name__}.")

def validate_series(obj: Any, param_name: str = "y") -> None:
    """Raise TypeError if *obj* is not a pd.Series."""
    if not isinstance(obj, pd.Series):
        raise TypeError(f"Expected pd.Series for '{param_name}', got {type(obj).__name__}.")

def validate_numeric_column(
    df: pd.DataFrame,
    col: str,
    param_name: str = "column",
) -> None:
    """Raise TypeError if column *col* is not numeric."""
    validate_column_exists(df, col, param_name)
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(
            f"Column '{col}' must be numeric but has dtype '{df[col].dtype}'."
        )

# =============================================================================
# VALUE RANGE / CARDINALITY
# =============================================================================

def validate_positive_int(value: Any, param_name: str) -> None:
    """Raise ValueError if *value* is not a positive integer."""
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"'{param_name}' must be a positive integer, got {value!r}.")


def validate_fraction(value: Any, param_name: str) -> None:
    """Raise ValueError if *value* is not in (0, 1)."""
    if not isinstance(value, (int, float, np.floating)) or not (0 < value < 1):
        raise ValueError(f"'{param_name}' must be in (0, 1), got {value!r}.")


def validate_min_samples(
    n: int,
    minimum: int = 2,
    context: str = "",
) -> None:
    """Raise ValueError if sample count is below *minimum*."""
    if n < minimum:
        msg = f"At least {minimum} samples required"
        if context:
            msg += f" ({context})"
        msg += f", got {n}."
        raise ValueError(msg)


# =============================================================================
# ALLOWED VALUES
# =============================================================================


def validate_in_set(
    value: Any,
    allowed: Collection[Any],
    param_name: str,
) -> None:
    """Raise ValueError if *value* is not in *allowed*."""
    if value not in allowed:
        raise ValueError(
            f"'{param_name}' must be one of {sorted(allowed)}, got {value!r}."
        )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _close_matches(name: str, candidates: Collection[str], n: int = 3) -> list:
    return difflib.get_close_matches(name, candidates, n=n, cutoff=0.6)
