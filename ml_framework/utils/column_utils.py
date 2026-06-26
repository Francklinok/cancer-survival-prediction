"""
column_utils.py — Column type detection utilities for the ml_framework.

Provides a single, consistent column classification API across all modules,
replacing the various ad-hoc `_num_cols()` / `_cat_cols()` / inline

Functions
---------
  get_numeric_columns(df, exclude, min_unique)  → List[str]
  get_categorical_columns(df, exclude)          → List[str]
  split_column_types(df, exclude, min_unique)   → (num_cols, cat_cols)
  infer_target_type(y)                          → 'binary' | 'multiclass' | 'continuous'
  has_sufficient_unique(s, min_unique)          → bool
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# COLUMN DETECTION
# =============================================================================


def get_numeric_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
    min_unique: int = 2,
) -> List[str]:
    """
    Return numeric columns with at least *min_unique* distinct values.

    Parameters
    ----------
    df         : input DataFrame
    exclude    : columns to skip (e.g. the target column)
    min_unique : minimum number of unique non-NaN values to be included

    Returns
    -------
    List of column names in the original DataFrame order.
    """
    exclude_set = set(exclude or [])
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_set
        and df[c].nunique(dropna=True) >= min_unique
    ]


def get_categorical_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
    max_unique: int = 200,
) -> List[str]:
    """
    Return object / category / bool columns.

    Numeric columns with <= 20 unique values are NOT included here;
    the caller must decide whether to treat them as categorical.

    Parameters
    ----------
    df         : input DataFrame
    exclude    : columns to skip
    max_unique : columns with more unique values than this are dropped
                 (likely free-text / IDs that slipped through)

    Returns
    -------
    List of column names in the original DataFrame order.
    """
    exclude_set = set(exclude or [])
    return [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c not in exclude_set
        and df[c].nunique(dropna=True) <= max_unique
    ]


def split_column_types(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
    min_unique: int = 2,
    max_cat_unique: int = 200,
) -> Tuple[List[str], List[str]]:
    """
    Convenience wrapper returning (numeric_cols, categorical_cols) together.

    Guarantees no column appears in both lists and no excluded column appears
    in either list.
    """
    num_cols = get_numeric_columns(df, exclude=exclude, min_unique=min_unique)
    cat_cols = get_categorical_columns(df, exclude=exclude, max_unique=max_cat_unique)
    return num_cols, cat_cols


def get_low_cardinality_numeric(
    df: pd.DataFrame,
    threshold: int = 20,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Return numeric columns that likely represent categories (< threshold unique values).
    Useful for deciding whether to treat them as ordinal or categorical.
    """
    exclude_set = set(exclude or [])
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_set
        and 1 < df[c].nunique(dropna=True) < threshold
    ]


# =============================================================================
# TARGET TYPE INFERENCE
# =============================================================================


def infer_target_type(y: pd.Series) -> str:
    """
    Classify the ML problem type from the target Series.

    Returns
    -------
    'binary'      — exactly 2 unique values
    'multiclass'  — 3-20 unique values (or numeric with <= 20 unique)
    'continuous'  — numeric with > 20 unique values
    """
    n_unique = y.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(y)

    if n_unique == 2:
        return "binary"
    if not is_numeric or n_unique <= 20:
        return "multiclass"
    return "continuous"


# =============================================================================
# MISC
# =============================================================================


def has_sufficient_unique(s: pd.Series, min_unique: int = 2) -> bool:
    """Return True if *s* has at least *min_unique* distinct non-NaN values."""
    return int(s.nunique(dropna=True)) >= min_unique


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with only one unique (non-NaN) value.

    Returns
    -------
    (cleaned_df, dropped_column_names)
    """
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=constant), constant
