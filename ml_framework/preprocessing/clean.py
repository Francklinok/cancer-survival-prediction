"""
clean.py — General DataFrame cleaning pipeline.

Features:
  - Removal of ID columns, quasi-constant columns, high-cardinality columns
  - Duplicate row removal
  - Data type correction (object → numeric)
  - String whitespace trimming
  - Column removal based on NaN threshold
  - Detailed before/after cleaning report
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.clean")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLEANING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────


def clean_dataframe(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
    max_missing_ratio: float = 0.60,
    min_variance: float = 1e-8,
    drop_duplicates: bool = True,
    strip_strings: bool = True,
    infer_types: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Complete non-destructive cleaning pipeline for a DataFrame.

    Parameters
    ----------
    df               : DataFrame to clean
    id_cols          : explicit identifier columns to drop
    max_missing_ratio: NaN threshold — columns exceeding this ratio are dropped
    min_variance     : minimum variance to retain a numeric column
    drop_duplicates  : remove duplicate rows
    strip_strings    : strip whitespace from string columns
    infer_types      : attempt to convert object columns to numeric
    verbose          : print the cleaning report

    Returns
    -------
    pd.DataFrame — cleaned DataFrame (copy)
    """
    df_clean = df.copy()
    shape_init = df_clean.shape
    ops_log: List[str] = []

    # 1. Drop explicit ID columns
    if id_cols:
        cols_to_drop = [c for c in id_cols if c in df_clean.columns]
        if cols_to_drop:
            df_clean.drop(columns=cols_to_drop, inplace=True)
            ops_log.append(f"ID columns dropped: {cols_to_drop}")

    # 2. Strip string columns
    if strip_strings:
        obj_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df_clean[col] = df_clean[col].str.strip()
        if len(obj_cols) > 0:
            ops_log.append(f"Whitespace stripped from {len(obj_cols)} string column(s).")

    # 3. Type conversion (object → numeric where possible)
    if infer_types:
        n_converted = 0
        for col in df_clean.select_dtypes(include=["object"]).columns:
            try:
                converted = pd.to_numeric(df_clean[col], errors="coerce")
                if converted.notna().mean() > 0.80:  # > 80% convertible
                    df_clean[col] = converted
                    n_converted += 1
            except Exception:
                pass
        if n_converted > 0:
            ops_log.append(f"Object→numeric conversion: {n_converted} column(s)")

    # 4. Drop columns with too many missing values
    missing_ratio = df_clean.isnull().mean()
    high_missing = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    if high_missing:
        df_clean.drop(columns=high_missing, inplace=True)
        ops_log.append(
            f"Columns > {max_missing_ratio*100:.0f}% NaN dropped: {high_missing}"
        )

    # 5. Drop constant / near-constant columns
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    low_var = [c for c in num_cols if df_clean[c].std() < min_variance]
    if low_var:
        df_clean.drop(columns=low_var, inplace=True)
        ops_log.append(f"Near-zero variance columns dropped: {low_var}")

    # Object columns with a single unique value
    obj_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    constant_obj = [c for c in obj_cols if df_clean[c].nunique(dropna=True) <= 1]
    if constant_obj:
        df_clean.drop(columns=constant_obj, inplace=True)
        ops_log.append(f"Constant categorical columns dropped: {constant_obj}")

    # 6. Remove duplicate rows
    if drop_duplicates:
        n_dup = df_clean.duplicated().sum()
        if n_dup > 0:
            df_clean.drop_duplicates(inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
            ops_log.append(f"{n_dup} duplicate row(s) removed.")

    # 7. Reset index
    df_clean.reset_index(drop=True, inplace=True)

    if verbose:
        _print_clean_report(shape_init, df_clean.shape, ops_log)

    return df_clean


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def _print_clean_report(
    shape_init: tuple,
    shape_final: tuple,
    ops_log: List[str],
) -> None:
    print("\n" + "═" * 60)
    print("  CLEANING REPORT")
    print("═" * 60)
    print(f"  Before : {shape_init[0]:,} rows × {shape_init[1]} columns")
    print(f"  After  : {shape_final[0]:,} rows × {shape_final[1]} columns")
    delta_rows = shape_init[0] - shape_final[0]
    delta_cols = shape_init[1] - shape_final[1]
    if delta_rows:
        print(f"  Rows removed   : {delta_rows:,}")
    if delta_cols:
        print(f"  Columns removed: {delta_cols}")
    print()
    for op in ops_log:
        print(f"  ✓ {op}")
    if not ops_log:
        print("  ✅ No modifications needed — data already clean.")
    print("═" * 60 + "\n")
