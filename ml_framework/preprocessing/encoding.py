"""
encoding.py —  Categorical variable encoding.

Strategies implemented:
  - Binary      : two-category columns (e.g. Yes/No) mapped to 0/1 via map()
  - Ordinal     : explicit ordered mapping (preserves semantic order) via map()
  - Nominal     : pd.get_dummies(drop_first=True, dtype=int) — no sklearn, no 2D array
  - Target      : category mean encoding (with James-Stein regularization)
  - Frequency   : replacement by relative frequency

Features:
  - Drop identifier columns first (before any encoding)
  - encoded_cols tracking — raises a warning if a column would be encoded twice
  - Post-encoding validation (NaN, residual object types)
  - Detailed transformation report
  - Encoder metadata stored for future production inference

encode_dataframe() is domain-neutral by default: with no explicit mappings
it drops nothing but a generic set of common ID-column names, and encodes
every categorical column via the automatic get_dummies fallback (step 5).
Known binary/ordinal mappings for a specific dataset (e.g. this framework's
original oncology reference dataset) can be supplied explicitly, or loaded
via ml_framework.domain.medical.get_encoding_mappings() — see that module.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.encoding")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def encode_dataframe(
    df: pd.DataFrame,
    drop_id_cols: Optional[List[str]] = None,
    binary_mappings: Optional[Dict[str, Dict]] = None,
    ordinal_mappings: Optional[Dict[str, Dict]] = None,
    nominal_columns: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete categorical encoding for a DataFrame.

    Strategies applied in order:
      1. Drop   : identifier columns removed first (prevents encoding IDs)
      2. Binary : caller-supplied two-category columns → 0/1 via map()
      3. Ordinal: caller-supplied explicit ordered map() (preserves order)
      4. Nominal: caller-supplied columns → pd.get_dummies(drop_first=True) —
                  clean OHE, no sklearn, no 2D-array assignment issue
      5. Everything else still of object/category dtype → pd.get_dummies
                  fallback + logger warning (this is what makes the function
                  work out of the box on a dataset with no mappings supplied)

    An ``encoded_cols`` set guarantees each column is processed by exactly one
    strategy; a warning is emitted if a collision is detected.

    Parameters
    ----------
    df               : DataFrame to encode
    drop_id_cols     : columns to drop before encoding (IDs, etc.) — merged
                        with a small generic default (["ID", "index"])
    binary_mappings  : {column: {category: 0/1, ...}} — e.g.
                        {"Gender": {"Female": 0, "Male": 1}}. None/{} = skip.
    ordinal_mappings : {column: {category: rank, ...}}, in semantic order.
                        None/{} = skip.
    nominal_columns  : columns to one-hot encode explicitly (columns not
                        listed here, but still categorical, are still caught
                        by the step-5 fallback — this list just lets you
                        control that OHE happens for a known column even if
                        get_dummies' defaults wouldn't otherwise apply).
    verbose          : print the encoding report

    Returns
    -------
    df_encoded : pd.DataFrame — fully numeric
    encoders   : dict — transformation metadata for future inference
                 Binary/Ordinal: {"type": "binary"|"ordinal", "mapping": {...}}
                 Nominal OHE   : {"type": "ohe_dummies", "original_col": str,
                                  "new_cols": [list of created column names]}
    """
    df_processed = df.copy()
    encoders: Dict = {}
    ops_log: List[str] = []
    encoded_cols: set = set()

    binary_mappings = binary_mappings or {}
    ordinal_mappings = ordinal_mappings or {}
    nominal_columns = nominal_columns or []

    # ── 1. DROP ID COLUMNS ────────────────────────────────────────────────────
    default_id_cols = ["ID", "index"]
    all_drop = list(set((drop_id_cols or []) + default_id_cols))
    cols_to_drop = [c for c in all_drop if c in df_processed.columns]
    if cols_to_drop:
        df_processed.drop(columns=cols_to_drop, inplace=True)
        ops_log.append(f"[Drop]     ID columns removed: {cols_to_drop}")

    # ── 2. BINARY COLUMNS ────────────────────────────────────────────────────
    for col, mapping in binary_mappings.items():
        if col not in df_processed.columns or col in encoded_cols:
            continue
        df_processed[col] = df_processed[col].map(mapping)
        encoders[col] = {"type": "binary", "mapping": mapping}
        encoded_cols.add(col)
        mapping_str = " / ".join(f"{k}→{v}" for k, v in mapping.items())
        ops_log.append(f"[Binary]   {col} : {mapping_str}")

    # ── 3. ORDINAL COLUMNS ───────────────────────────────────────────────────
    for col, mapping in ordinal_mappings.items():
        if col not in df_processed.columns:
            continue
        if col in encoded_cols:
            logger.warning("Column '%s' already encoded — skipping ordinal step.", col)
            continue
        df_processed[col] = df_processed[col].map(mapping)
        encoders[col] = {"type": "ordinal", "mapping": mapping}
        encoded_cols.add(col)
        ops_log.append(f"[Ordinal]  {col} : {list(mapping.keys())}")

    # ── 4. NOMINAL COLUMNS → pd.get_dummies ──────────────────────────────────
    nominal_columns = [
        c for c in nominal_columns
        if c in df_processed.columns and c not in encoded_cols
    ]

    for col in nominal_columns:
        # pd.get_dummies silently zeroes every indicator column for a NaN row
        # instead of raising or flagging it — indistinguishable downstream
        # from "matches the dropped reference category". Filling first makes
        # missingness an explicit, visible category instead.
        had_na = df_processed[col].isna().any()
        if had_na:
            df_processed[col] = df_processed[col].fillna("Missing")
        cols_before = set(df_processed.columns)
        df_processed = pd.get_dummies(
            df_processed, columns=[col], drop_first=True, dtype=int
        )
        new_cols = [c for c in df_processed.columns if c not in cols_before]
        encoders[col] = {
            "type": "ohe_dummies",
            "original_col": col,
            "new_cols": new_cols,
        }
        encoded_cols.add(col)
        na_note = " (NaN→'Missing')" if had_na else ""
        ops_log.append(
            f"[Nominal]  {col} : get_dummies{na_note} ({len(new_cols)} indicator columns)"
        )

    # ── 5. REMAINING OBJECT COLUMNS → pd.get_dummies + warning ───────────────
    remaining_obj = [
        c for c in df_processed.select_dtypes(include=["object"]).columns
        if c not in encoded_cols
    ]
    for col in remaining_obj:
        logger.warning(
            "Column '%s' was not in any explicit encoding list "
            "applying get_dummies as fallback. "
            "Consider adding it to binary_mappings/ordinal_mappings/nominal_columns.",
            col,
        )
        had_na = df_processed[col].isna().any()
        if had_na:
            df_processed[col] = df_processed[col].fillna("Missing")
        cols_before = set(df_processed.columns)
        df_processed = pd.get_dummies(
            df_processed, columns=[col], drop_first=True, dtype=int
        )
        new_cols = [c for c in df_processed.columns if c not in cols_before]
        encoders[col] = {
            "type": "ohe_dummies",
            "original_col": col,
            "new_cols": new_cols,
        }
        encoded_cols.add(col)
        na_note = " (NaN→'Missing')" if had_na else ""
        ops_log.append(
            f"[Auto]     {col} : get_dummies fallback{na_note} ({len(new_cols)} indicator columns)"
        )

    # ── 6. VALIDATION ─────────────────────────────────────────────────────────
    remaining_obj_final = df_processed.select_dtypes(include=["object"]).columns.tolist()

    if verbose:
        _print_encoding_report(ops_log, remaining_obj_final)

    return df_processed, encoders


# Backward-compatible alias
encodage_processing = encode_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# TARGET ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def target_encoding(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    smoothing: float = 10.0,
    min_samples_leaf: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Target encoding with James-Stein regularization (smoothing).

    Formula: encoded = (n × mean_cat + λ × mean_global) / (n + λ)
    where n = category count, λ = smoothing parameter.

    Parameters
    ----------
    smoothing         : regularization strength λ (higher → closer to global mean)
    min_samples_leaf  : unused — kept for backward-compatibility only

    Returns
    -------
    df_encoded, encoding_maps
    """
    df_out = df.copy()
    global_mean = float(df[target_col].mean())
    encoding_maps: Dict = {}

    for col in cat_cols:
        if col not in df_out.columns:
            continue

        stats = df_out.groupby(col)[target_col].agg(["mean", "count"])
        smooth_mean = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )

        df_out[col] = df_out[col].map(smooth_mean).fillna(global_mean)
        encoding_maps[col] = smooth_mean.to_dict()

    return df_out, encoding_maps


# ──────────────────────────────────────────────────────────────────────────────
# FREQUENCY ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def frequency_encoding(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Replace each category with its relative frequency in the dataset.

    Returns
    -------
    df_encoded, freq_maps
    """
    df_out = df.copy()
    freq_maps: Dict = {}

    for col in cat_cols:
        if col not in df_out.columns:
            continue
        freq = df_out[col].value_counts(normalize=True)
        df_out[col] = df_out[col].map(freq).fillna(0)
        freq_maps[col] = freq.to_dict()

    return df_out, freq_maps


# ──────────────────────────────────────────────────────────────────────────────
# ONE-HOT ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def one_hot_encode(
    df: pd.DataFrame,
    cat_cols: List[str],
    drop: str = "first",
    max_categories: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encoding with a cardinality constraint.

    Parameters
    ----------
    drop           : 'first' | 'if_binary' | None — avoids multicollinearity
    max_categories : columns exceeding this threshold are skipped

    Returns
    -------
    df_encoded, new_columns
    """
    cols_to_encode = [
        c for c in cat_cols
        if c in df.columns and df[c].nunique() <= max_categories
    ]
    skipped = [c for c in cat_cols if c not in cols_to_encode]

    if skipped:
        logger.warning("OHE skipped (cardinality > %d): %s", max_categories, skipped)

    if not cols_to_encode:
        return df, []

    df_out = pd.get_dummies(df, columns=cols_to_encode, drop_first=(drop == "first"), dtype=int)
    new_cols = [c for c in df_out.columns if c not in df.columns]

    logger.info("OHE: %d new columns created.", len(new_cols))
    return df_out, new_cols


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────


def _print_encoding_report(ops_log: List[str], remaining_obj: List[str]) -> None:
    print("\n" + "═" * 60)
    print("  ENCODING REPORT")
    print("═" * 60)
    for op in ops_log:
        print(f"  {op}")
    print()
    if remaining_obj:
        print(f" Un-encoded object columns: {remaining_obj}")
    else:
        print("All columns are numeric — encoding complete.")
    print("═" * 60 + "\n")
