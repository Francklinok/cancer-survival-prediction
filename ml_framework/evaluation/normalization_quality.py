"""
normalization_quality.py — Normalization quality evaluation.

Compares distributions before and after transformation:
  - Skewness and kurtosis (improvement)
  - Shapiro-Wilk normality test
  - Automatic interpretation

Public functions:
  - evaluate_normalization_quality(df_original, df_normalized, transformation_log) → pd.DataFrame
  - print_normalization_report(evaluation_df)                                       → None
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import shapiro

logger = logging.getLogger("ml_framework.normalization_quality")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _shapiro_p(arr: np.ndarray, max_n: int = 5_000) -> float:
    """
    Return Shapiro-Wilk p-value, or np.nan when n > max_n.

    At large n, Shapiro-Wilk reliably rejects H₀ even for trivial
    deviations — the p-value is meaningless as a normality gauge.
    Callers should fall back to skewness/kurtosis for large samples.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 3:
        return np.nan
    if n > max_n:
        return np.nan  
    try:
        _, p = shapiro(arr)
        return float(p)
    except Exception:
        return np.nan


def _interpret_improvement(
    skew_before: float,
    skew_after: float,
    p_before: float,
    p_after: float,
) -> str:
    """Return a textual interpretation of the normalization improvement."""
    skew_improved = abs(skew_after) < abs(skew_before)
    norm_improved = (
        not np.isnan(p_after) and not np.isnan(p_before) and p_after > p_before
    )

    if skew_improved and norm_improved:
        return " Clear improvement (skew + normality)"
    elif skew_improved:
        return " Skewness improved"
    elif norm_improved:
        return " Normality improved (Shapiro)"
    elif abs(skew_after) > abs(skew_before) * 1.1:
        return " Distribution degraded"
    else:
        return "≈ Little change"


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_normalization_quality(
    df_original: pd.DataFrame,
    df_normalized: pd.DataFrame,
    transformation_log: Dict[str, str],
    shapiro_sample: int = 5000,
) -> pd.DataFrame:
    """
    Compare distributions before and after normalization.

    Parameters
    ----------
    df_original       : DataFrame before transformation
    df_normalized     : DataFrame after transformation
    transformation_log: dict {col_name: transformation_description}
    shapiro_sample    : maximum sample size for Shapiro-Wilk test

    Returns
    -------
    pd.DataFrame with one row per transformed column and columns:
        column, transformation,
        skew_before, skew_after, skew_improvement,
        kurt_before, kurt_after, kurt_improvement,
        shapiro_p_before, shapiro_p_after,
        normality_improved, interpretation
    """
    results = []

    for col, transformation in transformation_log.items():
        # Skip untransformed or errored columns
        if any(tag in transformation for tag in ("SKIP", "None", "ERROR")):
            continue

        if col not in df_original.columns:
            logger.debug("Column '%s' missing from df_original — skipped.", col)
            continue

        # The column may share the same name (in-place) or carry a suffix
        col_norm = f"{col}_norm" if f"{col}_norm" in df_normalized.columns else col
        if col_norm not in df_normalized.columns:
            logger.debug("Normalized column '%s' not found.", col_norm)
            continue

        orig  = df_original[col].dropna()
        trans = df_normalized[col_norm].dropna()

        if len(orig) < 3 or len(trans) < 3:
            continue

        orig_skew  = float(orig.skew())
        trans_skew = float(trans.skew())
        orig_kurt  = float(orig.kurtosis())
        trans_kurt = float(trans.kurtosis())
        orig_p     = _shapiro_p(orig.values,  shapiro_sample)
        trans_p    = _shapiro_p(trans.values, shapiro_sample)

        # When sw_p is NaN (n > 5000), Shapiro-Wilk was skipped.
        # In that case, normality improvement is assessed via skewness and kurtosis only.
        interpretation = _interpret_improvement(orig_skew, trans_skew, orig_p, trans_p)

        results.append({
            "column":              col,
            "transformation":      transformation,
            "skew_before":         round(orig_skew,  4),
            "skew_after":          round(trans_skew, 4),
            "skew_improvement":    round(abs(orig_skew)  - abs(trans_skew), 4),
            "kurt_before":         round(orig_kurt,  4),
            "kurt_after":          round(trans_kurt, 4),
            "kurt_improvement":    round(abs(orig_kurt)  - abs(trans_kurt), 4),
            "shapiro_p_before":    round(orig_p,  6) if not np.isnan(orig_p)  else None,
            "shapiro_p_after":     round(trans_p, 6) if not np.isnan(trans_p) else None,
            "normality_improved":  bool((trans_p or 0.0) > (orig_p or 0.0)),
            "interpretation":      interpretation,
        })

    df_eval = pd.DataFrame(results)

    if not df_eval.empty:
        n_improved = df_eval["normality_improved"].sum()
        logger.info(
            "Normalization evaluation: %d/%d columns have improved normality.",
            n_improved, len(df_eval),
        )

    return df_eval


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────


def print_normalization_report(evaluation_df: pd.DataFrame) -> None:
    """
    Print a formatted normalization quality report.

    Parameters
    ----------
    evaluation_df : result of evaluate_normalization_quality()
    """
    if evaluation_df is None or evaluation_df.empty:
        print("  No columns evaluated (no transformations applied).")
        return

    print("\n" + "─" * 65)
    print("  NORMALIZATION QUALITY REPORT")
    print("─" * 65)

    fmt = "  {:<25} {:>8} {:>8} {:>8}  {}"
    print(fmt.format("Column", "Skew↓", "Kurt↓", "Shapiro↑", "Interpretation"))
    print("  " + "─" * 63)

    for _, row in evaluation_df.iterrows():
        skew_delta  = row["skew_improvement"]
        kurt_delta  = row["kurt_improvement"]
        shapiro_ok  = "yes" if row["normality_improved"] else "no"
        skew_symbol = "↑" if skew_delta > 0.01 else ("↓" if skew_delta < -0.01 else "≈")
        kurt_symbol = "↑" if kurt_delta > 0.01 else ("↓" if kurt_delta < -0.01 else "≈")

        print(fmt.format(
            str(row["column"])[:25],
            f"{skew_symbol}{abs(skew_delta):.3f}",
            f"{kurt_symbol}{abs(kurt_delta):.3f}",
            shapiro_ok,
            row["interpretation"],
        ))

    print("─" * 65)
    n_improved = evaluation_df["normality_improved"].sum()
    pct        = 100 * n_improved / max(len(evaluation_df), 1)
    print(f"  Normality improved: {n_improved}/{len(evaluation_df)} columns ({pct:.1f}%)")
    print()
