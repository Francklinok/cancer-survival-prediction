"""
normalisation_strategy.py — Decision engine for normalization strategies.

Consumes the output of analyze_column_properties() (column_analysis.py) as the
single source of truth.  

Decision tree (in priority order)
----------------------------------
1. Zero / near-zero variance           → skip (no transform, no scaler)
2. Binary column (n_unique ≤ 2)        → skip
3. Already Normal + low skew/kurt      → StandardScaler, no transform
4. High outlier rate (> threshold)     → RobustScaler ± winsorize
5. Uniform distribution (kurt < -0.9)  → MinMaxScaler
6. Negligible skew (|skew| < 0.5)      → StandardScaler
7. Moderate positive skew, no neg/zero → log → StandardScaler  (aligned with transformation_rec)
8. Moderate skew with neg/zero         → Yeo-Johnson → StandardScaler
9. High skew → Box-Cox (positive) or Yeo-Johnson (negative values)
10. High kurtosis only                 → RobustScaler
11. Wide range / high CV               → MinMaxScaler or log

Public API
----------
suggest_normalization_strategy(analysis_df, config, skip_cols) → pd.DataFrame
decide_strategy(row, config)                                    → dict
validate_input(analysis_df)                                     → None
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_framework.config.config import NormalizationConfig

logger = logging.getLogger("ml_framework.normalisation_strategy")


# =============================================================================
# REQUIRED INPUT COLUMNS
# =============================================================================

REQUIRED_COLUMNS = {
    "column", "skewness", "kurtosis", "cv", "range",
    "outliers_pct", "has_negative", "has_zero",
    "min", "n_unique", "is_low_variance",
    # normality fields — provided by analyze_column_properties
    "normality_score", "normality_verdict",
}

_LEGACY_MAP = {"colonne": "column"}


# =============================================================================
# VALIDATION
# =============================================================================


def validate_input(df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("analysis_df must be a pd.DataFrame.")
    effective = {_LEGACY_MAP.get(c, c) for c in df.columns}
    missing = REQUIRED_COLUMNS - effective
    if missing:
        raise ValueError(
            f"Missing columns in analysis_df: {sorted(missing)}\n"
            "Make sure you pass the output of analyze_column_properties()."
        )
    if df.empty:
        raise ValueError("analysis_df is empty.")


def _normalize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=_LEGACY_MAP)
    # Drop duplicate columns that may arise if both 'column' and 'colonne' were present
    return renamed.loc[:, ~renamed.columns.duplicated()]


# =============================================================================
# DECISION ENGINE
# =============================================================================


def decide_strategy(row: pd.Series, config: NormalizationConfig) -> Dict[str, Any]:
    """
    Choose the best (transform, scaler, winsorize) for one column.

    Reads pre-computed fields from analyze_column_properties() — no independent
    tests are run.  Decision order is documented in the module docstring.
    """
    sk        = _f(row, "skewness")
    ku        = _f(row, "kurtosis")
    outliers  = _f(row, "outliers_pct", default=0.0)
    cv        = _f(row, "cv", default=0.0)
    has_neg   = bool(row.get("has_negative", False))
    has_zero  = bool(row.get("has_zero", False))
    verdict   = str(row.get("normality_verdict", "Non-normal"))
    score     = _f(row, "normality_score", default=0.0)
    range_val = _f(row, "range", default=0.0)

    def _result(transform, scaler, winsorize, reason_type, **kw):
        return {
            "transform": transform,
            "scaler":    scaler,
            "winsorize": winsorize,
            "reason": {"type": reason_type, "score": score, "skew": sk, "kurt": ku, **kw},
        }

    # 1. Zero / near-zero variance or binary column → skip entirely
    if row.get("is_low_variance", False) or row.get("n_unique", 10) <= 2:
        return _result(None, None, False, "low_variance_or_binary")

    # 2. Already Normal: low skew & kurt → StandardScaler only
    if (verdict == "Normal"
            and not pd.isna(sk) and abs(sk) < config.normal_skew_tol
            and not pd.isna(ku) and abs(ku) < config.normal_kurt_tol):
        return _result(None, "standard", False, "already_normal")

    # 3. High outlier rate → RobustScaler ± winsorize
    if not pd.isna(outliers) and outliers > config.outlier_threshold:
        return _result(None, "robust", not has_neg, "high_outliers", outliers_pct=outliers)

    # 4. Uniform distribution (platykurtic + symmetric) → MinMaxScaler
    # Strict guard: require very flat kurtosis AND confirmed non-normal verdict
    # to avoid misclassifying discrete/ordinal columns as uniform.
    if (not pd.isna(ku) and ku < -1.2
            and not pd.isna(sk) and abs(sk) < 0.15
            and verdict == "Non-normal"):
        return _result(None, "minmax", False, "uniform_distribution", kurt=ku)

    # 5. Negligible skew → StandardScaler baseline
    if not pd.isna(sk) and abs(sk) < 0.5:
        return _result(None, "standard", False, "low_skew_baseline", skew=sk)

    # From here: significant skew (|sk| >= 0.5) — build candidate list
    sk_f = sk if not pd.isna(sk) else 0.0
    ku_f = ku if not pd.isna(ku) else 0.0
    cv_f = cv if not pd.isna(cv) else 0.0

    candidates: List[Dict[str, Any]] = [
        {"name": "baseline", "transform": None, "scaler": "standard", "score": 1},
    ]

    # 6. Mild skew (0.5 – threshold) → sqrt
    if 0.5 < abs(sk_f) <= config.skew_threshold:
        candidates.append({"name": "mild_skew_sqrt", "transform": "sqrt", "scaler": "standard", "score": 2})

    # 7. Moderate positive skew, no negatives/zeros → log
    if sk_f > config.skew_threshold and not has_neg and not has_zero:
        if cv_f > config.cv_very_high:
            candidates.append({"name": "log_high_var", "transform": "log", "scaler": "standard", "score": 4})
        else:
            candidates.append({"name": "log_mod",     "transform": "log", "scaler": "standard", "score": 3})

    # 8. Moderate positive skew but has negatives/zeros → Yeo-Johnson
    if sk_f > config.skew_threshold and (has_neg or has_zero):
        candidates.append({"name": "yeojohnson_pos", "transform": "yeojohnson", "scaler": "standard", "score": 3})

    # 9. High skew → power transform
    if abs(sk_f) > config.high_skew_threshold:
        if has_neg:
            candidates.append({"name": "yeojohnson_high", "transform": "yeojohnson", "scaler": "standard", "score": 5})
        else:
            candidates.append({"name": "boxcox_high",     "transform": "boxcox",     "scaler": "standard", "score": 5})

    # 10. High kurtosis only → robust scaler
    if abs(ku_f) > config.kurtosis_threshold:
        candidates.append({"name": "robust_kurt", "transform": None, "scaler": "robust", "score": 3})

    # 11. Wide range or high CV
    if not pd.isna(range_val) and (range_val > config.range_threshold or cv_f > config.cv_high):
        if not has_neg:
            if abs(sk_f) > 1:
                candidates.append({"name": "log_wide_range", "transform": "log", "scaler": "standard", "score": 4})
            else:
                candidates.append({"name": "minmax_wide",    "transform": None,  "scaler": "minmax",   "score": 2})
        else:
            candidates.append({"name": "standard_wide", "transform": None, "scaler": "standard", "score": 2})

    best = max(candidates, key=lambda x: x["score"])
    return _result(best["transform"], best["scaler"], False, best["name"])


def _f(row: pd.Series, key: str, default: float = np.nan) -> float:
    """Safe float extraction from a Series row."""
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# =============================================================================
# MAIN API
# =============================================================================


def suggest_normalization_strategy(
    analysis_df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
    skip_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate normalization strategies from analyze_column_properties() output.

    Parameters
    ----------
    analysis_df : result of analyze_column_properties()
    config      : NormalizationConfig (default if None)
    skip_cols   : column names to force-skip (e.g. target, IDs)

    Returns
    -------
    pd.DataFrame with columns:
        column, colonne, transform, scaler, winsorize, reason
    Prints a detailed Normalization Summary to stdout.
    """
    if config is None:
        config = NormalizationConfig()

    analysis_df = _normalize_col_names(analysis_df)
    validate_input(analysis_df)

    skip_set = set(skip_cols or [])
    strategies = []
    skipped_cols: List[str] = []

    for _, row in analysis_df.iterrows():
        col = str(row["column"])

        if col in skip_set:
            skipped_cols.append(col)
            strategies.append({
                "column": col,
                "transform": None, "scaler": None,
                "winsorize": False,
                "reason": {"type": "user_skip"},
            })
            continue

        s = decide_strategy(row, config)
        strategies.append({
            "column":    col,
            "transform": s["transform"],
            "scaler":    s["scaler"],
            "winsorize": s["winsorize"],
            "reason":    s["reason"],
        })

    result_df = pd.DataFrame(strategies)

    _print_normalization_summary(result_df, analysis_df)

    logger.info(
        "Normalization strategies generated: %d columns | skipped: %d",
        len(result_df), len(skipped_cols),
    )

    return result_df


# =============================================================================
# SUMMARY REPORT
# =============================================================================


def _print_normalization_summary(
    result_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> None:
    W = 60
    SEP = "─" * W

    active = result_df[result_df["scaler"].notna() | result_df["transform"].notna()]
    ignored = result_df[result_df["scaler"].isna() & result_df["transform"].isna()]

    n_total    = len(result_df)
    n_active   = len(active)
    n_ignored  = len(ignored)

    # Scaler counts (active only)
    scaler_counts  = active["scaler"].value_counts().to_dict()
    # Transform counts (active only)
    transform_counts = active["transform"].dropna().value_counts().to_dict()
    # Normality verdict breakdown from analysis_df
    verdict_counts: Dict[str, int] = {}
    if "normality_verdict" in analysis_df.columns:
        verdict_counts = analysis_df["normality_verdict"].value_counts().to_dict()

    print()
    print("╔" + "═" * W + "╗")
    print(f"║{'  NORMALIZATION SUMMARY':^{W}}║")
    print("╠" + "═" * W + "╣")
    print(f"║  {'Total columns analyzed':<35}{n_total:>6}       ║")
    print(f"║  {'Columns to transform/scale':<35}{n_active:>6}       ║")
    print(f"║  {'Columns skipped (binary/low-var/user)':<35}{n_ignored:>6}       ║")
    print("╠" + "═" * W + "╣")

    # Normality breakdown
    if verdict_counts:
        print(f"║{'  Normality Verdicts':^{W}}║")
        print(f"║  {SEP[:W-2]}  ║")
        for v, icon in [("Normal", "✓"), ("Borderline", "~"), ("Non-normal", "✗")]:
            cnt = verdict_counts.get(v, 0)
            label = f"{icon} {v}"
            print(f"║    {label:<33}{cnt:>4}         ║")

    print("╠" + "═" * W + "╣")

    # Scalers
    print(f"║{'  Scalers':^{W}}║")
    print(f"║  {SEP[:W-2]}  ║")
    for s_name, display in [
        ("standard", "StandardScaler"),
        ("robust",   "RobustScaler  "),
        ("minmax",   "MinMaxScaler  "),
        ("maxabs",   "MaxAbsScaler  "),
    ]:
        cnt = scaler_counts.get(s_name, 0)
        bar = "█" * min(cnt, 20)
        print(f"║    {display:<18}{cnt:>4}   {bar:<20}  ║")

    print("╠" + "═" * W + "╣")

    # Transforms
    print(f"║{'  Transforms Applied':^{W}}║")
    print(f"║  {SEP[:W-2]}  ║")
    for t_name, display in [
        ("log",          "Log           "),
        ("sqrt",         "Sqrt          "),
        ("boxcox",       "Box-Cox       "),
        ("yeojohnson",   "Yeo-Johnson   "),
    ]:
        cnt = transform_counts.get(t_name, 0)
        bar = "█" * min(cnt, 20)
        print(f"║    {display:<18}{cnt:>4}   {bar:<20}  ║")

    n_winsorize = int(result_df["winsorize"].sum()) if "winsorize" in result_df.columns else 0
    print(f"║    {'Winsorization':<18}{n_winsorize:>4}                       ║")
    print("╠" + "═" * W + "╣")

    # Ignored / skipped columns
    print(f"║{'  Ignored Columns':^{W}}║")
    print(f"║  {SEP[:W-2]}  ║")
    ignored_names = ignored["column"].tolist() if "column" in ignored.columns else []
    if ignored_names:
        for name in ignored_names:
            reason_type = ""
            row_match = result_df[result_df["column"] == name]
            if not row_match.empty:
                r = row_match.iloc[0]["reason"]
                if isinstance(r, dict):
                    reason_type = r.get("type", "")
            tag = "(user skip)" if reason_type == "user_skip" else "(binary/low-var)"
            line = f"    • {name} {tag}"
            if len(line) > W - 2:
                line = line[:W - 5] + "..."
            print(f"║{line:<{W}}║")
    else:
        print(f"║    {'— none —':<{W-4}}║")

    print("╚" + "═" * W + "╝")
    print()
