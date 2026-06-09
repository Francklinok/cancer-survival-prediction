"""
apply_normalisation.py — Application of normalization transformations and scalers.

Reads a strategies_df produced by suggest_normalization_strategy() and applies
sequentially for each column:
  1. Optional winsorization (clip to percentiles)
  2. Mathematical transformation (log, sqrt, boxcox, yeojohnson, identity)
  3. Scaling (standard, minmax, robust, maxabs)

Public functions:
  - apply_normalization(df, strategies_df, config, copy_original) → (original, normalized, log)
  - _apply_transform(series, transform, config)  → (series, description)
  - _apply_scaler(series, scaler_type)           → (series, description)
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ml_framework.config.config import NormalizationConfig

logger = logging.getLogger("ml_framework.apply_normalisation")


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ──────────────────────────────────────────────────────────────────────────────


def _apply_transform(
    series: pd.Series,
    transform: Optional[str],
    config: NormalizationConfig,
) -> Tuple[pd.Series, str]:
    """
    Apply a mathematical transformation to a series.

    Parameters
    ----------
    series    : pd.Series — data to transform
    transform : str or None — 'log', 'sqrt', 'boxcox', 'yeojohnson', None
    config    : NormalizationConfig

    Returns
    -------
    (transformed_series, description_str)
    """
    if transform is None:
        return series, "identity"

    s = series.copy()
    name = transform.lower().strip()

    if name == "log":
        min_val = s.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            s = s + shift
            s = np.log(s)
            return s, f"log(+{shift:.4f})"
        else:
            return np.log(s), "log"

    elif name == "sqrt":
        min_val = s.min()
        shift = abs(min_val) + 1e-6 if min_val < 0 else 0.0
        if shift > 0:
            s = s + shift
        s = np.sqrt(s)
        return s, f"sqrt(+{shift:.4f})" if shift > 0 else "sqrt"

    elif name == "boxcox":
        # Box-Cox requires strictly positive values
        min_val = s.min()
        shift = abs(min_val) + 1e-6 if min_val <= 0 else 0.0
        if shift > 0:
            s = s + shift
        try:
            transformed, lam = stats.boxcox(s.dropna())
            # Rebuild preserving NaN positions — use .loc with original index, not boolean mask
            result = s.copy().astype(float)
            result.loc[s.notna()] = transformed
            return result, f"boxcox(λ={lam:.3f})"
        except Exception as exc:
            logger.warning("boxcox failed on '%s': %s — fallback to log", series.name, exc)
            shift = abs(s.min()) + 1e-6 if s.min() <= 0 else 0.0
            return np.log(s + shift) if shift > 0 else np.log(s), "log(boxcox_fallback)"

    elif name == "yeojohnson":
        try:
            transformed, lam = stats.yeojohnson(s.dropna())
            result = s.copy().astype(float)
            result.loc[s.notna()] = transformed
            return result, f"yeojohnson(λ={lam:.3f})"
        except Exception as exc:
            logger.warning(
                "yeojohnson failed on '%s': %s — fallback to identity", series.name, exc
            )
            return series, "identity(yeojohnson_fallback)"

    elif name == "identity":
        return series, "identity"

    else:
        logger.warning("Unknown transformation: '%s' — ignored.", transform)
        return series, "identity"


# ──────────────────────────────────────────────────────────────────────────────
# SCALERS
# ──────────────────────────────────────────────────────────────────────────────


def _apply_scaler(
    series: pd.Series,
    scaler_type: Optional[str],
) -> Tuple[pd.Series, str]:
    """
    Apply a scaling method to a series.

    Parameters
    ----------
    series      : pd.Series
    scaler_type : 'standard', 'minmax', 'robust', 'maxabs', None

    Returns
    -------
    (scaled_series, description_str)
    """
    if scaler_type is None:
        return series, "identity"

    s = series.copy()
    name = scaler_type.lower().strip()

    if name == "standard":
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            return s - mu, "center_only"
        return (s - mu) / sigma, "StandardScaler"

    elif name == "minmax":
        mn, mx = s.min(), s.max()
        rng = mx - mn
        if rng == 0:
            return pd.Series(np.zeros(len(s)), index=s.index), "minmax_zero_range"
        return (s - mn) / rng, "MinMaxScaler"

    elif name == "robust":
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        if iqr == 0:
            return s - med, "center_only(robust)"
        return (s - med) / iqr, "RobustScaler"

    elif name == "maxabs":
        max_abs = s.abs().max()
        if max_abs == 0:
            return s, "identity(maxabs_zero)"
        return s / max_abs, "MaxAbsScaler"

    else:
        logger.warning("Unknown scaler: '%s' — ignored.", scaler_type)
        return series, "identity"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────


def apply_normalization(
    df: pd.DataFrame,
    strategies_df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
    copy_original: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Apply normalization strategies to a DataFrame.

    For each row in strategies_df:
      1. Winsorization (if winsorize=True)
      2. Mathematical transformation (log, sqrt, boxcox, yeojohnson)
      3. Scaling (standard, minmax, robust, maxabs)

    Parameters
    ----------
    df             : source DataFrame
    strategies_df  : result of suggest_normalization_strategy()
                     expected columns: column, transform, scaler, winsorize
    config         : NormalizationConfig (default if None)
    copy_original  : keep a copy of the original unmodified data

    Returns
    -------
    (df_original, df_normalized, transformation_log)
      - df_original      : copy of DataFrame before transformation
      - df_normalized    : DataFrame after transformation
      - transformation_log : dict {col_name: description_str}
    """
    if config is None:
        config = NormalizationConfig()

    df_original   = df.copy() if copy_original else df
    df_normalized = df.copy()
    transformation_log: Dict[str, str] = {}

    # named indices, and legacy 'colonne' aliases.
    strats = strategies_df.copy()

    # Deduplicate columns: if 'column' appears more than once, keep first occurrence only
    strats = strats.loc[:, ~strats.columns.duplicated()]

    # If 'column' is the index name (not a regular column), promote it
    if strats.index.name in ("column", "colonne") and "column" not in strats.columns:
        strats = strats.reset_index()

    # Legacy alias: rename 'colonne' → 'column' if 'column' still absent
    if "column" not in strats.columns and "colonne" in strats.columns:
        strats = strats.rename(columns={"colonne": "column"})

    # Last resort: the integer index holds row numbers — column names must be in a column
    if "column" not in strats.columns:
        raise ValueError(
            "strategies_df has no 'column' column. "
            "Pass the direct output of suggest_normalization_strategy()."
        )

    for _, strategy_row in strats.iterrows():
        raw = strategy_row["column"]
        # Scalar extraction — guard against any residual Series leakage
        col = raw.iloc[0] if isinstance(raw, pd.Series) else str(raw)
        transform    = strategy_row["transform"]
        scaler_type  = strategy_row["scaler"]
        do_winsorize = strategy_row.get("winsorize", False)

        # Pandas stores None as float('nan') in object columns
        transform    = None if (transform    is None or (isinstance(transform,    float) and np.isnan(transform)))    else transform
        scaler_type  = None if (scaler_type  is None or (isinstance(scaler_type,  float) and np.isnan(scaler_type)))  else scaler_type
        do_winsorize = bool(do_winsorize) if not (isinstance(do_winsorize, float) and np.isnan(do_winsorize)) else False

        # Column absent from DataFrame
        if col not in df.columns:
            transformation_log[col] = "SKIP — column absent from DataFrame"
            logger.debug("Column '%s' absent — skipped.", col)
            continue

        series = df_normalized[col].copy()

        # Empty series
        if series.dropna().empty:
            transformation_log[col] = "SKIP — empty series"
            continue

        # No transformation required
        if transform is None and scaler_type is None:
            transformation_log[col] = "No transformation"
            continue

        try:
            # 1. Winsorization
            if do_winsorize:
                q_low  = series.quantile(config.winsorize_lower)
                q_high = series.quantile(config.winsorize_upper)
                series = series.clip(q_low, q_high)
                winsz_desc = (
                    f"Winsorize({config.winsorize_lower*100:.0f}%-"
                    f"{config.winsorize_upper*100:.0f}%)"
                )
            else:
                winsz_desc = ""

            # 2. Transformation
            series, t_desc = _apply_transform(series, transform, config)

            # 3. Scaler
            series, s_desc = _apply_scaler(series, scaler_type)

            # Write back — preserve pandas index alignment (never use .values here)
            df_normalized[col] = series

            parts = [p for p in [winsz_desc, t_desc, s_desc] if p and p != "identity"]
            transformation_log[col] = " + ".join(parts) if parts else "No transformation"

        except Exception as exc:
            logger.error("Normalization error on column '%s': %s", col, exc, exc_info=True)
            transformation_log[col] = f"ERROR: {exc}"
            df_normalized[col] = df[col].copy()  # rollback

    n_transformed = sum(
        1 for v in transformation_log.values()
        if "SKIP" not in v and v != "No transformation" and "ERROR" not in v
    )
    logger.info(
        "Normalization applied: %d/%d columns transformed.",
        n_transformed, len(strats),
    )

    _print_normalization_quality_report(
        df, df_normalized, transformation_log, config.alpha
    )

    return df_original, df_normalized, transformation_log


# ──────────────────────────────────────────────────────────────────────────────
# POST-NORMALIZATION QUALITY REPORT
# ──────────────────────────────────────────────────────────────────────────────


def _print_normalization_quality_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    transformation_log: Dict[str, str],
    alpha: float = 0.05,
    sw_sample_size: int = 5_000,
) -> None:
    """
    Post-normalization quality report using skewness + Shapiro-Wilk on a sample.

    ── Statistical note on Shapiro-Wilk at large n ────────────────────────────
    With n > 5 000, Shapiro-Wilk ALWAYS rejects normality (p ≈ 0) because its
    power grows with n and it detects negligibly small deviations that are
    irrelevant in practice.  This is a well-known limitation of the test, NOT a
    failure of the normalization.

    For large datasets the relevant metric is skewness reduction:
      |skew| < 0.5  → distribution is practically symmetric → transformation OK
      |skew| < 1.0  → acceptable for most ML models
      |skew| ≥ 1.0  → still significantly skewed, consider stronger transform

    Shapiro p-values are shown only when n ≤ 5 000 and are reliable.
    For n > 5 000, skewness is the primary quality indicator.
    ────────────────────────────────────────────────────────────────────────────
    """
    from scipy.stats import shapiro as _shapiro

    transformed_cols = [
        col for col, desc in transformation_log.items()
        if "SKIP" not in desc and desc != "No transformation" and "ERROR" not in desc
        and col in df_before.columns and col in df_after.columns
    ]

    if not transformed_cols:
        print("\n  No columns were transformed — quality report skipped.\n")
        return

    # Determine if dataset is large (Shapiro unreliable)
    n_rows = len(df_before)
    large_n = n_rows > sw_sample_size

    W = 86
    print()
    print("╔" + "═" * W + "╗")
    print(f"║{'  POST-NORMALIZATION QUALITY REPORT':^{W}}║")
    if large_n:
        print(f"║{' n=' + str(n_rows) + ' > 5000 — Shapiro-Wilk always rejects at this size (expected)':^{W}}║")
        # print(f"║{'  Primary metric: skewness reduction  |skew| < 0.5 =  < 1.0 =   ≥ 1.0 = ':^{W}}║")
    else:
        print(f"║{'  Shapiro-Wilk re-tested  (α = ' + str(alpha) + ')  +  skewness reduction':^{W}}║")
    print("╠" + "═" * W + "╣")

    hdr = f"  {'Column':<30} {'Transform':<26} {'Skew before':^12} {'Skew after':^12} {'SW after':^10}"
    print(f"║{hdr:<{W}}║")
    print(f"║{'  ' + '─' * (W - 2):<{W}}║")

    skew_ok = 0
    skew_improved = 0
    still_skewed: List[str] = []

    for col in transformed_cols:
        s_before = df_before[col].dropna()
        s_after  = df_after[col].dropna()

        sk_before = float(s_before.skew()) if len(s_before) > 2 else float("nan")
        sk_after  = float(s_after.skew())  if len(s_after)  > 2 else float("nan")

        # Skewness verdict
        def _skew_tag(sk: float) -> str:
            if np.isnan(sk):   return "  —  "
            if abs(sk) < 0.5:  return f"{sk:+.2f}"
            if abs(sk) < 1.0:  return f"{sk:+.2f}"
            return             f"{sk:+.2f}"

        sk_b_tag = _skew_tag(sk_before)
        sk_a_tag = _skew_tag(sk_after)

        # Shapiro — only meaningful for small n; show "n/a" for large datasets
        if large_n:
            sw_tag = "  n/a  "
        elif len(s_after) >= 3:
            sample = s_after.sample(min(len(s_after), sw_sample_size), random_state=42)
            try:
                _, p = _shapiro(sample)
                sw_tag = f"p={p:.3f}" if p > alpha else f"p={p:.3f}"
            except Exception:
                sw_tag = "  ERR  "
        else:
            sw_tag = "  —   "

        desc = transformation_log[col][:24]
        line = f"  {col:<30} {desc:<26} {sk_b_tag:^12} {sk_a_tag:^12} {sw_tag:^10}"
        print(f"║{line:<{W}}║")

        if not np.isnan(sk_after):
            if abs(sk_after) < 0.5:
                skew_ok += 1
            if not np.isnan(sk_before) and abs(sk_after) < abs(sk_before):
                skew_improved += 1
            if abs(sk_after) >= 1.0:
                still_skewed.append((col, sk_before, sk_after))

    print("╠" + "═" * W + "╣")
    print(f"║  {'Skew |<0.5| (practically normal):':<44}{skew_ok:>4} / {len(transformed_cols):<4}          ║")
    print(f"║  {'Skew improved (reduced vs before):':<44}{skew_improved:>4} / {len(transformed_cols):<4}          ║")
    print(f"║  {'Still high skew |≥1.0| after transform:':<44}{len(still_skewed):>4}               ║")

    if large_n:
        print("╠" + "═" * W + "╣")
        note = " Shapiro p=0.000 at n>" + str(sw_sample_size) + " is EXPECTED — not a normalization failure."
        print(f"║{note:<{W}}║")
        note2 = "  Use skewness and kurtosis as quality metrics at this dataset size."
        print(f"║{note2:<{W}}║")

    if still_skewed:
        print("╠" + "═" * W + "╣")
        print(f"║{'High residual skew — consider yeojohnson or quantile transform:':^{W}}║")
        for c, sk_b, sk_a in still_skewed:
            line = f"    • {c:<30}  before={sk_b:+.2f}  after={sk_a:+.2f}"
            print(f"║{line:<{W}}║")

    print("╚" + "═" * W + "╝")
    print()
