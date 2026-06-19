"""
column_analysis.py — Canonical statistical analysis of numeric columns.

Single source of truth for normality testing, distribution characterization,
and transformation recommendations.  All downstream modules (normalisation_strategy,
normality_analysis in eda.py, visualization) must consume the output of
analyze_column_properties() rather than running independent tests.

Tests applied per column
------------------------
1. Shapiro-Wilk        — best power for n ≤ 5 000; stratified sample above that
2. Anderson-Darling    — tail-sensitive; replaces KS2-sample against a simulated normal
3. Lilliefors          — KS test corrected for estimated µ/σ (Lilliefors 1967).
                         Raw KS with estimated parameters is anti-conservative and
                         MUST NOT be used here. Falls back to raw KS with warning
                         if statsmodels is not installed.

Normality score (0–1)
---------------------
Each test contributes 1/3 when it passes.
Skewness/kurtosis penalties fine-tune borderline cases (max –0.10).

Score bands
-----------
≥ 0.75  → "Normal"       — StandardScaler / parametric models safe
0.40–0.75 → "Borderline" — inspect Q-Q plot; light transform may help
< 0.40  → "Non-normal"   — transformation or robust scaler recommended

Public API
----------
analyze_column_properties(df, columns, alpha, sw_sample_size, verbose)
    → pd.DataFrame — one row per numeric column, all metrics + normality verdict
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import anderson, kurtosis, kstest, normaltest, shapiro, skew

logger = logging.getLogger("ml_framework.column_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN API
# ──────────────────────────────────────────────────────────────────────────────


def analyze_column_properties(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    sw_sample_size: int = 5_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Complete statistical analysis of each numeric column.

    Parameters
    ----------
    df             : source DataFrame
    columns        : columns to analyze; defaults to all numeric columns
    alpha          : significance threshold for normality tests (default 0.05)
    sw_sample_size : max sample for Shapiro-Wilk on large datasets (default 5 000)
    verbose        : print the analysis table and alerts

    Returns
    -------
    pd.DataFrame — one row per column with columns:
        column, colonne,
        n, n_unique, is_low_variance,
        mean, median, std, variance, min, max, range, q1, q3, iqr,
        skewness, kurtosis,
        cv, has_negative, has_zero,
        n_outliers_iqr, outliers_pct,
        shapiro_stat, shapiro_p,
        ad_stat, ad_critical_5pct, ad_reject_h0,
        ks_stat, ks_p,
        normality_score, normality_verdict,
        is_normal,
        linear_model_rec, tree_model_rec, transformation_rec,
        transform_recommendation
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        logger.warning("No numeric columns found for analysis.")
        return pd.DataFrame()

    results = []

    for col in columns:
        serie = df[col].dropna()

        if serie.empty:
            logger.warning("Column '%s' is empty after NaN removal — skipped.", col)
            continue

        n        = len(serie)
        n_unique = int(serie.nunique())
        min_val  = float(serie.min())
        max_val  = float(serie.max())
        col_range = max_val - min_val
        mean_val  = float(serie.mean())
        median_val = float(serie.median())
        std_val   = float(serie.std(ddof=1))
        var_val   = float(serie.var(ddof=1))

        is_low_variance = std_val < 1e-8 or n_unique <= 1

        sk = float(skew(serie, nan_policy="omit"))
        ku = float(kurtosis(serie, nan_policy="omit"))  # Fisher excess; normal → 0

        has_neg  = bool((serie < 0).any())
        has_zero = bool((serie == 0).any())

        # IQR outliers
        q1, q3   = float(serie.quantile(0.25)), float(serie.quantile(0.75))
        iqr_val  = q3 - q1
        n_outliers_iqr = int(((serie < q1 - 1.5 * iqr_val) | (serie > q3 + 1.5 * iqr_val)).sum())
        outliers_pct   = round(n_outliers_iqr / n * 100, 2)
        has_outliers   = n_outliers_iqr > 0

        # Coefficient of variation
        cv = round(std_val / abs(mean_val), 4) if abs(mean_val) > 1e-10 else float("inf")

        # ── 1. Shapiro-Wilk ───────────────────────────────────────────────────
        sw_stat = sw_p = np.nan
        if n >= 3:
            if n > sw_sample_size:
                try:
                    bins   = pd.qcut(serie, q=10, duplicates="drop", labels=False)
                    sample = (
                        serie.groupby(bins, observed=True)
                        .apply(lambda g: g.sample(min(len(g), sw_sample_size // 10), random_state=42))
                        .reset_index(drop=True)
                    )
                except Exception:
                    sample = serie.sample(sw_sample_size, random_state=42)
            else:
                sample = serie
            try:
                sw_stat, sw_p = shapiro(sample)
            except Exception:
                pass

        sw_pass = not np.isnan(sw_p) and float(sw_p) > alpha

        # ── 2. Anderson-Darling ───────────────────────────────────────────────
        ad_stat = ad_crit5 = np.nan
        ad_pass = False
        try:
            ad_result = anderson(serie.values, dist="norm")
            ad_stat   = float(ad_result.statistic)
            ad_crit5  = float(ad_result.critical_values[2])   # 5 % level
            ad_pass   = ad_stat < ad_crit5
        except Exception:
            pass

        # ── 3. Lilliefors test (KS corrected for estimated parameters) ──────────
        # Standardizing by empirical (µ̂, σ̂) then calling kstest("norm") is
        # anti-conservative: the null distribution assumes KNOWN parameters, so
        # the p-value is systematically inflated. Lilliefors (1967) provides the
        # correct critical values for the case where µ and σ are estimated.
        ks_stat = ks_p = np.nan
        ks_pass = False
        if std_val > 0:
            try:
                from statsmodels.stats.diagnostic import lilliefors as _lilliefors
                ks_stat, ks_p = _lilliefors(serie.values, dist="norm", pvalmethod="approx")
                ks_pass = float(ks_p) > alpha
            except Exception:
                # statsmodels unavailable — fall back to raw KS with prominent caveat
                try:
                    ks_stat, ks_p = kstest((serie.values - mean_val) / std_val, "norm")
                    ks_pass = float(ks_p) > alpha
                    logger.warning(
                        "Column '%s': using raw KS (anti-conservative) — "
                        "install statsmodels for Lilliefors correction.",
                        col,
                    )
                except Exception:
                    pass

        # ── Normality score (0–1) ─────────────────────────────────────────────
        base_score = (int(sw_pass) + int(ad_pass) + int(ks_pass)) / 3.0
        skew_pen   = max(0.0, (abs(sk) - 0.5) * 0.05)    # penalty if |skew| > 0.5
        kurt_pen   = max(0.0, (abs(ku) - 1.5) * 0.025)   # penalty if |kurt| > 1.5
        norm_score = round(max(0.0, min(1.0, base_score - skew_pen - kurt_pen)), 3)

        if norm_score >= 0.75:
            verdict = "Normal"
        elif norm_score >= 0.40:
            verdict = "Borderline"
        else:
            verdict = "Non-normal"

        # is_normal: backward-compat boolean (True ↔ verdict == "Normal")
        is_normal = verdict == "Normal"

        # ── ML-aware recommendations ──────────────────────────────────────────
        from ml_framework.config.config import NormalizationConfig as _NC
        _outlier_pct_threshold = _NC().outlier_threshold  # default 10.0 %
        if verdict == "Normal":
            linear_rec  = "StandardScaler"
            transfo_rec = "None — distribution is approximately normal"
        elif has_outliers and outliers_pct > _outlier_pct_threshold:
            linear_rec  = "RobustScaler"
            transfo_rec = "Winsorization or RobustScaler"
        elif abs(sk) > 1 and not has_neg and not has_zero:
            linear_rec  = "log1p → StandardScaler"
            transfo_rec = "log1p  (positive skew, no zero/negative)"
        elif abs(sk) > 1:
            linear_rec  = "Yeo-Johnson → StandardScaler"
            transfo_rec = "Yeo-Johnson  (handles negatives and zeros)"
        elif abs(sk) > 0.5:
            linear_rec  = "Yeo-Johnson or Box-Cox → StandardScaler"
            transfo_rec = "Box-Cox (positive only) or Yeo-Johnson"
        else:
            linear_rec  = "StandardScaler"
            transfo_rec = "MinMaxScaler (uniform-like, low skew)"

        tree_rec = "No scaling needed (tree models are distribution-invariant)"

        # transform_recommendation: backward-compat alias for transform_rec
        transform_recommendation = _recommend_transform(sk, ku, outliers_pct, cv, is_normal)

        results.append({
            "column":               col,
            "n":                    n,
            "n_unique":             n_unique,
            "is_low_variance":      is_low_variance,
            "mean":                 round(mean_val,   6),
            "median":               round(median_val, 6),
            "std":                  round(std_val,    6),
            "variance":             round(var_val,    6),
            "min":                  round(min_val,    4),
            "max":                  round(max_val,    4),
            "range":                round(col_range,  4),
            "q1":                   round(q1,         4),
            "q3":                   round(q3,         4),
            "iqr":                  round(iqr_val,    4),
            "skewness":             round(sk, 4),
            "kurtosis":             round(ku, 4),
            "cv":                   cv,
            "has_negative":         has_neg,
            "has_zero":             has_zero,
            "n_outliers_iqr":       n_outliers_iqr,
            "outliers_pct":         outliers_pct,
            # Shapiro-Wilk
            "shapiro_stat":         round(float(sw_stat), 4) if not np.isnan(sw_stat) else np.nan,
            "shapiro_p":            round(float(sw_p),    4) if not np.isnan(sw_p)    else np.nan,
            # Anderson-Darling
            "ad_stat":              round(ad_stat,  4) if not np.isnan(ad_stat)  else np.nan,
            "ad_critical_5pct":     round(ad_crit5, 4) if not np.isnan(ad_crit5) else np.nan,
            "ad_reject_h0":         not ad_pass,
            # KS
            "ks_stat":              round(float(ks_stat), 4) if not np.isnan(ks_stat) else np.nan,
            "ks_p":                 round(float(ks_p),    4) if not np.isnan(ks_p)    else np.nan,
            # Verdict
            "normality_score":      norm_score,
            "normality_verdict":    verdict,
            "is_normal":            is_normal,
            # Recommendations
            "linear_model_rec":     linear_rec,
            "tree_model_rec":       tree_rec,
            "transformation_rec":   transfo_rec,
            "transform_recommendation": transform_recommendation,
        })

    analysis_df = pd.DataFrame(results)

    if verbose and not analysis_df.empty:
        _print_analysis(analysis_df, alpha)

    return analysis_df


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _recommend_transform(sk: float, ku: float, outlier_pct: float, cv: float, is_normal: bool) -> str:
    """Backward-compat heuristic recommendation (string label)."""
    if is_normal and outlier_pct < 5:
        return "StandardScaler"
    if abs(sk) > 2 or outlier_pct > 10:
        return "RobustScaler + log/Yeo-Johnson"
    if abs(sk) > 1:
        return "log1p / sqrt (moderate asymmetry)"
    if cv > 2:
        return "RobustScaler (high variability)"
    if cv > 1:
        return "MinMaxScaler"
    if outlier_pct > 5:
        return "Winsorization + StandardScaler"
    return "StandardScaler"


def _print_analysis(df: pd.DataFrame, alpha: float = 0.05) -> None:
    W = 90
    print("\n" + "═" * W)
    print("  STATISTICAL ANALYSIS OF NUMERIC COLUMNS")
    print(f"  α = {alpha} | Tests: Shapiro-Wilk + Anderson-Darling + Lilliefors")
    print(f"  Score ≥ 0.75 → Normal | 0.40–0.75 → Borderline | < 0.40 → Non-normal")
    print("═" * W)

    icons = {"Normal": "✅", "Borderline": "🟡", "Non-normal": "❌"}
    display_cols = [c for c in [
        "column", "n", "mean", "std", "skewness", "kurtosis",
        "outliers_pct", "normality_score", "normality_verdict",
    ] if c in df.columns]

    # Add verdict icon
    df_display = df[display_cols].copy()
    df_display["verdict"] = df["normality_verdict"].map(icons) + " " + df["normality_verdict"]
    df_display = df_display.drop(columns=["normality_verdict"])

    print(df_display.to_string(index=False))
    print()

    # Alerts
    col_key = "column"
    non_normal = df[df["normality_verdict"] == "Non-normal"][col_key].tolist()
    borderline = df[df["normality_verdict"] == "Borderline"][col_key].tolist()
    high_out   = df[df["outliers_pct"] > 10][col_key].tolist()
    low_var    = df[df["is_low_variance"]][col_key].tolist()

    print("  ALERTS:")
    if non_normal:
        print(f"  ❌  Non-normal        : {non_normal}")
    if borderline:
        print(f"  🟡  Borderline        : {borderline}")
    if high_out:
        print(f"  ⚠️   Outliers > 10 %   : {high_out}")
    if low_var:
        print(f"  ⚠️   Near-zero variance: {low_var}")
    if not (non_normal or borderline or high_out or low_var):
        print("  ✅  No critical alerts detected.")
    print("═" * W + "\n")
