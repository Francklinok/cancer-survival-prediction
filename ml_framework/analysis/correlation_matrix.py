"""
correlation_matrix.py — Multi-method correlation analysis with statistical tests.

Features:
  - Correlation matrix (Pearson, Spearman, Kendall)
  - Significance tests for each pair (p-values)
  - Identification of highly correlated pairs with recommendations
  - Annotated heatmap with triangular mask
  - Multicollinearity analysis (VIF)
  - Automatic scientific interpretation

All plotting is delegated to visualization.analysis.correlation_plots.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from ml_framework.visualization.analysis.correlation_plots import (
    plot_correlation_heatmap,
    plot_correlation_methods_comparison,
    plot_vif_chart,
)
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.correlation")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CORRELATION MATRIX
# ──────────────────────────────────────────────────────────────────────────────


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.70,
    figsize: tuple = (14, 12),
    annot_limit: int = 25,
    show_pvalues: bool = False,
) -> pd.DataFrame:
    """
    Compute and visualize the correlation matrix.

    Parameters
    ----------
    df           : DataFrame (non-numeric columns are ignored)
    method       : 'pearson' | 'spearman' | 'kendall'
    threshold    : threshold for identifying highly correlated pairs
    figsize      : figure size
    annot_limit  : annotate heatmap cells if n_cols ≤ annot_limit
    show_pvalues : if True, annotate each cell with ``r\\np=...`` instead of
                   just the coefficient.  Stars are added for significance:
                   *** p<0.001 | ** p<0.01 | * p<0.05

    Returns
    -------
    pd.DataFrame — highly correlated pairs (|corr| ≥ threshold)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found.")

    corr_matrix = numeric_df.corr(method=method)

    # Build p-value matrix when requested
    pval_matrix: Optional[pd.DataFrame] = None
    if show_pvalues:
        pval_matrix = _compute_pvalue_matrix(numeric_df, method)

    plot_correlation_heatmap(
        corr_matrix, method, threshold, figsize, annot_limit,
        pval_matrix=pval_matrix,
    )

    corr_pairs = _extract_high_corr_pairs(corr_matrix, threshold, numeric_df, method)

    if not corr_pairs.empty:
        section_header(f"HIGHLY CORRELATED PAIRS (|r| ≥ {threshold})")
        print(corr_pairs.to_string(index=False))
        _interpret_correlations(corr_pairs, threshold)
    else:
        print(f"\n No pairs with |r| ≥ {threshold}.")

    return corr_pairs


def _compute_pvalue_matrix(
    numeric_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Compute a symmetric matrix of pairwise p-values for every column pair.

    Diagonal is set to NaN (self-correlation has no meaningful p-value).
    """
    cols   = numeric_df.columns.tolist()
    n      = len(cols)
    pvals  = np.full((n, n), np.nan)

    stat_fn = {
        "pearson":  pearsonr,
        "spearman": spearmanr,
        "kendall":  kendalltau,
    }.get(method, pearsonr)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                x = numeric_df.iloc[:, i].dropna()
                y = numeric_df.iloc[:, j].dropna()
                idx = x.index.intersection(y.index)
                _, p = stat_fn(x.loc[idx].values, y.loc[idx].values)
                pvals[i, j] = p
                pvals[j, i] = p
            except Exception:
                pass

    return pd.DataFrame(pvals, index=cols, columns=cols)


def _extract_high_corr_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float,
    numeric_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Extract highly correlated pairs with p-values."""
    cols = corr_matrix.columns.tolist()
    rows = []

    for c1, c2 in combinations(cols, 2):
        val = corr_matrix.loc[c1, c2]
        if abs(val) < threshold:
            continue

        try:
            x = numeric_df[c1].dropna()
            y = numeric_df[c2].dropna()
            idx = x.index.intersection(y.index)
            x, y = x.loc[idx].values, y.loc[idx].values

            if method == "pearson":
                _, pval = pearsonr(x, y)
            elif method == "spearman":
                _, pval = spearmanr(x, y)
            elif method == "kendall":
                _, pval = kendalltau(x, y)
            else:
                pval = np.nan
        except Exception:
            pval = np.nan

        rows.append({
            "feature1":    c1,
            "feature2":    c2,
            "correlation": round(float(val), 4),
            "abs_corr":    round(abs(float(val)), 4),
            "p_value":     round(float(pval), 6) if not np.isnan(pval) else np.nan,
            "significant": bool(pval < 0.05)     if not np.isnan(pval) else None,
        })

    if not rows:
        return pd.DataFrame(columns=["feature1", "feature2", "correlation", "abs_corr", "p_value", "significant"])

    return (
        pd.DataFrame(rows)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-METHOD COMPARISON
# ──────────────────────────────────────────────────────────────────────────────


def compare_correlation_methods(
    df: pd.DataFrame,
    threshold: float = 0.70,
) -> dict:
    """
    Compare Pearson, Spearman, and Kendall on the same DataFrame.

    Returns
    -------
    dict[str, pd.DataFrame] — one key per method, value = highly correlated pairs.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    methods = ["pearson", "spearman", "kendall"]
    results = {
        method: _extract_high_corr_pairs(
            numeric_df.corr(method=method), threshold, numeric_df, method
        )
        for method in methods
    }
    plot_correlation_methods_comparison(numeric_df, results, methods)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# VIF ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────


def compute_vif(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) to detect multicollinearity.

    VIF > 5  → moderate multicollinearity
    VIF > 10 → high multicollinearity — candidate for removal

    Returns
    -------
    pd.DataFrame with columns: feature, VIF
    """
    from sklearn.linear_model import LinearRegression

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if columns:
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]

    if numeric_df.shape[1] < 2:
        logger.warning("VIF requires at least 2 numeric columns.")
        return pd.DataFrame()

    vif_data = []
    cols = numeric_df.columns.tolist()

    for i, col in enumerate(cols):
        X = numeric_df.drop(columns=[col]).values
        y = numeric_df[col].values

        try:
            r_sq = LinearRegression().fit(X, y).score(X, y)
            vif = 1 / (1 - r_sq) if r_sq < 1.0 else float("inf")
        except Exception:
            vif = float("nan")

        vif_data.append({"feature": col, "VIF": round(float(vif), 4)})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    plot_vif_chart(vif_df)

    high_vif = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
    mod_vif  = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]["feature"].tolist()
    if high_vif:
        print(f"\n  ⚠️  High multicollinearity (VIF>10)    : {high_vif}")
        print("      → Consider removal or PCA/regularization.")
    if mod_vif:
        print(f"  ⚠️  Moderate multicollinearity (VIF 5-10): {mod_vif}")

    return vif_df


# ──────────────────────────────────────────────────────────────────────────────
# AUTOMATIC INTERPRETATION
# ──────────────────────────────────────────────────────────────────────────────


def _interpret_correlations(corr_pairs: pd.DataFrame, threshold: float) -> None:
    """Generate automatic statistical/clinical interpretations."""
    section_header("INTERPRETATIONS")
    for _, row in corr_pairs.iterrows():
        r = row["correlation"]
        strength = (
            "very strong" if abs(r) >= 0.9
            else "strong"    if abs(r) >= 0.7
            else "moderate"
        )
        direction = "positive" if r > 0 else "negative"
        sig_txt   = "significant (p<0.05)" if row.get("significant") else "not tested"

        print(
            f"  • {row['feature1']} ↔ {row['feature2']}: "
            f"r={r:.3f} — {strength} {direction} correlation, {sig_txt}"
        )
        if abs(r) >= 0.9:
            print(
                f"    ↳ Near-perfect redundancy — consider removing one of the two features."
            )
