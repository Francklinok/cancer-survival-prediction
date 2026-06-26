"""
visualization/analysis/correlation_plots.py
— Correlation matrix heatmaps and VIF bar charts.

Public functions:
  plot_correlation_heatmap(corr_matrix, method, threshold)
  plot_correlation_methods_comparison(numeric_df, results, methods)
  plot_vif_chart(vif_df)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.correlation_plots")


def _significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.70,
    figsize: tuple = (15, 12),
    annot_limit: int = 25,
    pval_matrix: Optional[pd.DataFrame] = None,
) -> None:
    """
    Single triangular heatmap — always annotated, font size adapts to matrix size.

    Annotation content per cell
    ---------------------------
    show_pvalues=False (pval_matrix=None)
        ``0.85``                           — coefficient only

    show_pvalues=True  (pval_matrix provided)
        ≤ 15 cols   →  ``0.85***``         — coefficient + stars, larger font
                       ``p=0.001``          — p-value on second line
        > 15 cols   →  ``0.85***``         — coefficient + stars only (single line)
                                              font scales down with n_cols

    Stars: *** p<0.001  ** p<0.01  * p<0.05

    Parameters
    ----------
    corr_matrix  : square correlation DataFrame
    method       : 'pearson' | 'spearman' | 'kendall' (for title)
    threshold    : threshold shown in title
    figsize      : base figure size (auto-scaled to keep cells readable)
    annot_limit  : legacy — kept for backwards compatibility, ignored
    pval_matrix  : p-value matrix returned by _compute_pvalue_matrix()
    """
    n     = len(corr_matrix)
    mask  = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap  = sns.diverging_palette(230, 20, as_cmap=True)

    font_size = max(5, min(10, int(120 / max(n, 1))))

    # ── Build annotation matrix (always string so fmt="" works uniformly) ──────
    annot_data = pd.DataFrame(
        index=corr_matrix.index, columns=corr_matrix.columns, dtype=object
    )
    two_lines = (pval_matrix is not None) and (n <= 15)

    for row_lbl in corr_matrix.index:
        for col_lbl in corr_matrix.columns:
            r = corr_matrix.loc[row_lbl, col_lbl]

            if pval_matrix is not None:
                p = pval_matrix.loc[row_lbl, col_lbl]
                stars = _significance_stars(p) if not np.isnan(p) else ""
                if two_lines and not np.isnan(p):
                    annot_data.loc[row_lbl, col_lbl] = f"{r:.2f}{stars}\np={p:.3f}"
                else:
                    annot_data.loc[row_lbl, col_lbl] = f"{r:.2f}{stars}"
            else:
                annot_data.loc[row_lbl, col_lbl] = f"{r:.2f}"

    # ── Auto-scale figure: target ~55–70 px per cell ──────────────────────────
    cell_in   = (0.85 if two_lines else 0.65)   # inches per cell
    side      = max(cell_in * n, figsize[0])
    side      = min(side, 28)
    fig_w, fig_h = side, side * 0.88

    # ── Draw single heatmap ───────────────────────────────────────────────────
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0, vmin=-1.0, center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.5, "label": f"{method.capitalize()} r"},
        annot=annot_data,
        fmt="",
        annot_kws={
            "size":        font_size,
            "va":          "center",
            "linespacing": 1.25 if two_lines else 1.0,
        },
    )

    title_suffix = "  (*** p<0.001  ** p<0.01  * p<0.05)" if pval_matrix is not None else ""
    ax.set_title(
        f"Correlation Matrix ({method.capitalize()}) — threshold {threshold}{title_suffix}",
        fontsize=12, fontweight="bold", pad=14,
    )
    lbl_size = max(5, min(9, 110 // n))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=lbl_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=lbl_size)
    plt.tight_layout()
    plt.show()


def plot_correlation_methods_comparison(
    numeric_df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    methods: List[str],
) -> None:
    """
    Side-by-side heatmaps comparing Pearson, Spearman, and Kendall.

    Parameters
    ----------
    numeric_df : numeric-only DataFrame
    results    : dict keyed by method name (unused — recomputes internally)
    methods    : list of method names to compare
    """
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones((len(numeric_df.columns),) * 2, dtype=bool))

    fig, axes = plt.subplots(1, len(methods), figsize=(22, 8))

    for ax, method in zip(axes, methods):
        corr = numeric_df.corr(method=method)
        sns.heatmap(
            corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=0.2, ax=ax,
            annot=len(corr) <= 15, fmt=".2f",
            annot_kws={"size": 6},
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(f"{method.capitalize()}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.suptitle("Correlation Method Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_vif_chart(vif_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of Variance Inflation Factors.

    Parameters
    ----------
    vif_df : DataFrame with columns ['feature', 'VIF']
    """
    if vif_df.empty:
        return

    colors = [
        "crimson" if v > 10 else "orange" if v > 5 else "steelblue"
        for v in vif_df["VIF"]
    ]
    plt.figure(figsize=(10, max(4, len(vif_df) * 0.35 + 1)))
    plt.barh(vif_df["feature"], vif_df["VIF"], color=colors, edgecolor="white")
    plt.axvline(5,  color="orange", linestyle="--", linewidth=1.5, label="VIF=5 (moderate)")
    plt.axvline(10, color="crimson", linestyle="--", linewidth=1.5, label="VIF=10 (high)")
    plt.xlabel("VIF")
    plt.title("Variance Inflation Factor — Multicollinearity", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()
