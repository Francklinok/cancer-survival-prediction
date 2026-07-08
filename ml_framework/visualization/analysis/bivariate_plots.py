"""
visualization/analysis/bivariate_plots.py
— Bivariate analysis plots (scatter, boxplot-by-class, crosstab heatmaps).

Public functions:
  plot_numeric_vs_numeric(df, num_cols, target_col)
  plot_numeric_vs_categorical(df, num_cols, target_col)
  plot_categorical_vs_categorical(df, cat_cols, target_col, assoc_df)
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger("ml_framework.visualization.bivariate_plots")


def plot_numeric_vs_numeric(
    df: pd.DataFrame,
    num_cols: List[str],
    target_col: str,
) -> None:
    """Scatter plots + Pearson regression line for numeric features vs numeric target."""
    if not num_cols:
        return

    n_c = 3
    n_r = (len(num_cols) + n_c - 1) // n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 5, n_r * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax   = axes[i]
        data = df[[col, target_col]].dropna()
        r, p = stats.pearsonr(data[col], data[target_col])
        n = len(data)
        alpha = max(0.05, min(0.3, 300.0 / n)) if n > 0 else 0.3
        sns.regplot(
            x=col, y=target_col, data=data, ax=ax,
            scatter_kws={"alpha": alpha, "s": 20, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1.5},
        )
        sig = "✓" if p < 0.05 else "✗"
        ax.set_title(f"{col}  r={r:.3f} {sig} (p={p:.3f})", fontsize=9, fontweight="bold")

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Scatter + Pearson r — vs {target_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_numeric_vs_categorical(
    df: pd.DataFrame,
    num_cols: List[str],
    target_col: str,
) -> None:
    """Boxplots of numeric features grouped by categorical target, with median annotations."""
    if not num_cols:
        return

    n_c = 3
    n_r = (len(num_cols) + n_c - 1) // n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 5, n_r * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.boxplot(x=target_col, y=col, data=df, ax=ax, palette="Set2")
        medians = df.groupby(target_col)[col].median()
        for tick, label in enumerate(ax.get_xticklabels()):
            cat = label.get_text()
            if cat in medians.index:
                ax.text(
                    tick, medians[cat],
                    f"{medians[cat]:.1f}",
                    ha="center", va="bottom", fontsize=8, color="black",
                    fontweight="bold",
                )
        ax.set_title(f"{col} × {target_col}", fontsize=9)
        ax.tick_params(axis="x", rotation=25)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Boxplots — Numerics by class {target_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_categorical(
    df: pd.DataFrame,
    top_cats: List[str],
    target_col: str,
    assoc_df: pd.DataFrame,
) -> None:
    """Crosstab heatmaps for top categorical features vs categorical target."""
    if not top_cats:
        return

    n_c = 3
    n_r = (len(top_cats) + n_c - 1) // n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 5, n_r * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(top_cats):
        ct_norm = pd.crosstab(df[col], df[target_col], normalize="index")
        sns.heatmap(
            ct_norm, annot=True, fmt=".2f", cmap="Blues",
            ax=axes[i], linewidths=0.5,
            cbar_kws={"shrink": 0.7},
        )
        if "feature" in assoc_df.columns:
            cv_vals = assoc_df.loc[assoc_df["feature"] == col, "cramers_v"].values
        else:
            cv_vals = assoc_df.loc[assoc_df.index == col, "cramers_v"].values
        cv_val = cv_vals[0] if len(cv_vals) > 0 else 0.0
        axes[i].set_title(
            f"{col} × {target_col}\nCramér's V={cv_val:.3f}",
            fontsize=9, fontweight="bold",
        )

    for j in range(len(top_cats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Crosstab Heatmaps — Categoricals × {target_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
