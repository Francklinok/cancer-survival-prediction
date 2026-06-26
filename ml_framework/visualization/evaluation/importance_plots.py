"""
visualization/evaluation/importance_plots.py
— Post-training feature importance visualization.

Public functions:
  plot_feature_importance_bar(top_df, method_name, top_n)
  plot_cumulative_importance(importance_df)
  plot_permutation_importance(top_df, scoring, top_n)
  plot_importance_methods_comparison(rank_df)
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.evaluation.importance_plots")


def plot_feature_importance_bar(
    top_df: pd.DataFrame,
    method_name: str,
    top_n: int,
) -> None:
    """
    Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    top_df      : DataFrame with columns ['Feature', 'Importance', 'Importance_pct']
    method_name : string description of the importance method
    top_n       : N shown in title
    """
    plt.figure(figsize=(10, max(5, top_n * 0.4 + 1)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top_df)))
    bars = plt.barh(top_df["Feature"][::-1], top_df["Importance"][::-1],
                    color=colors[::-1], edgecolor="white")
    for bar, pct in zip(bars, top_df["Importance_pct"][::-1]):
        plt.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=8)
    plt.xlabel(f"Importance ({method_name})")
    plt.title(f"Top {top_n} Most Important Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_cumulative_importance(importance_df: pd.DataFrame) -> None:
    """
    Line chart of cumulative feature importance percentage.

    Parameters
    ----------
    importance_df : DataFrame with column 'Importance_cumsum_pct'
    """
    plt.figure(figsize=(9, 5))
    x = range(1, len(importance_df) + 1)
    plt.plot(x, importance_df["Importance_cumsum_pct"], color="steelblue", lw=2, marker=".")
    plt.axhline(80, color="orange", linestyle="--", lw=1.5, label="80%")
    plt.axhline(95, color="red",    linestyle="--", lw=1.5, label="95%")
    n_80 = (importance_df["Importance_cumsum_pct"] <= 80).sum() + 1
    plt.axvline(n_80, color="orange", linestyle=":", lw=1)
    plt.xlabel("Number of features")
    plt.ylabel("Cumulative importance (%)")
    plt.title("Cumulative Importance Curve", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(f"\n  {n_80} features account for 80% of total importance.")


def plot_permutation_importance(
    top_df: pd.DataFrame,
    scoring: str,
    top_n: int,
) -> None:
    """
    Horizontal bar chart with error bars for permutation importance.

    Parameters
    ----------
    top_df  : DataFrame with columns ['Feature', 'Importance_mean', 'Importance_std']
    scoring : metric name used for permutation
    top_n   : N shown in title
    """
    plt.figure(figsize=(10, max(5, top_n * 0.4 + 1)))
    plt.barh(
        top_df["Feature"][::-1], top_df["Importance_mean"][::-1],
        xerr=top_df["Importance_std"][::-1],
        color="steelblue", alpha=0.8, edgecolor="white",
        error_kw={"ecolor": "black", "capsize": 3},
    )
    plt.xlabel(f"{scoring} decrease after permutation")
    plt.title(f"Permutation Importance — Top {top_n}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_importance_methods_comparison(rank_df: pd.DataFrame) -> None:
    """
    Heatmap comparing feature ranks across importance methods.

    Parameters
    ----------
    rank_df : DataFrame (features × methods) with rank values
    """
    if rank_df.empty:
        return

    plt.figure(figsize=(10, max(5, len(rank_df) * 0.4 + 1)))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                linewidths=0.5, cbar_kws={"label": "Rank (1 = most important)"})
    plt.title("Rank Comparison — Importance Methods", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
