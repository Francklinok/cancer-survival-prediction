"""
visualization/analysis/class_balance_plots.py
— Class imbalance visualization: effectif bar + proportion pie.

Public functions:
  plot_class_distribution(counts, pct, severity, ratio)
  plot_rebalance_comparison(y_before, y_after, strategy)
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.class_balance_plots")


def plot_class_distribution(
    counts: pd.Series,
    pct: pd.Series,
    severity: str,
    ratio: float,
) -> None:
    """
    Bar chart of class counts + pie chart of proportions.

    Parameters
    ----------
    counts   : value_counts() Series
    pct      : percentage Series (counts / total * 100)
    severity : imbalance severity label (for title)
    ratio    : majority/minority ratio
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = sns.color_palette("Set2", len(counts))

    axes[0].bar(counts.index.astype(str), counts.values, color=palette, edgecolor="white")
    axes[0].set_title(f"Class Counts  (ratio={ratio:.1f}:1)", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for i, (idx, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + counts.max() * 0.01, str(cnt), ha="center", fontsize=9)

    axes[1].pie(
        pct.values, labels=pct.index.astype(str),
        autopct="%1.1f%%", colors=palette,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        startangle=90,
    )
    axes[1].set_title(f"Proportions  [{severity.upper()}]", fontweight="bold")

    severity_color = {
        "balanced": "green", "mild": "yellowgreen",
        "moderate": "orange", "severe": "orangered", "extreme": "crimson",
        # French labels kept for backward compatibility
        "équilibré": "green", "léger": "yellowgreen",
        "modéré": "orange", "sévère": "orangered", "extrême": "crimson",
    }
    fig.patch.set_edgecolor(severity_color.get(severity, "grey"))
    fig.patch.set_linewidth(3)

    plt.suptitle("Class Imbalance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_rebalance_comparison(
    y_before: pd.Series,
    y_after: pd.Series,
    strategy: str,
) -> None:
    """
    Side-by-side bar charts comparing class distribution before and after rebalancing.

    Parameters
    ----------
    y_before : original target series
    y_after  : rebalanced target series
    strategy : name of the rebalancing strategy applied
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    all_classes = sorted(set(y_before.unique()) | set(y_after.unique()), key=str)
    palette_map = dict(zip(all_classes, sns.color_palette("Set2", len(all_classes))))

    for ax, y_d, title in zip(axes, [y_before, y_after], ["Before", "After"]):
        vc = y_d.value_counts()
        bar_colors = [palette_map[c] for c in vc.index]
        ax.bar(vc.index.astype(str), vc.values, color=bar_colors, edgecolor="white")
        ax.set_title(f"{title} Rebalancing", fontweight="bold")
        ax.set_ylabel("Count")
        for i, (idx, cnt) in enumerate(vc.items()):
            ax.text(i, cnt + vc.max() * 0.01, str(cnt), ha="center", fontsize=9)

    plt.suptitle(f"Rebalancing Impact ({strategy.upper()})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
