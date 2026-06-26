"""
visualization/evaluation/fairness_plots.py
— Fairness audit visualization: group metric bar charts.

Public functions:
  plot_fairness_metrics(group_metrics, attr)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.fairness_plots")


def plot_fairness_metrics(group_metrics: Dict, attr: str) -> None:
    """
    Bar charts of fairness metrics per sensitive group.

    Parameters
    ----------
    group_metrics : dict {group_value: {tpr, ppv, selection_rate, tnr, ...}}
    attr          : name of the sensitive attribute (for title)
    """
    groups = list(group_metrics.keys())
    metrics_list = ["selection_rate", "tpr", "ppv", "tnr"]

    fig, axes = plt.subplots(1, len(metrics_list), figsize=(16, 4))
    palette = sns.color_palette("Set2", len(groups))

    for ax, metric in zip(axes, metrics_list):
        vals = [group_metrics[g][metric] for g in groups]
        ax.bar([str(g) for g in groups], vals, color=palette, edgecolor="white")
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_ylim([0, 1])
        ax.axhline(float(np.mean(vals)), color="red", linestyle="--", lw=1.2, label="Mean")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    plt.suptitle(f"Fairness Metrics by Group — {attr}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
