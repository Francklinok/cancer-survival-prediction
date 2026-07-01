"""
visualization/monitoring/performance_plots.py
— Performance history visualization for MLOps monitoring.

Public functions:
  plot_performance_history(history_df, metrics)
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("ml_framework.visualization.performance_plots")


def plot_performance_history(history_df: pd.DataFrame, metrics: List[str]) -> None:
    """
    Line chart of model performance metrics over time (evaluations).

    Parameters
    ----------
    history_df : DataFrame with one row per evaluation,
                 timestamp/model_id columns + one column per metric
    metrics    : list of metric column names to plot
    """
    metric_cols = [m for m in metrics if m in history_df.columns]
    if not metric_cols:
        logger.warning("No metric columns found in history DataFrame.")
        return

    plt.figure(figsize=(12, 6))
    for metric in metric_cols:
        plt.plot(
            range(len(history_df)), history_df[metric],
            marker="o", lw=2, label=metric,
        )

    plt.xlabel("Evaluation #")
    plt.ylabel("Score")
    plt.title("Performance Trend Over Time", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
