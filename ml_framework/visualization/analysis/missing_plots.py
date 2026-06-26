"""
visualization/analysis/missing_plots.py
— Visualizations for missing value analysis.

Public functions:
  plot_missing_overview(df, missing_df)
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.missing_plots")


def plot_missing_overview(df: pd.DataFrame, missing_df: pd.DataFrame) -> None:
    """
    Bar chart of missing value rates + heatmap of NaN patterns.

    Parameters
    ----------
    df         : original DataFrame
    missing_df : DataFrame with columns ['missing_count', 'missing_pct']
                 (index = column names)
    """
    if missing_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    missing_df["missing_pct"].plot.barh(ax=axes[0], color="salmon", edgecolor="white")
    axes[0].set_xlabel("% Missing Values")
    axes[0].set_title("NaN Rate by Column", fontweight="bold")
    axes[0].axvline(5,  color="orange", linestyle="--", label="5%")
    axes[0].axvline(20, color="red",    linestyle="--", label="20%")
    axes[0].legend()

    sample = df[missing_df.index].isnull().sample(min(500, len(df)), random_state=42)
    sns.heatmap(sample.T, cbar=False, cmap="Blues", ax=axes[1], yticklabels=True)
    axes[1].set_title("NaN Pattern (500-row sample)", fontweight="bold")
    axes[1].set_xlabel("Observations")

    plt.suptitle("Missing Value Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
