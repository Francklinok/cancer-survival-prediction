"""
visualization/analysis/drift_plots.py
— Data drift visualization: KDE/bar comparison of reference vs current.

Public functions:
  plot_drift_overview(drift_df, df_ref, df_cur, columns)
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.drift_plots")


def plot_drift_overview(
    drift_df: pd.DataFrame,
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    columns: List[str],
) -> None:
    """
    Overlay reference vs current distributions for drifted columns.

    Parameters
    ----------
    drift_df : result DataFrame from detect_data_drift()
               with columns ['colonne', 'drift_detected', 'psi', ...]
    df_ref   : reference dataset
    df_cur   : current dataset
    columns  : list of column names to plot
    """
    drifted = drift_df[drift_df["drift_detected"]]["colonne"].tolist()
    cols_to_plot = [c for c in columns if c in drifted] or columns[:6]

    if not cols_to_plot:
        return

    n_cols = min(3, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        is_numeric = pd.api.types.is_numeric_dtype(df_ref[col])

        if is_numeric:
            sns.kdeplot(df_ref[col].dropna(), ax=ax, label="Reference", fill=True, alpha=0.4, color="steelblue")
            sns.kdeplot(df_cur[col].dropna(), ax=ax, label="Current",   fill=True, alpha=0.4, color="salmon")
        else:
            ref_pct = df_ref[col].value_counts(normalize=True).rename("Reference")
            cur_pct = df_cur[col].value_counts(normalize=True).rename("Current")
            pd.concat([ref_pct, cur_pct], axis=1).plot(kind="bar", ax=ax, color=["steelblue", "salmon"])

        match = drift_df[drift_df["colonne"] == col]
        row   = match.iloc[0] if len(match) > 0 else {}
        psi_str = f" | PSI={row.get('psi', '-')}" if isinstance(row, dict) is False and "psi" in row else ""
        is_drifted = row.get("drift_detected", False) if hasattr(row, "get") else False
        ax.set_title(f"{col}{psi_str}", fontsize=10, fontweight="bold",
                     color="crimson" if is_drifted else "black")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution Comparison — Data Drift", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
