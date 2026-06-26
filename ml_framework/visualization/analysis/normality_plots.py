"""
visualization/analysis/normality_plots.py
— Q-Q plots for normality analysis.

Public functions:
  plot_qq_grid(df, columns, normality_df)
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("ml_framework.visualization.normality_plots")


def plot_qq_grid(
    df: pd.DataFrame,
    columns: List[str],
    normality_df: pd.DataFrame,
) -> None:
    """
    Grid of Q-Q plots for normality visual inspection.

    Title colour reflects the three-tier verdict from ``normality_analysis()``:
       green   — Normal       (score ≥ 0.75)
       orange  — Borderline   (0.40 ≤ score < 0.75)
       red     — Non-normal   (score < 0.40)

    Parameters
    ----------
    df           : source DataFrame
    columns      : list of numeric column names to plot
    normality_df : DataFrame indexed by variable name returned by
                   ``normality_analysis()``.  Supported columns:
                   ``normality_verdict`` (new) or ``is_normal`` (legacy bool).
    """
    if not columns:
        return

    # colour + icon mapping for 3-tier verdict
    _tier_style = {
        "Normal":     ("green",  "✅"),
        "Borderline": ("darkorange", "🟡"),
        "Non-normal": ("red",    "❌"),
        # legacy boolean fallback
        True:         ("green",  "✅"),
        False:        ("red",    "❌"),
    }

    n_c   = min(3, len(columns))
    n_r   = (len(columns) + n_c - 1) // n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 4, n_r * 3.8))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax   = axes[i]
        data = df[col].dropna()
        stats.probplot(data, dist="norm", plot=ax)

        if col in normality_df.index:
            if "normality_verdict" in normality_df.columns:
                verdict = normality_df.loc[col, "normality_verdict"]
                score   = normality_df.loc[col, "normality_score"] \
                          if "normality_score" in normality_df.columns else ""
                score_str = f"  score={score:.2f}" if score != "" else ""
            else:
                verdict = normality_df.loc[col, "is_normal"]   # legacy bool
                score_str = ""
        else:
            verdict, score_str = "Non-normal", ""

        color, icon = _tier_style.get(verdict, ("grey", "?"))
        ax.set_title(
            f"{col}  {icon}{score_str}",
            fontsize=9, fontweight="bold", color=color,
        )
        # Style the reference line
        for line in ax.get_lines():
            if line.get_linestyle() in ("--", "-"):
                line.set_color("steelblue")
                line.set_linewidth(1.2)

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Q-Q Plots — Normality Inspection\n"
        "Normal   Borderline   Non-normal",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
