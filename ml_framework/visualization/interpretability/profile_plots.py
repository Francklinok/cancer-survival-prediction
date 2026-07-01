"""
visualization/interpretability/profile_plots.py
— Risk profile visualization: bar chart of scores + z-score heatmap.

Public functions:
  plot_risk_profiles(risk_profiles, sorted_features, population_stats)
  plot_profiles_heatmap(df, top_n)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.profile_plots")


def plot_risk_profiles(
    risk_profiles: Dict[str, Dict],
    sorted_features: List[Tuple[str, float]],
    population_stats: Dict[str, Dict[str, float]],
) -> None:
    """
    Two-panel chart: risk score bar + z-score heatmap for top features.

    Parameters
    ----------
    risk_profiles    : dict {level_name: {'features': {...}, 'risk_score': float}}
    sorted_features  : list of (feature_name, importance) sorted descending
    population_stats : dict {'mean': {feat: val}, 'std': {feat: val}}
    """
    try:
        sorted_levels = sorted(risk_profiles.keys(),
                               key=lambda x: risk_profiles[x]["risk_score"])
        colors = sns.color_palette("YlOrRd", len(sorted_levels))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Patient Risk Profiles", fontsize=14, fontweight="bold")

        scores = [risk_profiles[lv]["risk_score"] for lv in sorted_levels]
        bars   = axes[0].bar(sorted_levels, scores, color=colors, edgecolor="black")
        for bar, score in zip(bars, scores):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.2f}", ha="center", va="bottom", fontweight="bold",
            )
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Risk Score")
        axes[0].set_title("Score by Profile")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(axis="y", alpha=0.3)

        top_feats = [f for f, _ in sorted_features[:5]]
        z_matrix  = []
        for level in sorted_levels:
            profile = risk_profiles[level]["features"]
            z_row = []
            for feat in top_feats:
                val  = profile.get(feat, 0.0)
                mean = population_stats["mean"].get(feat, 0.0)
                std  = population_stats["std"].get(feat, 1.0) or 1.0
                z_row.append((val - mean) / std)
            z_matrix.append(z_row)

        z_df = pd.DataFrame(z_matrix, index=sorted_levels, columns=[f[:20] for f in top_feats])
        sns.heatmap(
            z_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, ax=axes[1], linewidths=0.5,
        )
        axes[1].set_title("Z-scores — Top 5 Features by Profile")
        axes[1].set_xlabel("Feature")

        plt.tight_layout()
        plt.show()

    except Exception as exc:
        logger.warning("Risk profile plot failed: %s", exc)


def plot_profiles_heatmap(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Standalone heatmap comparing z-scores across profiles.

    Parameters
    ----------
    df    : DataFrame (profiles × features) with z-score values
    top_n : maximum number of feature columns to show
    """
    if df.empty:
        return

    cols = df.columns[:top_n]
    plt.figure(figsize=(max(8, len(cols) * 0.8), max(4, len(df) * 0.8)))
    sns.heatmap(df[cols], annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, linewidths=0.5)
    plt.title(f"Profile Comparison — Z-scores (Top {top_n} features)")
    plt.tight_layout()
    plt.show()
