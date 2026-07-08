"""
visualization/features/selection_plots.py
— Feature selection visualization.

Public functions:
  plot_combined_selection(features, stat_norm, pca_norm, combined, sorted_features,
                          statistical_results, alpha_statistical, top_n)
  plot_robustness(robustness_results)
  plot_model_based_importance(top_df, model_name)
  plot_rfecv_curve(mean_scores, min_features, n_optimal, scoring)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.selection_plots")


def plot_combined_selection(
    features: List[str],
    stat_norm: Dict[str, float],
    pca_norm: Dict[str, float],
    combined: Dict[str, float],
    sorted_features: List[Tuple[str, float]],
    statistical_results: Dict[str, float],
    alpha_statistical: float,
    top_n: int,
) -> None:
    """
    Four-panel visualization of combined feature selection scores.
    """
    try:
        stat_vals  = [stat_norm.get(f, 0)  for f in features]
        pca_vals   = [pca_norm.get(f, 0)   for f in features]
        combo_vals = [combined.get(f, 0)   for f in features]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Combined Feature Selection Analysis", fontsize=15, fontweight="bold")

        bar_h = 0.38
        y_pos = np.arange(len(features))
        axes[0, 0].barh(y_pos + bar_h / 2, stat_vals, height=bar_h,
                        label="Statistical Score", color="steelblue")
        axes[0, 0].barh(y_pos - bar_h / 2, pca_vals, height=bar_h,
                        label="PCA Score", color="salmon")
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([f[:25] for f in features], fontsize=7)
        axes[0, 0].set_xlabel("Normalized Score")
        axes[0, 0].set_title("Score Comparison by Method")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        top_items = sorted_features[:top_n]
        axes[0, 1].barh(range(len(top_items)), [s[1] for s in top_items], color="mediumseagreen")
        axes[0, 1].set_yticks(range(len(top_items)))
        axes[0, 1].set_yticklabels([s[0][:25] for s in top_items])
        axes[0, 1].set_xlabel("Combined Score")
        axes[0, 1].set_title(f"Top {top_n} Features — Combined Score")
        axes[0, 1].grid(True, alpha=0.3)

        p_vals = [float(v) for v in statistical_results.values()]
        axes[1, 0].hist(p_vals, bins=20, color="gold", edgecolor="black", alpha=0.8)
        axes[1, 0].axvline(alpha_statistical, color="red", linestyle="--",
                           label=f"α = {alpha_statistical}")
        axes[1, 0].set_xlabel("P-values")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("P-value Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        score_df = pd.DataFrame({
            "Statistical": stat_vals,
            "PCA": pca_vals,
            "Combined": combo_vals,
        })
        sns.heatmap(score_df.corr(), annot=True, cmap="coolwarm", center=0,
                    square=True, ax=axes[1, 1])
        axes[1, 1].set_title("Correlation between Methods")

        plt.tight_layout()
        plt.show()
    except Exception as exc:
        logger.warning("Combined selection plot failed: %s", exc)


def plot_robustness(robustness_results: List[Dict[str, Any]]) -> None:
    """
    Line chart showing feature stability as statistical weight varies.
    """
    try:
        all_features: set = set()
        for r in robustness_results:
            all_features.update(r["top_5_features"])

        weights = [r["weight_stat"] for r in robustness_results]

        plt.figure(figsize=(12, 6))
       
        OUT_OF_TOP5 = 6
        for feature in all_features:
            positions = []
            for r in robustness_results:
                if feature in r["top_5_features"]:
                    positions.append(r["top_5_features"].index(feature) + 1)
                else:
                    positions.append(np.nan)

            if any(p <= 3 for p in positions if not np.isnan(p)):
                line, = plt.plot(weights, positions, marker="o", label=feature[:20], linewidth=2)
                dropped_w = [w for w, p in zip(weights, positions) if np.isnan(p)]
                if dropped_w:
                    plt.scatter(dropped_w, [OUT_OF_TOP5] * len(dropped_w),
                                marker="x", color=line.get_color(), alpha=0.5, zorder=3)

        plt.axhline(OUT_OF_TOP5 - 0.5, color="grey", linestyle=":", linewidth=1,
                    label="Out of top 5 (below this line)")
        plt.xlabel("Statistical Method Weight")
        plt.ylabel("Rank in Top 5")
        plt.title("Feature Selection Robustness by Weight")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        logger.warning("Robustness plot failed: %s", exc)


def plot_model_based_importance(top_df: pd.DataFrame, model_name: str) -> None:
    """
    Simple horizontal bar chart of model-based feature importances.
    """
    if top_df.empty:
        return
    plt.figure(figsize=(10, max(4, len(top_df) * 0.35)))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.show()


def plot_rfecv_curve(
    mean_scores: np.ndarray,
    min_features: int,
    n_optimal: int,
    scoring: str,
) -> None:
    """
    RFECV score curve: CV score vs number of features.
    """
    try:
        x = range(min_features, len(mean_scores) + min_features)
        plt.figure(figsize=(10, 4))
        plt.plot(x, mean_scores, marker="o")
        plt.axvline(n_optimal, color="red", linestyle="--", label=f"Optimal: {n_optimal}")
        plt.xlabel("Number of Features")
        plt.ylabel(f"CV Score ({scoring})")
        plt.title("RFECV — Score by Number of Features")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        logger.warning("RFECV plot failed: %s", exc)
