"""
visualization/evaluation/clinical_plots.py
— Clinical model report visualizations: risk distribution, score histogram, ROC.

Public functions:
  plot_risk_visualizations(results, thresholds, y_test, y_proba, has_proba)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger("ml_framework.visualization.clinical_plots")


def plot_risk_visualizations(
    results: pd.DataFrame,
    thresholds: Dict[str, float],
    y_test,
    y_proba,
    has_proba: bool,
) -> None:
    """
    Three-panel clinical visualization: risk bar + score histogram + ROC curve.

    Parameters
    ----------
    results    : DataFrame with column 'Risk_Category'
    thresholds : dict {category: threshold_value}
    y_test     : true binary labels
    y_proba    : predicted probabilities
    has_proba  : whether the model produces probabilities
    """
    n_plots = 3 if has_proba else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    risk_order  = ["Very low", "Low", "Moderate", "High",
                   # French labels for backward compatibility
                   "Très faible", "Faible", "Modéré", "Élevé"]
    risk_colors_map = {
        "Very low": "#2ecc71", "Low": "#f1c40f", "Moderate": "#e67e22", "High": "#e74c3c",
        "Très faible": "#2ecc71", "Faible": "#f1c40f", "Modéré": "#e67e22", "Élevé": "#e74c3c",
    }

    vc   = results["Risk_Category"].value_counts()
    cats = [c for c in risk_order if c in vc.index]
   
    unmapped = [c for c in vc.index if c not in risk_order]
    if unmapped:
        logger.warning(
            "plot_risk_visualizations: unrecognized risk category label(s) %s "
            "not in the known risk_order — plotting them anyway with a "
            "default color.", unmapped,
        )
        cats = cats + sorted(unmapped, key=str)

    axes[0].bar(cats, [vc[c] for c in cats],
                color=[risk_colors_map.get(c, "#95a5a6") for c in cats],
                edgecolor="white")
    axes[0].set_title("Patients by Risk Level", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    for i, cat in enumerate(cats):
        axes[0].text(i, vc[cat] + 0.5, str(vc[cat]), ha="center", fontsize=10)

    if has_proba:
        axes[1].hist(y_proba, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        for cat, thresh in thresholds.items():
            axes[1].axvline(thresh, linestyle="--", lw=1.5, label=f"{cat} ({thresh})")
        axes[1].set_title("Risk Score Distribution", fontweight="bold")
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend(fontsize=8)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[2].plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC={auc:.3f}")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1)
        axes[2].set_xlabel("1 - Specificity")
        axes[2].set_ylabel("Sensitivity")
        axes[2].set_title("ROC Curve", fontweight="bold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.4)

    plt.suptitle("Clinical Report — Patient Risk Stratification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
