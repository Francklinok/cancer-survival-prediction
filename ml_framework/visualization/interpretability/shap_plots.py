"""
visualization/interpretability/shap_plots.py
— SHAP-based model explanation plots.

Public functions:
  plot_shap_summary(shap_matrix, X_test, feature_names, max_display, plot_type, shap_values)
  plot_shap_dependence(shap_matrix, X_test, feature_names, top_n)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.visualization.shap_plots")


def plot_shap_summary(
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]],
    max_display: int = 20,
    plot_type: str = "summary",
    shap_values=None,
) -> None:
    """
    SHAP summary, bar, or beeswarm plot.

    Parameters
    ----------
    shap_matrix   : 2D array of SHAP values (n_samples × n_features)
    X_test        : test features DataFrame
    feature_names : list of feature names
    max_display   : number of features to display
    plot_type     : 'summary' | 'bar' | 'beeswarm'
    shap_values   : original SHAP Explanation object (required for beeswarm)
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed: pip install shap")
        return

    plot_types = ["summary", "bar", "beeswarm"] if plot_type == "all" else [plot_type]

    for pt in plot_types:
        if pt == "summary":
            shap.summary_plot(
                shap_matrix, X_test,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )
            plt.title("Feature Importance and Impact (SHAP Summary)", fontweight="bold")
            plt.tight_layout()
            plt.show()

        elif pt == "bar":
            shap.summary_plot(
                shap_matrix, X_test,
                feature_names=feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False,
            )
            plt.title("Feature Importance (SHAP Mean |shap|)", fontweight="bold")
            plt.tight_layout()
            plt.show()

        elif pt == "beeswarm" and shap_values is not None and hasattr(shap_values, "values"):
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
            plt.title("SHAP Value Distribution (Beeswarm)", fontweight="bold")
            plt.tight_layout()
            plt.show()


def plot_shap_dependence(
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 3,
) -> None:
    """
    Dependence plots for the top-N most important features.

    Parameters
    ----------
    shap_matrix   : 2D array of SHAP values
    X_test        : test features DataFrame
    feature_names : list of feature names
    top_n         : number of dependence plots to generate
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed: pip install shap")
        return

    mean_abs   = np.abs(shap_matrix).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-top_n:][::-1]

    for idx in top_indices:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        try:
            shap.dependence_plot(
                idx, shap_matrix, X_test,
                feature_names=feature_names,
                show=False,
            )
            plt.title(f"Dependence Plot — {feat_name}", fontweight="bold")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            logger.warning("Dependence plot for '%s' failed: %s", feat_name, exc)
