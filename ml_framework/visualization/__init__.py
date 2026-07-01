"""
visualization/ — Centralized visualization layer for ml_framework.

All plotting code lives here, separated from business/ML logic.
Domain modules import from this package; they never call plt directly.

Sub-packages:
  analysis/        — EDA, distributions, correlations, drift, class balance
  evaluation/      — Confusion matrix, ROC, PR, calibration, overfitting
  features/        — Feature selection, dimensionality reduction
  interpretability/ — SHAP, LIME, PDP, clinical profiles
  monitoring/      — Performance history, drift alerts
"""

from ml_framework.visualization.base import configure_plot_style, section_header, save_or_show

__all__ = ["configure_plot_style", "section_header", "save_or_show"]
