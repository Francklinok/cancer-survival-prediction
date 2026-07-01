"""
visualization/analysis/ — Plots for the analysis layer.

Modules:
  distributions  — numeric histograms, KDE, box/violin plots,
                   dtype pie chart, column-type breakdown pie
  categorical    — categorical bar/pie charts
  target         — target variable analysis plots
  bivariate      — scatter, boxplot-by-class, crosstab heatmaps
  missing        — missing-value heatmap and bar chart
  normality      — Q-Q plots, Shapiro-Wilk summary
  importance     — feature importance exploration charts
  insights       — business insights stacked-bar plots
  leakage        — data leakage risk bar chart
  correlation    — correlation heatmap, VIF bar chart
  drift          — data drift KDE / bar comparison
  class_balance  — class imbalance bar + pie charts
  diagnostic    — association bar + class histogram (diagnostic_analysis.py)
                  Cohen's d + significance bar (causal_analysis.py)

Quick imports:
  from ml_framework.visualization.analysis.distributions import (
      plot_dtypes_pie,
      plot_column_types_pie,
      plot_numeric_distributions,
      plot_categorical_distributions,
      plot_boxplots,
      plot_violins,
  )
"""

from ml_framework.visualization.analysis.distributions import (
    plot_dtypes_pie,
    plot_column_types_pie,
    plot_numeric_distributions,
    plot_categorical_distributions,
    plot_boxplots,
    plot_violins,
)
