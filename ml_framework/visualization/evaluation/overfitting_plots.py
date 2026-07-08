"""
visualization/evaluation/overfitting_plots.py
— Learning curves and validation curves for bias/variance analysis.

Public functions:
  plot_learning_curves(train_sz, train_mean, train_std, val_mean, val_std, scoring)
  plot_validation_curve(param_range, train_mean, train_std, val_mean, val_std, param_name, scoring, best_val_idx)
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("ml_framework.visualization.overfitting_plots")


def plot_learning_curves(
    train_sz: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    scoring: str = "roc_auc",
) -> None:
    """
    Learning curves showing training and validation score vs dataset size.

    Parameters
    ----------
    train_sz    : array of training set sizes
    train_mean  : mean training scores per size
    train_std   : std of training scores per size
    val_mean    : mean validation scores per size
    val_std     : std of validation scores per size
    scoring     : metric name for y-axis label
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_sz, train_mean, "o-", color="steelblue", label="Train", lw=2)
    plt.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="steelblue")
    plt.plot(train_sz, val_mean, "o-", color="salmon", label="Validation", lw=2)
    plt.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="salmon")

    plt.xlabel("Training set size")
    plt.ylabel(scoring)
    plt.title("Learning Curves — Bias/Variance Analysis", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_validation_curve(
    param_range: List,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    param_name: str,
    scoring: str = "roc_auc",
    best_val_idx: int = 0,
) -> None:
    """
    Validation curve for a single hyperparameter.

    Parameters
    ----------
    param_range   : list of hyperparameter values tested
    train_mean    : mean training scores per value
    train_std     : std of training scores
    val_mean      : mean validation scores
    val_std       : std of validation scores
    param_name    : hyperparameter name (x-axis label)
    scoring       : metric name (y-axis label)
    best_val_idx  : index of optimal value in param_range
    """
    plt.figure(figsize=(9, 5))
    plt.plot(param_range, train_mean, "o-", color="steelblue", label="Train", lw=2)
    plt.plot(param_range, val_mean,   "o-", color="salmon",    label="Validation", lw=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.12, color="steelblue")
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.12, color="salmon")
    plt.axvline(param_range[best_val_idx], color="green", linestyle="--",
                label=f"Optimal = {param_range[best_val_idx]}")

    numeric_range = [p for p in param_range if isinstance(p, (int, float)) and p > 0]
    if len(numeric_range) == len(param_range) and len(numeric_range) >= 3:
        ratios = [numeric_range[i + 1] / numeric_range[i] for i in range(len(numeric_range) - 1)]
        spans_orders_of_magnitude = max(numeric_range) / min(numeric_range) >= 100
        roughly_geometric = (max(ratios) / min(ratios)) < 3 if min(ratios) > 0 else False
        if spans_orders_of_magnitude and roughly_geometric:
            plt.xscale("log")

    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.title(f"Validation Curve — {param_name}", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
