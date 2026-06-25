"""
evaluation.py — Complete and advanced model evaluation.

Metrics computed:
  Binary      : Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC,
                Brier Score, Youden Index, optimal threshold
  Multiclass  : Accuracy, Macro/Weighted F1, OVR ROC-AUC, Cohen's Kappa
  Regression  : MSE, RMSE, MAE, MAPE, R², Adjusted R²

Visualizations delegated to visualization.evaluation.model_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_curve,
)

from ml_framework.visualization.evaluation.model_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_calibration,
    plot_score_distribution,
)
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.evaluation")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.50,
    verbose: bool = True,
) -> Dict:
    """
    Complete evaluation of a classification model.

    Parameters
    ----------
    model       : trained model (sklearn compatible)
    X_test      : test features
    y_test      : test target
    X_train     : training features (for overfitting detection)
    y_train     : training target
    class_names : class label names
    threshold   : classification threshold (binary)
    verbose     : detailed output

    Returns
    -------
    dict — complete performance metrics
    """
    section_header("ADVANCED MODEL EVALUATION")

    n_classes = len(np.unique(y_test))
    y_pred    = model.predict(X_test)

    y_proba      = None
    y_proba_full = None
    try:
        y_proba_full = model.predict_proba(X_test)
        y_proba = y_proba_full[:, 1] if n_classes == 2 else y_proba_full
    except AttributeError:
        pass

    metrics = compute_base_metrics(y_test, y_pred, y_prob=y_proba)

    if verbose:
        print("\n  Performance metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"    {name:<30} : {value:.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=class_names))

    if n_classes == 2 and y_proba is not None:
        opt_threshold = _optimal_threshold(y_test, y_proba)
        metrics["optimal_threshold"] = opt_threshold
        if verbose:
            print(f"\n  Optimal threshold (Youden J): {opt_threshold:.4f}")

    plot_confusion_matrix(y_test, y_pred, class_names)

    if n_classes == 2 and y_proba is not None:
        plot_roc_curve(y_test, y_proba, auc_score=metrics.get("roc_auc", 0))
        plot_precision_recall(y_test, y_proba)
        plot_calibration(model, X_test, y_test)
        plot_score_distribution(y_proba, y_test)

    if X_train is not None and y_train is not None:
        overfitting_info = check_overfitting(model, X_train, y_train, X_test, y_test)
        metrics.update(overfitting_info)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────


def compute_base_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average: str = "weighted",
) -> dict:
    """
    Delegate to the canonical implementation in utils/metrics.py.
    Single source of truth for base metrics computation.
    """
    from ml_framework.utils.metrics import compute_base_metrics as _compute
    return _compute(y_true, y_pred, y_prob=y_prob, average=average)


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMAL THRESHOLD
# ──────────────────────────────────────────────────────────────────────────────


def _optimal_threshold(y_test, y_proba) -> float:
    """Threshold that maximizes Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


# ──────────────────────────────────────────────────────────────────────────────
# OVERFITTING CHECK
# ──────────────────────────────────────────────────────────────────────────────


def check_overfitting(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.05,
) -> dict:
    """
    Delegate to the canonical implementation in evaluation/overfitting_check.py.
    This wrapper exists for backward compatibility only.
    """
    from ml_framework.evaluation.overfiting_check import check_overfitting as _check
    return _check(model, X_train, y_train, X_test, y_test, threshold=threshold)
