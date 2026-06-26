"""
metrics.py — Centralized metrics utilities.

Functions:
  - compute_roc_auc              : ROC-AUC for binary and multiclass problems
  - compute_base_metrics         : Full classification metrics block
  - bootstrap_confidence_interval: Bootstrap confidence interval for any metric
  - format_metrics_table         : Format metrics dict for display
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("ml_framework.metrics")


# =============================================================================
# ROC-AUC (binary + multiclass)
# =============================================================================


def compute_roc_auc(y_true, y_proba, n_classes: int) -> float:
    """
    Compute ROC-AUC adapted to the number of classes.

    Parameters
    ----------
    y_true   : true labels
    y_proba  : probability vector (n_samples,) for binary,
               or matrix (n_samples, n_classes) for multiclass
    n_classes: number of target classes
    """
    try:
        if n_classes == 2:
            if hasattr(y_proba, "ndim") and y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return float(roc_auc_score(y_true, y_proba))
        else:
            # weighted average — accounts for class imbalance; macro ignores support
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except Exception as exc:
        logger.warning("ROC-AUC could not be computed: %s", exc)
        return float("nan")


# =============================================================================
# BASE METRICS
# =============================================================================


def compute_base_metrics(
    y_test,
    y_pred,
    y_proba=None,
    n_classes: int = 2,
    *,
    y_prob=None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute a complete set of classification metrics.

    Parameters
    ----------
    y_test    : true labels
    y_pred    : predicted labels
    y_proba   : probability scores (legacy positional parameter)
    n_classes : number of target classes (inferred from y_test when omitted)
    y_prob    : probability scores (preferred keyword alias for y_proba)
    average   : averaging strategy for precision/recall/f1 (default: 'weighted')

    Returns
    -------
    dict with: accuracy, precision, recall, f1, f1_macro, mcc, kappa,
               roc_auc, avg_precision, brier_score (where applicable)
    """
    proba = y_prob if y_prob is not None else y_proba

    if n_classes == 2:
        n_classes = len(np.unique(y_test))

    metrics: Dict[str, float] = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        "f1_macro":  float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_test, y_pred)),
        "kappa":     float(cohen_kappa_score(y_test, y_pred)),
    }

    if proba is not None:
        metrics["roc_auc"] = compute_roc_auc(y_test, proba, n_classes)

        if n_classes == 2:
            proba_1d = proba[:, 1] if (hasattr(proba, "ndim") and proba.ndim == 2) else proba
            try:
                metrics["avg_precision"] = float(average_precision_score(y_test, proba_1d))
                metrics["brier_score"] = float(brier_score_loss(y_test, proba_1d))
            except Exception:
                pass

    return metrics


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================


def bootstrap_confidence_interval(
    y_true,
    y_pred,
    metric_fn: Callable,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple:
    """
    Compute a bootstrap confidence interval for a given metric.

    Parameters
    ----------
    y_true      : true labels
    y_pred      : predicted labels
    metric_fn   : sklearn.metrics function or callable (y_true, y_pred) → float
    n_bootstraps: number of resampling iterations
    alpha       : significance level (0.05 → 95% CI)
    random_state: random seed for reproducibility

    Returns
    -------
    (lower_bound, upper_bound) — CI at (1-alpha)*100%
    """
    rng = np.random.RandomState(random_state)
    scores = []
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(float(score))
        except Exception:
            pass

    if not scores:
        return (float("nan"), float("nan"))

    lower = float(np.percentile(scores, alpha / 2 * 100))
    upper = float(np.percentile(scores, (1 - alpha / 2) * 100))
    return lower, upper


# =============================================================================
# FORMATTING
# =============================================================================


def format_metrics_table(
    metrics: Dict[str, float],
    title: str = "Performance Metrics",
    decimal: int = 4,
) -> pd.DataFrame:
    """
    Format a metrics dict into a displayable DataFrame.

    Returns
    -------
    pd.DataFrame with columns: Metric, Value, Interpretation
    """
    interpretations = {
        "accuracy":    lambda v: "Excellent" if v >= 0.95 else "Very Good" if v >= 0.90 else "Good" if v >= 0.80 else "Moderate",
        "roc_auc":     lambda v: "Excellent" if v >= 0.95 else "Very Good" if v >= 0.90 else "Good" if v >= 0.80 else "Moderate" if v >= 0.70 else "Poor",
        "f1":          lambda v: "Excellent" if v >= 0.90 else "Very Good" if v >= 0.85 else "Good" if v >= 0.75 else "Moderate",
        "mcc":         lambda v: "Strong" if v >= 0.60 else "Moderate" if v >= 0.30 else "Weak",
        "brier_score": lambda v: "Good" if v <= 0.10 else "Moderate" if v <= 0.25 else "Poor (insufficient calibration)",
    }

    rows = []
    for metric, value in metrics.items():
        interp = ""
        if metric in interpretations and not np.isnan(value):
            try:
                interp = interpretations[metric](value)
            except Exception:
                pass
        rows.append({
            "Metric": metric,
            "Value": round(value, decimal),
            "Interpretation": interp,
        })

    df = pd.DataFrame(rows)

    print(f"\n  {title}")
    print("  " + "─" * 58)
    print(df.to_string(index=False))

    return df
