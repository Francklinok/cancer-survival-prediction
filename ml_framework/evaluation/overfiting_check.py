"""
overfitting_check.py — Overfitting and underfitting analysis.

Features:
  - Train vs test multi-metric comparison
  - Learning curves (bias/variance decomposition)
  - Validation curves (hyperparameter vs score)
  - Model complexity analysis
  - Automatic recommendations

Visualizations delegated to visualization.evaluation.overfitting_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, validation_curve

from ml_framework.visualization.evaluation.overfitting_plots import (
    plot_learning_curves,
    plot_validation_curve,
)

logger = logging.getLogger("ml_framework.overfitting")


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN VS TEST COMPARISON
# ──────────────────────────────────────────────────────────────────────────────


def check_overfitting(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    threshold: float = 0.10,
    *,
    test_acc: float = None,
) -> Dict:
    """
    Detect overfitting by comparing train vs test accuracy.

    Parameters
    ----------
    model     : trained model
    X_train   : training features
    y_train   : training target
    X_test    : test features (used to compute test_acc when provided)
    y_test    : test target   (used to compute test_acc when provided)
    threshold : gap threshold to trigger warnings
    test_acc  : pre-computed test accuracy (legacy keyword; ignored when
                X_test/y_test are supplied)

    Returns
    -------
    dict with: train_accuracy, test_accuracy, overfitting_gap,
               overfitting_severity
    """
    y_train_pred = model.predict(X_train)
    train_acc    = float(accuracy_score(y_train, y_train_pred))

    # Compute test accuracy from data if available; fall back to pre-computed value
    if X_test is not None and y_test is not None:
        test_acc = float(accuracy_score(y_test, model.predict(X_test)))
    elif test_acc is None:
        raise ValueError("Provide either (X_test, y_test) or the test_acc keyword argument.")

    gap          = train_acc - test_acc

    if gap > threshold * 1.5:
        severity = "high"
    elif gap > threshold:
        severity = "moderate"
    elif gap < -threshold:
        severity = "underfitting"
    else:
        severity = "none"

    print(f"\n  Train Accuracy  : {train_acc:.4f}")
    print(f"  Test Accuracy   : {test_acc:.4f}")
    print(f"  Gap             : {gap:.4f}")

    if severity == "high":
        print(f" Strong overfitting (gap={gap:.3f} > {threshold*1.5:.2f})")
        print("     Recommendations: increase regularization, reduce features, get more data")
    elif severity == "moderate":
        print(f" Mild overfitting tendency (gap={gap:.3f})")
    elif severity == "underfitting":
        print(f" Underfitting detected (test > train by {abs(gap):.3f})")
        print("     Recommendations: use a more complex model, add more features")
    else:
        print(f" Good generalization (gap={gap:.3f} ≤ {threshold:.2f})")

    return {
        "train_accuracy":       round(train_acc, 4),
        "test_accuracy":        round(test_acc, 4),
        "overfitting_gap":      round(gap, 4),
        "overfitting_severity": severity,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LEARNING CURVES
# ──────────────────────────────────────────────────────────────────────────────


def compute_learning_curves(
    model,
    X,
    y,
    cv: int = 5,
    scoring: str = "roc_auc",
    train_sizes: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> Dict:
    """
    Compute and plot learning curves for bias/variance analysis.

    Parameters
    ----------
    model       : sklearn-compatible estimator (must be cloneable)
    X           : features
    y           : target
    cv          : cross-validation folds
    scoring     : sklearn scoring string
    train_sizes : fraction array (default: 10 steps from 0.1 to 1.0)
    n_jobs      : parallel workers (-1 = all CPUs)

    Returns
    -------
    dict with: train_sizes, train_scores_mean, val_scores_mean,
               final_gap, final_val_score
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.10, 1.0, 10)

    print(f"\n  Computing learning curves ({scoring})...")

    try:
        train_sz, train_sc, val_sc = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            shuffle=True,
            random_state=42,
        )
    except Exception as exc:
        logger.error("Learning curve failed: %s", exc)
        return {}

    train_mean = np.mean(train_sc, axis=1)
    train_std  = np.std(train_sc,  axis=1)
    val_mean   = np.mean(val_sc,   axis=1)
    val_std    = np.std(val_sc,    axis=1)

    plot_learning_curves(train_sz, train_mean, train_std, val_mean, val_std, scoring)

    final_gap = float(train_mean[-1] - val_mean[-1])
    final_val = float(val_mean[-1])

    print(f"\n  Final validation score : {final_val:.4f}")
    print(f"  Final train-val gap    : {final_gap:.4f}")

    if final_gap > 0.10:
        print("  → Large train/val gap: HIGH VARIANCE (overfitting) — add regularization")
    elif final_val < 0.70:
        print("  → Low validation score: HIGH BIAS (underfitting) — try a more complex model")
    else:
        print("  → Good bias/variance balance")

    return {
        "train_sizes":        train_sz.tolist(),
        "train_scores_mean":  train_mean.tolist(),
        "val_scores_mean":    val_mean.tolist(),
        "final_gap":          round(final_gap, 4),
        "final_val_score":    round(final_val, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION CURVES
# ──────────────────────────────────────────────────────────────────────────────


def compute_validation_curve(
    model,
    X,
    y,
    param_name: str,
    param_range,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> None:
    """
    Compute and plot the validation curve for a given hyperparameter.

    Identifies the optimal complexity point where validation performance peaks.

    Parameters
    ----------
    model       : sklearn-compatible estimator
    X           : features
    y           : target
    param_name  : hyperparameter name (e.g. 'max_depth')
    param_range : values to evaluate
    cv          : cross-validation folds
    scoring     : evaluation metric
    n_jobs      : parallel workers
    """
    try:
        train_sc, val_sc = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        logger.error("Validation curve failed: %s", exc)
        return

    train_mean = np.mean(train_sc, axis=1)
    train_std  = np.std(train_sc,  axis=1)
    val_mean   = np.mean(val_sc,   axis=1)
    val_std    = np.std(val_sc,    axis=1)

    best_val_idx = int(np.argmax(val_mean))

    plot_validation_curve(
        param_range, train_mean, train_std, val_mean, val_std,
        param_name, scoring, best_val_idx,
    )

    print(f"\n  Optimal value of {param_name} : {param_range[best_val_idx]}")
    print(f"  Optimal validation score      : {val_mean[best_val_idx]:.4f}")
