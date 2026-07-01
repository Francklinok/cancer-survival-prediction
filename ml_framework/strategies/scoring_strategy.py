"""
scoring_strategy.py — Adaptive scoring strategy for ML models.

Automatically determines the correct scikit-learn scorer based on:
  - Number of classes (binary vs multi-class)
  - Class distribution (imbalance)
  - Task type (binary/multi-class classification, regression)

Public functions:
  - get_scoring_strategy(y, task="classification") → (n_classes, scoring_name, scorer_kwargs)
  - list_available_scorers()                        → list[str]
  - make_scorer_from_config(config)                 → dict
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.scoring_strategy")


# =============================================================================
# AVAILABLE SCORERS
# =============================================================================

_BINARY_SCORERS = [
    "roc_auc",
    "average_precision",
    "f1",
    "f1_weighted",
    "balanced_accuracy",
    "precision",
    "recall",
    "accuracy",
]

_MULTICLASS_SCORERS = [
    "roc_auc_ovr",
    "roc_auc_ovo",
    "f1_weighted",
    "f1_macro",
    "balanced_accuracy",
    "accuracy",
]

_REGRESSION_SCORERS = [
    "r2",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
    "neg_mean_absolute_percentage_error",
]


def list_available_scorers(task: str = "classification") -> list:
    """
    Return the list of available scorers for the given task type.

    Parameters
    ----------
    task : 'binary', 'multiclass', 'classification', 'regression'

    Returns
    -------
    list[str]
    """
    task = task.lower()
    if task == "regression":
        return _REGRESSION_SCORERS
    elif task == "multiclass":
        return _MULTICLASS_SCORERS
    else:
        return _BINARY_SCORERS + _MULTICLASS_SCORERS


# =============================================================================
# IMBALANCE DETECTION
# =============================================================================


def _imbalance_ratio(y: pd.Series) -> float:
    """Return the minority/majority ratio (0–1). 1 = perfectly balanced."""
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return 1.0
    return float(counts.min() / counts.max())


# =============================================================================
# MAIN STRATEGY
# =============================================================================


def get_scoring_strategy(
    y,
    task: str = "classification",
    prefer_proba: bool = True,
    *,
    problem_type: str = None,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    Return the scoring strategy adapted to the target y.

    Decision logic:
      - Regression                              : r2
      - Binary balanced + probabilities         : roc_auc
      - Binary imbalanced + probabilities       : average_precision (PR-AUC)
      - Multi-class (≥ 3 classes)               : roc_auc_ovr (weighted)
      - Multi-class highly imbalanced           : f1_weighted

    Parameters
    ----------
    y            : pd.Series or array of the target variable
    task         : 'classification' | 'regression'
    prefer_proba : if True, prefer probability-based scorers
    problem_type : alias for task accepted as a keyword argument
                   ('binary' and 'multiclass' map to 'classification';
                   'regression' maps to 'regression')

    Returns
    -------
    (n_classes, scoring_name, scorer_kwargs)
      - n_classes     : int (number of distinct classes)
      - scoring_name  : str (sklearn-compatible scoring name)
      - scorer_kwargs : dict (extra kwargs for cross_val_score if needed)
    """
    # Map problem_type keyword to the task parameter used internally
    if problem_type is not None:
        _pt = problem_type.lower()
        if _pt == "regression":
            task = "regression"
        else:
            task = "classification"

    y_s = pd.Series(y)
    n_classes = y_s.nunique()

    if task.lower() == "regression":
        logger.info("Scoring: regression → r2")
        return n_classes, "r2", {}

    imb = _imbalance_ratio(y_s)
    heavily_imbalanced = imb < 0.2  # minority < 20% of majority class

    if n_classes == 2:
        if prefer_proba:
            if heavily_imbalanced:
                scoring = "average_precision"
                note = f"binary imbalanced (ratio={imb:.2f}) → average_precision (PR-AUC)"
            else:
                scoring = "roc_auc"
                note = f"binary balanced (ratio={imb:.2f}) → roc_auc"
        else:
            scoring = "f1_weighted" if heavily_imbalanced else "balanced_accuracy"
            note = f"binary no-proba → {scoring}"

        logger.info("Scoring: %s", note)
        return n_classes, scoring, {}

    else:  # multi-class
        if prefer_proba:
            scoring = "roc_auc_ovr"
        else:
            scoring = "f1_weighted" if heavily_imbalanced else "balanced_accuracy"

        note = (
            f"multiclass ({n_classes} classes, ratio={imb:.2f}, "
            f"imbalanced={heavily_imbalanced}) → {scoring}"
        )
        logger.info("Scoring: %s", note)

        # roc_auc_ovr requires needs_proba=True in sklearn
        scorer_kwargs: Dict[str, Any] = {}
        if "roc_auc" in scoring:
            scorer_kwargs["multi_class"] = "ovr"

        return n_classes, scoring, scorer_kwargs


# =============================================================================
# UTILITY FROM CONFIG
# =============================================================================


def make_scorer_from_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a multi-scorer mapping from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict with optional keys:
        - 'task'         : 'classification' | 'regression'
        - 'extra_scorers': list[str] of additional scorers

    Returns
    -------
    dict {scorer_name: scorer_name_str}
    """
    task = config_dict.get("task", "classification")
    base_scorers = (
        ["r2", "neg_mean_absolute_error"]
        if task == "regression"
        else ["roc_auc", "f1_weighted", "balanced_accuracy", "average_precision"]
    )
    extra = config_dict.get("extra_scorers", [])
    all_scorers = list(dict.fromkeys(base_scorers + extra))  # deduplicate, preserve order
    return {s: s for s in all_scorers}
