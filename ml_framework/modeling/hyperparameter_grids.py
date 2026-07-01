"""
hyperparameter_grids.py — Hyperparameter grids for GridSearchCV / RandomizedSearchCV.

Two grid levels:
  - 'fast' : small grid for rapid exploration
  - 'full' : complete grid for final optimization

Models covered:
  rf, gb, lr, svm, knn, dt, et, ada, xgb, lgb
"""

from __future__ import annotations

from typing import Dict


# =============================================================================
# FAST GRIDS
# =============================================================================

_PARAM_GRIDS_FAST: Dict[str, Dict] = {
    "rf": {
        "n_estimators":     [50, 100],
        "max_depth":        [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt", "log2"],
    },
    "gb": {
        "n_estimators":  [50, 100],
        "learning_rate": [0.05, 0.10],
        "max_depth":     [3, 5],
        "subsample":     [0.8, 1.0],
    },
    "lr": {
        "C":        [0.01, 0.1, 1, 10],
        "penalty":  ["l2"],
        "solver":   ["lbfgs"],
        "max_iter": [1_000],
    },
    "svm": {
        "C":      [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
    "dt": {
        "max_depth":        [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion":        ["gini", "entropy"],
    },
    "et": {
        "n_estimators":     [50, 100],
        "max_depth":        [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "ada": {
        "n_estimators":  [50, 100],
        "learning_rate": [0.5, 1.0],
    },
    "xgb": {
        "n_estimators":     [50, 100],
        "learning_rate":    [0.05, 0.1],
        "max_depth":        [3, 5, 7],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "lgb": {
        "n_estimators":  [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5, -1],
        "num_leaves":    [31, 63],
        "subsample":     [0.8, 1.0],
    },
}


# =============================================================================
# FULL GRIDS
# =============================================================================

_PARAM_GRIDS_FULL: Dict[str, Dict] = {
    "rf": {
        "n_estimators":     [100, 200, 500],
        "max_depth":        [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2", 0.5],
        "bootstrap":        [True, False],
    },
    "gb": {
        "n_estimators":     [100, 200, 300],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "max_depth":        [3, 5, 7],
        "subsample":        [0.7, 0.8, 1.0],
        "min_samples_split": [2, 5],
    },
    "lr": {
        "C":        [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty":  ["l1", "l2", "elasticnet"],
        "solver":   ["saga"],
        "max_iter": [2_000],
        "l1_ratio": [0.3, 0.5, 0.7],
    },
    "svm": {
        "C":      [0.01, 0.1, 1, 10, 100],
        "kernel": ["rbf", "poly", "sigmoid"],
        "gamma":  ["scale", "auto", 0.001, 0.01],
        "degree": [2, 3],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan", "chebyshev"],
    },
    "dt": {
        "max_depth":        [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion":        ["gini", "entropy"],
        "max_features":     [None, "sqrt", "log2"],
    },
    "et": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "max_features":     ["sqrt", "log2"],
    },
    "ada": {
        "n_estimators":  [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0, 2.0],
        "algorithm":     ["SAMME.R", "SAMME"],
    },
    "xgb": {
        "n_estimators":     [100, 200, 300],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "max_depth":        [3, 5, 7, 9],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 1],
        "reg_lambda":       [1, 2, 5],
        "gamma":            [0, 0.1, 0.5],
    },
    "lgb": {
        "n_estimators":     [100, 200, 300],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [3, 5, 7, -1],
        "num_leaves":       [15, 31, 63, 127],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 1],
        "reg_lambda":       [0, 0.1, 1],
    },
}


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================


def get_param_grid(model_prefix: str, level: str = "fast") -> Dict:
    """
    Return the hyperparameter grid for a given model.

    Parameters
    ----------
    model_prefix : model prefix (e.g. 'rf', 'gb', 'lr', 'xgb', ...)
    level        : 'fast' | 'full'

    Returns
    -------
    dict — parameter grid
    """
    grids = _PARAM_GRIDS_FAST if level == "fast" else _PARAM_GRIDS_FULL

    if model_prefix not in grids:
        available = list(grids.keys())
        raise ValueError(
            f"Grid '{model_prefix}' not found. Available: {available}"
        )

    return grids[model_prefix]


def get_all_param_grids(level: str = "fast") -> Dict[str, Dict]:
    """Return all available parameter grids for the given level."""
    return _PARAM_GRIDS_FAST.copy() if level == "fast" else _PARAM_GRIDS_FULL.copy()
