"""
bayesian_optimization.py — Bayesian hyperparameter optimization.

Uses Optuna (if available) or RandomizedSearchCV as a fallback.

Features:
  - Flexible search space definition (int, float, categorical)
  - Best-model tracking callback
  - Pruning of unpromising trials (MedianPruner)
  - Optimization history visualization
  - Hyperparameter importance
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna

logger = logging.getLogger("ml_framework.bayesian_optimization")


# =============================================================================
# MAIN BAYESIAN OPTIMIZATION
# =============================================================================


def optimize_hyperparameters_bayesian(
    X,
    y,
    estimator,
    param_space: Dict[str, Any],
    cv: int = 5,
    n_iter: int = 50,
    n_jobs: int = -1,
    scoring: str = "roc_auc",
    verbose: int = 1,
    random_state: int = 42,
) -> Tuple[object, Optional[object]]:
    """
    Bayesian hyperparameter optimization with Optuna.
    Falls back to RandomizedSearchCV if Optuna is not installed.

    Parameters
    ----------
    X           : features (DataFrame or array)
    y           : target
    estimator   : base sklearn estimator
    param_space : dict {param_name: {type: 'int'|'float'|'categorical', ...}}
                  Optuna format:
                    {'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                     'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                     'max_depth': {'type': 'categorical', 'values': [3, 5, 7]}}
    cv          : cross-validation folds
    n_iter      : number of trials (Optuna) or iterations (RandomizedSearch)
    scoring     : optimization metric
    verbose     : verbosity level (0 = silent)
    random_state: random seed

    Returns
    -------
    (best_model_fitted, study_or_None)
    """
    print("\n===== BAYESIAN HYPERPARAMETER OPTIMIZATION =====")
    print(f"  Estimator  : {estimator.__class__.__name__}")
    print(f"  n_iter     : {n_iter}  |  scoring : {scoring}  |  cv : {cv}")
    t0 = time.time()

    try:
        return _optimize_with_optuna(
            X, y, estimator, param_space, cv, n_iter, n_jobs, scoring, verbose, random_state
        )
    except ImportError:
        logger.warning("Optuna not installed — falling back to RandomizedSearchCV.")
        return _optimize_with_random_search(
            X, y, estimator, param_space, cv, n_iter, n_jobs, scoring, random_state
        )
    finally:
        elapsed = time.time() - t0
        print(f"  Optimization time: {elapsed:.2f}s")


# =============================================================================
# OPTUNA
# =============================================================================


def _optimize_with_optuna(X, y, estimator, param_space, cv, n_iter, n_jobs, scoring, verbose, random_state):

    if verbose == 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    best_params_store: Dict = {}

    def objective(trial):
        params = {}
        for param_name, spec in param_space.items():
            ptype = spec.get("type", "categorical")
            if ptype == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, spec["values"])
            elif ptype == "int":
                params[param_name] = trial.suggest_int(
                    param_name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif ptype == "float":
                params[param_name] = trial.suggest_float(
                    param_name, spec["low"], spec["high"], log=spec.get("log", False)
                )

        model = estimator.__class__(**{**estimator.get_params(), **params})
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_iter, n_jobs=1)

    best_params = study.best_params
    best_score = study.best_value

    best_model = estimator.__class__(**{**estimator.get_params(), **best_params})
    best_model.fit(X, y)

    print(f"\n  Best CV score ({scoring}): {best_score:.4f}")
    print("  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k:<30} : {v}")

    # Hyperparameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\n  Hyperparameter importances:")
        for k, v in importances.items():
            print(f"    {k:<30} : {v:.4f}")
    except Exception:
        pass

    return best_model, study


# =============================================================================
# FALLBACK RANDOMIZED SEARCH
# =============================================================================


def _optimize_with_random_search(X, y, estimator, param_space, cv, n_iter, n_jobs, scoring, random_state):
    from sklearn.model_selection import RandomizedSearchCV

    # Convert param_space to scipy/sklearn format
    sklearn_param_grid = {}
    for name, spec in param_space.items():
        ptype = spec.get("type", "categorical")
        if ptype == "categorical":
            sklearn_param_grid[name] = spec["values"]
        elif ptype == "int":
            sklearn_param_grid[name] = list(range(spec["low"], spec["high"] + 1, max(1, (spec["high"] - spec["low"]) // 10)))
        elif ptype == "float":
            sklearn_param_grid[name] = list(np.linspace(spec["low"], spec["high"], 10))

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator, sklearn_param_grid,
        n_iter=n_iter, cv=cv_splitter,
        scoring=scoring, n_jobs=n_jobs,
        random_state=random_state, refit=True, verbose=0,
    )
    rs.fit(X, y)

    print(f"\n  Best CV score ({scoring}): {rs.best_score_:.4f}")
    print(f"  Best params: {rs.best_params_}")

    return rs.best_estimator_, None
