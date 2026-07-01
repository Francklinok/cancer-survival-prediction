"""
model_trainer.py — Model training, selection, and hyperparameter optimization pipeline.

Features:
  - Multi-model training with stratified cross-validation
  - Best model selection by CV score
  - GridSearchCV / RandomizedSearchCV
  - Nested Cross-Validation for unbiased performance estimation
  - Automatic binary / multiclass handling
  - Detailed logging and comparison report
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from ml_framework.modeling.model_registry import get_models
from ml_framework.modeling.hyperparameter_grids import get_param_grid
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.model_trainer")

# =============================================================================
# ADAPTIVE SCORING
# =============================================================================

def get_scoring_strategy(y, problem_type: str = "binary") -> tuple:
    """
    Delegate to the canonical implementation in strategies/scoring_strategy.py.
    Single source of truth for scoring metric selection.

    Returns (n_classes, scoring_name) for backward compatibility with existing
    call sites in this module that unpack exactly two values.
    """
    from ml_framework.strategies.scoring_strategy import get_scoring_strategy as _get
    n_classes, scoring, _ = _get(y, problem_type=problem_type)
    return n_classes, scoring


# =============================================================================
# MULTI-MODEL TRAINING
# =============================================================================

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_to_test: Optional[List[str]] = None,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict:
    """
    Train multiple models, evaluate by stratified CV, and return results.

    Parameters
    ----------
    X              : features (numeric, no NaN)
    y              : target
    models_to_test : model prefixes ['rf', 'gb', 'lr', 'svm', ...]
    test_size      : test split proportion
    cv_folds       : cross-validation folds
    random_state   : random seed
    n_jobs         : parallel workers
    verbose        : detailed output

    Returns
    -------
    dict with: models, evaluation_results, best_model,
               X_train, X_test, y_train, y_test
    """
    n_classes, scoring = get_scoring_strategy(y)

    # Stratified split BEFORE any imputation to prevent data leakage
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    # Avoid Data Leakage: Impute AFTER the train/test split!
    # We must calculate the median values using the training set ONLY, 
    # and then use those same medians to fill missing values in the test set.
    # 
    # If we calculate medians on the whole dataset beforehand, the training 
    # model gets a "sneak peek" at the test set's distribution. This is a 
    # classic case of data leakage that would artificially boost our test scores.
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)  # use training medians on test — no leakage

    cv         = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    all_models = get_models(random_state=random_state, n_jobs=n_jobs, subset=models_to_test)

    state = {
        "models":             {},
        "evaluation_results": {},
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "best_model": None,
    }

    best_score      = 0.0
    best_model_name: Optional[str] = None
    comparison_rows = []

    section_header("MODEL TRAINING")
    print(f"  Scoring: {scoring}  |  CV: {cv_folds} folds  |  Models: {len(all_models)}")
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print()

    for model_name, model in all_models.items():
        t0 = time.time()
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            # Test score
            try:
                if hasattr(model, "predict_proba") and n_classes == 2:
                    y_proba    = model.predict_proba(X_test)[:, 1]
                    test_score = roc_auc_score(y_test, y_proba)
                else:
                    test_score = accuracy_score(y_test, model.predict(X_test))
            except Exception:
                test_score = cv_scores.mean()

            result = {
                "cv_scores":  cv_scores,
                "cv_mean":    float(cv_scores.mean()),
                "cv_std":     float(cv_scores.std()),
                "test_score": float(test_score),
                "elapsed_s":  round(elapsed, 2),
            }

            state["models"][model_name]             = model
            state["evaluation_results"][model_name] = result

            if cv_scores.mean() > best_score:
                best_score      = cv_scores.mean()
                best_model_name = model_name

            comparison_rows.append({
                "model":      model_name,
                "CV_mean":    round(cv_scores.mean(), 4),
                "CV_std":     round(cv_scores.std(), 4),
                "test_score": round(test_score, 4),
                "time_s":     round(elapsed, 2),
            })

            if verbose:
                print(
                    f"  {model_name:<35} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
                    f"  Test={test_score:.4f}  ({elapsed:.1f}s)"
                )

        except Exception as exc:
            logger.error("Training error for %s: %s", model_name, exc)

    state["best_model"] = best_model_name

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows).sort_values("CV_mean", ascending=False)
        section_header("MODEL COMPARISON")
        print(comp_df.to_string(index=False))
        print(f"\n  Best model: {best_model_name}  (CV={best_score:.4f})")

    return state


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================


def hyperparameter_tuning(
    state: Dict,
    search_type: str = "grid",
    n_iter: int = 50,
    cv_folds: int = 5,
    grid_level: str = "fast",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict:
    """
    Optimize the best model's hyperparameters.

    Parameters
    ----------
    state       : result of train_models()
    search_type : 'grid' | 'random' | 'bayesian'
    n_iter      : iterations for RandomizedSearch / Bayesian
    grid_level  : 'fast' | 'full'
    random_state: random seed
    n_jobs      : parallel workers
    verbose     : print best parameters

    Returns
    -------
    state dict updated with the optimized model
    """
    best_model_name = state.get("best_model")
    if not best_model_name:
        logger.warning("No best model available — tuning skipped.")
        return state

    model        = state["models"][best_model_name]
    model_prefix = best_model_name.split("_")[0]
    X_train      = state["X_train"]
    y_train      = state["y_train"]
    _, scoring   = get_scoring_strategy(y_train)

    try:
        param_grid = get_param_grid(model_prefix, level=grid_level)
    except ValueError:
        logger.warning("No grid available for '%s' — tuning skipped.", model_prefix)
        return state

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    section_header(f"HYPERPARAMETER OPTIMIZATION — {best_model_name}")
    print(f"  Method: {search_type}  |  Scoring: {scoring}  |  Grid: {grid_level}")
    t0 = time.time()

    if search_type == "grid":
        searcher = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=n_jobs, refit=True, verbose=0,
        )
    elif search_type == "random":
        searcher = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
            n_jobs=n_jobs, refit=True, random_state=random_state, verbose=0,
        )
    elif search_type == "bayesian":
        from ml_framework.optimization.bayesian_optimization import (
            optimize_hyperparameters_bayesian,
        )
        best_model, _ = optimize_hyperparameters_bayesian(
            X_train, y_train, model, param_space={},
            cv=cv_folds, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs,
        )
        opt_name = f"{best_model_name}_optimized"
        state["models"][opt_name]             = best_model
        state["best_model"]                   = opt_name
        return state
    else:
        raise ValueError(
            f"Invalid search_type '{search_type}'. Choose: 'grid', 'random', 'bayesian'"
        )

    searcher.fit(X_train, y_train)
    elapsed   = time.time() - t0
    opt_name  = f"{best_model_name}_optimized"
    best_est  = searcher.best_estimator_

    # Evaluate optimized model
    X_test   = state["X_test"]
    y_test   = state["y_test"]
    n_cls, _ = get_scoring_strategy(y_test)

    try:
        if hasattr(best_est, "predict_proba") and n_cls == 2:
            test_score = roc_auc_score(y_test, best_est.predict_proba(X_test)[:, 1])
        else:
            test_score = accuracy_score(y_test, best_est.predict(X_test))
    except Exception:
        test_score = searcher.best_score_

    state["models"][opt_name] = best_est
    state["evaluation_results"][opt_name] = {
        "cv_mean":    searcher.best_score_,
        "cv_std":     0.0,
        "test_score": float(test_score),
        "best_params": searcher.best_params_,
        "elapsed_s":  round(elapsed, 2),
    }
    state["best_model"] = opt_name

    print(f"\n Optimized score: {searcher.best_score_:.4f}  (Test={test_score:.4f})  [{elapsed:.1f}s]")
    if verbose:
        print(f"  Best params: {searcher.best_params_}")

    return state


# =============================================================================
# NESTED CROSS-VALIDATION
# =============================================================================


def nested_cv_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    model_prefix: str,
    outer_folds: int = 5,
    inner_folds: int = 3,
    grid_level: str = "fast",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """
    Nested Cross-Validation for unbiased performance estimation.

    Returns
    -------
    dict with: outer_scores, mean, std, ci_95, elapsed_s
    """
    X = X.fillna(X.median(numeric_only=True))
    _, scoring = get_scoring_strategy(y)

    try:
        param_grid = get_param_grid(model_prefix, level=grid_level)
    except ValueError:
        logger.warning("No grid available for Nested CV '%s'.", model_prefix)
        return {}

    inner_cv     = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    outer_cv     = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    outer_scores = []

    section_header(f"NESTED CROSS-VALIDATION — {model_prefix}")
    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        gs = GridSearchCV(model, param_grid, cv=inner_cv, scoring=scoring, n_jobs=n_jobs)
        gs.fit(X_tr, y_tr)

        n_cls, _ = get_scoring_strategy(y_te)
        try:
            if hasattr(gs.best_estimator_, "predict_proba") and n_cls == 2:
                score = roc_auc_score(y_te, gs.best_estimator_.predict_proba(X_te)[:, 1])
            else:
                score = accuracy_score(y_te, gs.best_estimator_.predict(X_te))
        except Exception:
            score = gs.best_score_

        outer_scores.append(float(score))
        print(f"  Fold {fold}/{outer_folds}: {score:.4f}")

    elapsed  = time.time() - t0
    mean_sc  = np.mean(outer_scores)
    std_sc   = np.std(outer_scores)

    print(f"\n  Nested CV Score : {mean_sc:.4f} ± {std_sc:.4f}  [{elapsed:.1f}s]")
    print(f"  95% CI          : [{mean_sc - 2*std_sc:.4f}, {mean_sc + 2*std_sc:.4f}]")

    return {
        "outer_scores": outer_scores,
        "mean":         round(float(mean_sc), 4),
        "std":          round(float(std_sc), 4),
        "ci_95":        (round(mean_sc - 2 * std_sc, 4), round(mean_sc + 2 * std_sc, 4)),
        "elapsed_s":    round(elapsed, 2),
    }
