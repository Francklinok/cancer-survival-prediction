"""
ensemble.py — Ensemble methods.

Implemented methods:
  - Stacking  (StackingClassifier with configurable meta-learner)
  - Soft Voting (VotingClassifier)
  - Hard Voting
  - Blending (hold-out validation)

Features:
  - Automatic comparative evaluation: ensemble vs base models
  - Adaptive scoring (roc_auc binary / f1_macro multiclass)
  - Detailed performance report
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml_framework.modeling.model_trainer import get_scoring_strategy
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.ensemble")


# =============================================================================
# STACKING
# =============================================================================


def create_stacking_model(
    base_models: List[Tuple[str, object]],
    meta_model: Optional[object] = None,
    cv: int = 5,
    passthrough: bool = True,
    n_jobs: int = -1,
    verbose: bool = True,
) -> StackingClassifier:
    """
    Create a stacking ensemble model.

    Parameters
    ----------
    base_models : list of (name, estimator) tuples
    meta_model  : meta-learner (LogisticRegression by default)
    cv          : cross-validation folds for generating meta-features
    passthrough : pass original features to the meta-model
    n_jobs      : parallelization

    Returns
    -------
    StackingClassifier ready to fit
    """
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=1_000, random_state=42, C=1.0)

    if verbose:
        section_header("STACKING MODEL")
        print(f"  Meta-model  : {meta_model.__class__.__name__}")
        print("  Base models :")
        for name, model in base_models:
            print(f"    - {name} : {model.__class__.__name__}")

    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        passthrough=passthrough,
        n_jobs=n_jobs,
    )

    return ensemble


# =============================================================================
# VOTING
# =============================================================================


def create_voting_model(
    base_models: List[Tuple[str, object]],
    voting: str = "soft",
    weights: Optional[List[float]] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> VotingClassifier:
    """
    Create a voting ensemble model.

    Parameters
    ----------
    voting   : 'soft' (weighted probabilities) | 'hard' (majority vote)
    weights  : per-model weights (None = equal weights)

    Returns
    -------
    VotingClassifier
    """
    if verbose:
        section_header(f"VOTING MODEL ({voting.upper()})")
        for name, model in base_models:
            print(f"  - {name} : {model.__class__.__name__}")

    return VotingClassifier(
        estimators=base_models,
        voting=voting,
        weights=weights,
        n_jobs=n_jobs,
    )


# =============================================================================
# COMPARATIVE EVALUATION
# =============================================================================


def evaluate_ensemble_vs_base(
    ensemble,
    base_models: List[Tuple[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compare ensemble performance against each base model.

    Parameters
    ----------
    scoring : metric name for cross_val_score. If None, auto-selected via
              get_scoring_strategy (roc_auc for binary, f1_macro for multiclass).

    Returns
    -------
    pd.DataFrame — comparative results sorted by CV_mean descending
    """
    n_classes, auto_scoring = get_scoring_strategy(y_train)
    scoring = scoring or auto_scoring

    cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rows = []

    all_models = base_models + [("ensemble", ensemble)]

    section_header("ENSEMBLE vs BASE MODELS COMPARISON")
    print(f"  Scoring: {scoring}  |  Classes: {n_classes}  |  CV folds: {cv_folds}")
    print()

    for name, model in all_models:
        t0 = time.time()
        try:
            cv_sc = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
            model.fit(X_train, y_train)

            try:
                if hasattr(model, "predict_proba") and n_classes == 2:
                    test_sc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                elif hasattr(model, "predict_proba") and n_classes > 2:
                    test_sc = f1_score(y_test, model.predict(X_test), average="macro")
                else:
                    test_sc = accuracy_score(y_test, model.predict(X_test))
            except Exception:
                test_sc = float(cv_sc.mean())

            elapsed = time.time() - t0
            rows.append({
                "model":      name,
                "CV_mean":    round(float(cv_sc.mean()), 4),
                "CV_std":     round(float(cv_sc.std()), 4),
                "test_score": round(float(test_sc), 4),
                "time_s":     round(elapsed, 2),
            })
            print(
                f"  {name:<30} CV={cv_sc.mean():.4f}±{cv_sc.std():.4f}"
                f"  Test={test_sc:.4f}  ({elapsed:.1f}s)"
            )

        except Exception as exc:
            logger.error("Evaluation error for '%s': %s", name, exc)

    result_df = pd.DataFrame(rows).sort_values("CV_mean", ascending=False)
    if not result_df.empty:
        best = result_df.iloc[0]
        print(f"\n  Best: {best['model']} — CV={best['CV_mean']:.4f}  Test={best['test_score']:.4f}")

    return result_df


# =============================================================================
# BLENDING (HOLD-OUT)
# =============================================================================


def create_blending_model(
    base_models: List[Tuple[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    blend_size: float = 0.20,
    meta_model: Optional[object] = None,
    random_state: int = 42,
) -> Tuple[object, pd.DataFrame]:
    """
    Blending: train base models on (1 - blend_size),
    generate predictions on (blend_size), train meta-model on those predictions.

    Returns
    -------
    (meta_model_fitted, blend_predictions_df)
    """
    from sklearn.model_selection import train_test_split

    X_base, X_blend, y_base, y_blend = train_test_split(
        X_train, y_train, test_size=blend_size, random_state=random_state, stratify=y_train
    )

    meta_features = {}

    for name, model in base_models:
        model.fit(X_base, y_base)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_blend)[:, 1]
        else:
            preds = model.predict(X_blend).astype(float)
        meta_features[name] = preds

    blend_df = pd.DataFrame(meta_features)

    if meta_model is None:
        meta_model = LogisticRegression(max_iter=1_000, random_state=random_state)

    meta_model.fit(blend_df, y_blend)
    logger.info("Blending complete — meta-model: %s", meta_model.__class__.__name__)

    return meta_model, blend_df


def create_ensemble_model(
    base_models: List[Tuple[str, object]],
    meta_model: Optional[object] = None,
    cv: int = 5,
    use_features: bool = True,
    verbose: bool = True,
) -> StackingClassifier:
    """
    DEPRECATED — use create_stacking_model() directly.

    The parameter ``use_features`` maps to ``passthrough`` in create_stacking_model.
    """
    import warnings
    warnings.warn(
        "create_ensemble_model() is deprecated. "
        "Call create_stacking_model(passthrough=use_features) directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_stacking_model(
        base_models=base_models,
        meta_model=meta_model,
        cv=cv,
        passthrough=use_features,
        verbose=verbose,
    )
