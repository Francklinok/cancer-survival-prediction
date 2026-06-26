"""
feature_selection.py — Combined feature selection (statistical + PCA).

Complements statistical_feature_selection.py with combined methods:
  - Weighted statistical + PCA (loadings) selection
  - Robustness analysis (weight sensitivity)
  - Model-based feature importance (Random Forest / Gradient Boosting)
  - Recursive elimination (RFE / RFECV)
  - Boruta support (if installed)

Public functions:
  - combined_feature_selection(data, target, statistical_results,
                                pca_components, ...) → dict
  - robustness_analysis(data, target, statistical_results, weight_range) → list
  - model_based_selection(X, y, model, top_n, plot)                      → tuple
  - recursive_feature_elimination(X, y, estimator, cv, scoring)          → tuple

Visualizations delegated to visualization.features.selection_plots.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance as perm_imp
from sklearn.model_selection import StratifiedKFold

from ml_framework.visualization.features.selection_plots import (
    plot_combined_selection,
    plot_robustness,
    plot_model_based_importance,
    plot_rfecv_curve,
)

logger = logging.getLogger("ml_framework.feature_selection")


# =============================================================================
# COMBINED SELECTION (STATISTICAL + PCA)
# =============================================================================


def combined_feature_selection(
    data: pd.DataFrame,
    target_column: str,
    statistical_results: Dict[str, float],
    pca_components: Optional[np.ndarray] = None,
    alpha_statistical: float = 0.05,
    n_pca_features: int = 10,
    weight_statistical: float = 0.6,
    weight_pca: float = 0.4,
    top_n: int = 10,
    plot_results: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Combined feature selection based on statistical tests and PCA loadings.

    Parameters
    ----------
    data               : full DataFrame
    target_column      : target column name
    statistical_results: dict {feature: p_value} from statistical tests
    pca_components     : array (n_features, n_components) — PCA loadings (optional)
    alpha_statistical  : significance threshold
    n_pca_features     : number of PCA components to consider
    weight_statistical : weight for the statistical score (0–1)
    weight_pca         : weight for the PCA score (0–1)
    top_n              : number of final features to select
    plot_results       : display visualizations
    random_state       : random seed

    Returns
    -------
    dict with keys:
        significant_features_statistical, top_features_combined,
        scores_detail, final_ranking, summary
    """
    if not statistical_results:
        raise ValueError("statistical_results is empty.")

    features = list(statistical_results.keys())

    stat_raw  = {k: 1.0 - float(v) for k, v in statistical_results.items()}
    max_stat  = max(stat_raw.values()) or 1.0
    stat_norm = {k: v / max_stat for k, v in stat_raw.items()}

    if pca_components is not None:
        n_comp = min(n_pca_features, pca_components.shape[1] if pca_components.ndim > 1 else 1)
        if pca_components.ndim == 1:
            pca_imp = np.abs(pca_components[:len(features)])
        else:
            pca_imp = np.abs(pca_components[:len(features), :n_comp]).mean(axis=1)
        pca_raw = dict(zip(features, pca_imp))
    else:
        np.random.seed(random_state)
        pca_raw = {k: np.random.random() for k in features}

    max_pca  = max(pca_raw.values()) or 1.0
    pca_norm = {k: v / max_pca for k, v in pca_raw.items()}

    combined = {
        feat: weight_statistical * stat_norm[feat] + weight_pca * pca_norm[feat]
        for feat in features
    }

    sorted_features = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    top_features    = [f[0] for f in sorted_features[:top_n]]
    significant_stat = [k for k, v in statistical_results.items() if float(v) < alpha_statistical]

    if plot_results:
        plot_combined_selection(
            features, stat_norm, pca_norm, combined, sorted_features,
            statistical_results, alpha_statistical, top_n,
        )

    results = {
        "method": "Combined Selection (Statistical + PCA)",
        "parameters": {
            "alpha_statistical": alpha_statistical,
            "n_pca_features":    n_pca_features,
            "weight_statistical": weight_statistical,
            "weight_pca":         weight_pca,
        },
        "significant_features_statistical": significant_stat,
        "top_features_combined":            top_features,
        "scores_detail": {
            "statistical": stat_norm,
            "pca":         pca_norm,
            "combined":    combined,
        },
        "final_ranking": sorted_features,
        "summary": {
            "n_significant_stat":   len(significant_stat),
            "n_selected":           len(top_features),
            "top_feature":          sorted_features[0][0] if sorted_features else None,
            "top_score":            sorted_features[0][1] if sorted_features else None,
        },
    }

    print(f"\n  Combined selection: {len(top_features)} features selected")
    print(f"    Top features: {top_features[:5]}")
    return results


# =============================================================================
# ROBUSTNESS ANALYSIS
# =============================================================================


def robustness_analysis(
    data: pd.DataFrame,
    target_column: str,
    statistical_results: Dict[str, float],
    weight_range: Optional[np.ndarray] = None,
    plot: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze selection robustness across different weighting schemes.

    Varies the statistical method weight from 0.1 to 0.9 and observes
    the stability of the top-5 features.

    Parameters
    ----------
    data               : full DataFrame
    target_column      : target column name
    statistical_results: dict {feature: p_value}
    weight_range       : weights to test (default: np.arange(0.1, 1.0, 0.1))
    plot               : display visualizations

    Returns
    -------
    list of dicts — one per tested weight
    """
    if weight_range is None:
        weight_range = np.arange(0.1, 1.0, 0.1)

    robustness_results = []

    for w_stat in weight_range:
        w_pca = round(1.0 - w_stat, 6)
        res = combined_feature_selection(
            data, target_column, statistical_results,
            weight_statistical=float(w_stat),
            weight_pca=float(w_pca),
            plot_results=False,
        )
        robustness_results.append({
            "weight_stat":   round(float(w_stat), 2),
            "weight_pca":    round(float(w_pca),  2),
            "top_5_features": res["top_features_combined"][:5],
            "top_feature":   res["final_ranking"][0][0],
            "top_score":     res["final_ranking"][0][1],
        })

    if plot:
        plot_robustness(robustness_results)

    return robustness_results


# =============================================================================
# MODEL-BASED SELECTION
# =============================================================================


def model_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    top_n: int = 20,
    plot: bool = True,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Feature selection based on a tree model's importance scores.

    Parameters
    ----------
    X            : features
    y            : target
    model        : sklearn estimator (default: RandomForestClassifier)
    top_n        : number of features to retain
    plot         : display the importance chart
    random_state : random seed

    Returns
    -------
    (selected_features, importance_df)
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        perm = perm_imp(model, X, y, n_repeats=5, random_state=random_state)
        importances = perm.importances_mean

    importance_df = (
        pd.DataFrame({"feature": X.columns.tolist(), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    selected = importance_df.head(top_n)["feature"].tolist()

    if plot:
        top_df = importance_df.head(min(top_n, 25))
        plot_model_based_importance(top_df, model.__class__.__name__)

    print(f"\n  Model-based selection ({model.__class__.__name__}): {len(selected)} features retained")
    return selected, importance_df


# =============================================================================
# RECURSIVE FEATURE ELIMINATION
# =============================================================================


def recursive_feature_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    estimator=None,
    cv: int = 5,
    scoring: str = "roc_auc",
    min_features_to_select: int = 5,
    random_state: int = 42,
) -> Tuple[List[str], Any]:
    """
    Cross-validated recursive feature elimination (RFECV).

    Parameters
    ----------
    X                      : features
    y                      : target
    estimator              : sklearn estimator (default: RandomForestClassifier)
    cv                     : number of CV folds
    scoring                : optimization metric
    min_features_to_select : minimum number of features to keep
    random_state           : random seed

    Returns
    -------
    (selected_features, rfecv_object)
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv_splitter,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )

    rfecv.fit(X, y)
    selected = X.columns[rfecv.support_].tolist()

    print(f"\n  RFECV: {len(selected)} optimal features (scoring={scoring})")
    print(f"    Features: {selected[:10]}")

    try:
        mean_scores = rfecv.cv_results_["mean_test_score"]
        plot_rfecv_curve(mean_scores, min_features_to_select, rfecv.n_features_, scoring)
    except Exception as exc:
        logger.warning("RFECV visualization failed: %s", exc)

    return selected, rfecv
