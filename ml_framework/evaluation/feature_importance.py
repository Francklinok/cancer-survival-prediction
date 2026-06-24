"""
feature_importance.py — Comprehensive feature importance analysis.

Methods:
  - Intrinsic importances (tree-based models)
  - Coefficients (linear models)
  - Permutation Importance (model-agnostic, robust to impurity bias)
  - SHAP mean absolute values
  - Multi-method comparison

Visualizations delegated to visualization.evaluation.importance_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_framework.visualization.evaluation.importance_plots import (
    plot_feature_importance_bar,
    plot_cumulative_importance,
    plot_permutation_importance,
    plot_importance_methods_comparison,
)
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.feature_importance")


# ──────────────────────────────────────────────────────────────────────────────
# INTRINSIC IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────


def create_feature_importance_summary(
    model,
    feature_names: List[str],
    top_n: int = 15,
    plot: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Summarize feature importance based on the model type.

    Supports:
      - feature_importances_  (Random Forest, GBM, XGBoost, LightGBM, ...)
      - coef_                 (LogisticRegression, LinearSVC, SGD, ...)

    Parameters
    ----------
    model         : trained sklearn-compatible model
    feature_names : list of feature names
    top_n         : number of top features to display
    plot          : display visualizations

    Returns
    -------
    pd.DataFrame with columns: Feature, Importance, Importance_pct,
                                Importance_cumsum_pct
    """
    section_header("FEATURE IMPORTANCE")

    importances: Optional[np.ndarray] = None
    method_name = ""

    if hasattr(model, "feature_importances_"):
        importances = np.abs(model.feature_importances_)
        method_name = "feature_importances_ (impurity)"

    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        method_name = "coefficients (absolute value)"

    if importances is None:
        print(" This model does not expose intrinsic importances.")
        print("  → Use permutation_feature_importance() instead.")
        return None

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    total = importance_df["Importance"].sum()
    importance_df["Importance_pct"]        = (importance_df["Importance"] / total * 100).round(2)
    importance_df["Importance_cumsum_pct"] = importance_df["Importance_pct"].cumsum().round(2)

    top = importance_df.head(top_n)
    print(f"\n  Method : {method_name}")
    print(f"\n  Top {top_n} features:")
    print(top[["Feature", "Importance", "Importance_pct"]].to_string(index=False))

    top5_pct = importance_df.head(5)["Importance_pct"].sum()
    print(f"\n  The top 5 features account for {top5_pct:.1f}% of total importance.")

    if plot:
        plot_feature_importance_bar(top, method_name, top_n)
        plot_cumulative_importance(importance_df)

    return importance_df

# ──────────────────────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

def permutation_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    top_n: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance (model-agnostic, corrects impurity bias).

    Parameters
    ----------
    model        : trained model
    X_test       : test features
    y_test       : test target
    n_repeats    : number of permutation repeats
    scoring      : evaluation metric
    top_n        : number of top features to display
    random_state : random seed

    Returns
    -------
    pd.DataFrame with columns: Feature, Importance_mean, Importance_std
    """
    section_header("PERMUTATION IMPORTANCE")
    print(f"  Scoring: {scoring} | n_repeats: {n_repeats}")

    try:
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception as exc:
        logger.error("Permutation importance failed: %s", exc)
        return pd.DataFrame()

    perm_df = (
        pd.DataFrame({
            "Feature":         X_test.columns,
            "Importance_mean": result.importances_mean,
            "Importance_std":  result.importances_std,
        })
        .sort_values("Importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    top = perm_df.head(top_n)
    plot_permutation_importance(top, scoring, top_n)

    negative = perm_df[perm_df["Importance_mean"] < 0]
    if not negative.empty:
        print(f"\n Harmful features (importance < 0): {negative['Feature'].tolist()}")
        print("     → Consider removing them.")

    return perm_df

# ──────────────────────────────────────────────────────────────────────────────
# MULTI-METHOD COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def compare_importance_methods(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compare intrinsic importance vs permutation importance.

    Returns
    -------
    pd.DataFrame — feature rank comparison across methods
    """
    methods: Dict[str, pd.Series] = {}

    intrinsic = create_feature_importance_summary(
        model, X_test.columns.tolist(), top_n=top_n, plot=False
    )
    if intrinsic is not None:
        methods["Intrinsic"] = intrinsic.set_index("Feature")["Importance"].rank(ascending=False)

    perm_df = permutation_feature_importance(model, X_test, y_test, top_n=top_n)
    if not perm_df.empty:
        methods["Permutation"] = perm_df.set_index("Feature")["Importance_mean"].rank(ascending=False)

    if len(methods) < 2:
        return pd.DataFrame()

    rank_df = pd.DataFrame(methods).dropna()
    plot_importance_methods_comparison(rank_df)

    return rank_df
