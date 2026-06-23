"""
class_imbalance.py — Class imbalance analysis and rebalancing.

Features:
  - Complete imbalance diagnostics (ratio, entropy, visualizations)
  - Rebalancing strategies: SMOTE, ADASYN, BorderlineSMOTE,
    RandomOverSampler, RandomUnderSampler, TomekLinks, SMOTEENN
  - Rebalancing impact evaluation
  - Automatic recommendations based on imbalance level
  - Medical contextual interpretations

All plotting delegated to visualization.analysis.class_balance_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ml_framework.visualization.analysis.class_balance_plots import (
    plot_class_distribution,
    plot_rebalance_comparison,
)

logger = logging.getLogger("ml_framework.class_balance")


# ──────────────────────────────────────────────────────────────────────────────
# IMBALANCE DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────


def diagnose_class_imbalance(
    y,
    verbose: bool = True,
    plot: bool = True,
) -> Dict:
    """
    Complete class imbalance analysis.

    Returns
    -------
    dict with: class_counts, class_pct, imbalance_ratio,
               entropy, severity, recommendations
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    counts    = y.value_counts().sort_values(ascending=False)
    pct       = (counts / len(y) * 100).round(2)
    n_classes = len(counts)

    majority_count = counts.iloc[0]
    minority_count = counts.iloc[-1]
    ratio = majority_count / minority_count if minority_count > 0 else float("inf")

    probs       = (counts / len(y)).values
    entropy     = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_entropy = float(np.log2(n_classes))
    balance_score = entropy / max_entropy if max_entropy > 0 else 1.0

    severity = (
        "balanced" if ratio < 1.5
        else "mild"     if ratio < 3
        else "moderate" if ratio < 10
        else "severe"   if ratio < 50
        else "extreme"
    )

    recommendations = _get_recommendations(ratio, severity)

    result = {
        "class_counts":   counts.to_dict(),
        "class_pct":      pct.to_dict(),
        "n_classes":      n_classes,
        "imbalance_ratio": round(ratio, 2),
        "entropy":        round(entropy, 4),
        "balance_score":  round(balance_score, 4),
        "severity":       severity,
        "majority_class": counts.index[0],
        "minority_class": counts.index[-1],
        "recommendations": recommendations,
    }

    if verbose:
        _print_diagnosis(result)

    if plot:
        plot_class_distribution(counts, pct, severity, ratio)

    return result


def _get_recommendations(ratio: float, severity: str) -> list:
    base = [
        "Use 'class_weight=\"balanced\"' in sklearn models.",
        "Evaluate with F1-macro, ROC-AUC, PR-AUC rather than accuracy.",
        "Adjust the classification threshold (Youden index).",
    ]
    if severity in ("moderate", "severe", "extreme"):
        base += [
            "Apply SMOTE or ADASYN (synthetic oversampling).",
            "Consider Focal Loss for neural networks.",
        ]
    if severity in ("severe", "extreme"):
        base += [
            "Combine over + under sampling (SMOTEENN, SMOTETomek).",
            "Consider anomaly detection as an alternative approach.",
        ]
    return base


def _print_diagnosis(result: Dict) -> None:
    print("\n" + "═" * 60)
    print("  CLASS IMBALANCE DIAGNOSTICS")
    print("═" * 60)
    print(f"  Number of classes   : {result['n_classes']}")
    print(f"  Imbalance ratio     : {result['imbalance_ratio']:.2f}:1")
    print(f"  Shannon entropy     : {result['entropy']:.4f}")
    print(f"  Balance score       : {result['balance_score']:.4f} (1.0 = perfect)")
    print(f"  Severity            : {result['severity'].upper()}")
    print()
    for cls, cnt in result["class_counts"].items():
        bar = "█" * int(result["class_pct"][cls] / 2)
        print(f"  Class {str(cls):<15} {cnt:>6} obs. ({result['class_pct'][cls]:5.1f}%)  {bar}")
    print()
    print("  Recommendations:")
    for rec in result["recommendations"]:
        print(f"    • {rec}")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# REBALANCING
# ──────────────────────────────────────────────────────────────────────────────


def rebalance_classes(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "smote",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply a class rebalancing strategy.

    Parameters
    ----------
    X            : features
    y            : target
    strategy     : 'smote' | 'adasyn' | 'borderline_smote' |
                   'over' | 'under' | 'tomek' | 'smoteenn' | 'smotetomek'
    random_state : random seed

    Returns
    -------
    X_res, y_res : rebalanced data
    """
    strategy = strategy.lower()

    if strategy in ("smote", "adasyn", "borderline_smote", "smoteenn", "smotetomek"):
        logger.warning(
            "rebalance_classes('%s'): MUST be called on training data ONLY "
            "(after train_test_split). Applying to the full dataset before splitting "
            "leaks test-set information through synthetic interpolated points. "
            "Pass X_train, y_train — not X, y.",
            strategy,
        )
        print(
            f"\n WARNING: '{strategy}' generates synthetic samples between "
            f"real observations.\n"
            f"     Call rebalance_classes() AFTER train_test_split, on X_train/y_train only.\n"
            f"     Applying before split contaminates the test set with synthetic data.\n"
        )

    try:
        if strategy == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)

        elif strategy == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=random_state)

        elif strategy == "borderline_smote":
            from imblearn.over_sampling import BorderlineSMOTE
            sampler = BorderlineSMOTE(random_state=random_state)

        elif strategy == "over":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=random_state)

        elif strategy == "under":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)

        elif strategy == "tomek":
            from imblearn.under_sampling import TomekLinks
            sampler = TomekLinks()

        elif strategy == "smoteenn":
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN(random_state=random_state)

        elif strategy == "smotetomek":
            from imblearn.combine import SMOTETomek
            sampler = SMOTETomek(random_state=random_state)

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: "
                "smote, adasyn, borderline_smote, over, under, tomek, smoteenn, smotetomek"
            )

        X_res, y_res = sampler.fit_resample(X, y)

    except ImportError:
        logger.error("imbalanced-learn not installed: pip install imbalanced-learn")
        return X, y

    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)

    if verbose:
        print(f"\n  Rebalancing '{strategy}' applied")
        print(f"  Before: {len(y)} obs. — distribution: {y.value_counts().to_dict()}")
        print(f"  After : {len(y_res)} obs. — distribution: {y_res.value_counts().to_dict()}")
        plot_rebalance_comparison(y, y_res, strategy)

    return X_res, y_res
