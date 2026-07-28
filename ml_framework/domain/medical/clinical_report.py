"""
clinical_report.py — Clinical medical report for the prediction model.

Provides:
  - Patient risk stratification (low / moderate / high / critical)
  - Optimal threshold (Youden J) and clinical cost-benefit analysis
  - High-risk patient profiles
  - ROC curve with clinical interpretation
  - Executive summary for clinicians

Visualizations delegated to visualization.evaluation.clinical_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ml_framework.visualization.evaluation.clinical_plots import plot_risk_visualizations
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.clinical_report")

DEFAULT_RISK_THRESHOLDS = {
    "Very Low":  0.20,
    "Low":       0.40,
    "Moderate":  0.60,
    "High":      0.80,
}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN MEDICAL REPORT
# ──────────────────────────────────────────────────────────────────────────────


def medical_model_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
    patient_ids: Optional[List] = None,
    risk_thresholds: Optional[Dict[str, float]] = None,
    generate_profiles: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate a complete medical report for clinical interpretation.

    Parameters
    ----------
    model             : trained model
    X_test            : test features
    y_test            : test target
    feature_names     : feature names
    patient_ids       : patient identifiers
    risk_thresholds   : custom risk thresholds
    generate_profiles : generate high-risk patient profiles
    verbose           : detailed output

    Returns
    -------
    pd.DataFrame — predictions, scores, and risk categories per patient
    """
    section_header("CLINICAL MODEL REPORT")

    if feature_names is None and hasattr(X_test, "columns"):
        feature_names = X_test.columns.tolist()

    thresholds = risk_thresholds or DEFAULT_RISK_THRESHOLDS

    # Predictions
    y_pred   = model.predict(X_test)
    has_proba = hasattr(model, "predict_proba")
    y_proba  = model.predict_proba(X_test)[:, 1] if has_proba else y_pred.astype(float)

    results = pd.DataFrame({
        "Patient_ID":       patient_ids if patient_ids is not None else range(len(y_test)),
        "Real_Status":      y_test.values if hasattr(y_test, "values") else y_test,
        "Predicted_Status": y_pred,
        "Risk_Score":       y_proba,
    })

    results["Risk_Category"] = results["Risk_Score"].apply(
        lambda s: _categorize_risk(s, thresholds)
    )

    if verbose:
        _print_risk_distribution(results)

    plot_risk_visualizations(results, thresholds, y_test, y_proba, has_proba)

    # Optimal threshold (Youden J)
    if has_proba:
        fpr, tpr, thresh = roc_curve(y_test, y_proba)
        j_idx      = int(np.argmax(tpr - fpr))
        opt_thresh = float(thresh[j_idx])
        print(f"\n  Optimal threshold (Youden J) : {opt_thresh:.4f}")
        print(f"  At this threshold → Sensitivity={tpr[j_idx]:.3f}  Specificity={1-fpr[j_idx]:.3f}")

        _cost_benefit_analysis(y_test, y_proba, opt_thresh)

    if generate_profiles and has_proba and feature_names:
        _generate_high_risk_profiles(results, X_test, feature_names)

    _executive_summary(results, model)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# RISK CATEGORIZATION
# ──────────────────────────────────────────────────────────────────────────────


def _categorize_risk(score: float, thresholds: Dict[str, float]) -> str:
    """Map a probability score to a risk category label."""
    sorted_thresh = sorted(thresholds.items(), key=lambda x: x[1])
    for category, limit in sorted_thresh:
        if score <= limit:
            return category
    return list(thresholds.keys())[-1]


# ──────────────────────────────────────────────────────────────────────────────
# RISK DISTRIBUTION REPORT
# ──────────────────────────────────────────────────────────────────────────────


def _print_risk_distribution(results: pd.DataFrame) -> None:
    risk_counts = results["Risk_Category"].value_counts()
    n           = len(results)

    print("\n  Risk level distribution:")
    risk_order = ["Very Low", "Low", "Moderate", "High"]

    for cat in risk_order:
        if cat in risk_counts:
            cnt = risk_counts[cat]
            pct = cnt / n * 100
            bar = "█" * int(pct / 5)
            print(f"    {cat:<12} {cnt:>5} patients  ({pct:5.1f}%)  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# COST-BENEFIT & PROFILES
# ──────────────────────────────────────────────────────────────────────────────


def _cost_benefit_analysis(y_test, y_proba, threshold: float) -> None:
    """Clinical cost-benefit analysis at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    print(f"\n  Cost-benefit analysis (threshold={threshold:.3f}):")
    print(f"    True Positives  (sick patients detected)     : {tp}")
    print(f"    False Positives (healthy misclassified as sick): {fp}")
    print(f"    False Negatives (sick patients missed — high cost): {fn}")
    print(f"    True Negatives  (healthy correctly classified) : {tn}")
    if fn > 0:
        print(f"\n {fn} sick patients were missed → high clinical risk.")
        print("     Consider lowering the threshold to reduce FN (at the cost of more FP).")


def _generate_high_risk_profiles(results, X_test, feature_names) -> None:
    """Display profiles of high-risk patients."""
    high_risk = results[results["Risk_Category"] == "High"]
    if high_risk.empty:
        print("\n  No high-risk patients identified.")
        return

    print(f"\n  High-risk patient profiles ({len(high_risk)} patients):")
    sample = high_risk.sample(min(3, len(high_risk)), random_state=42)

    for _, patient in sample.iterrows():
        pid   = patient["Patient_ID"]
        score = patient["Risk_Score"]
        idx   = list(results["Patient_ID"]).index(pid)

        features = (
            X_test.iloc[idx] if hasattr(X_test, "iloc")
            else pd.Series(X_test[idx], index=feature_names)
        )
        top_features = features.nlargest(5)

        print(f"\n    Patient {pid} — Score: {score:.4f}")
        print("    Top features:")
        for feat, val in top_features.items():
            print(f"      • {feat:<30} : {val:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ──────────────────────────────────────────────────────────────────────────────


def _executive_summary(results: pd.DataFrame, model) -> None:
    """Executive summary for clinicians."""
    n          = len(results)
    n_correct  = (results["Real_Status"] == results["Predicted_Status"]).sum()
    accuracy   = n_correct / n * 100
    high_risk_n = (results["Risk_Category"] == "High").sum()

    section_header("EXECUTIVE SUMMARY")
    print(f"  Model used        : {model.__class__.__name__}")
    print(f"  Patients analyzed : {n}")
    print(f"  Overall accuracy  : {accuracy:.1f}%")
    print(f"  High-risk patients identified: {high_risk_n} ({high_risk_n/n*100:.1f}%)")
    print()
    print("  Clinical recommendations:")
    print("    • 'High' risk patients should be reviewed as a priority.")
    print("    • Verify false negatives before clinical deployment.")
    print("    • Recalibrate the threshold to match institutional medical policy.")
