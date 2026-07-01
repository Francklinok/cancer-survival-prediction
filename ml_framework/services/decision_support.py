"""
decision_support.py — Clinical decision support system.

Wraps model prediction logic in a clinician-oriented interface:
  - Individual risk scores
  - Contextualized recommendations (action / follow-up / monitoring)
  - Identification of dominant risk factors
  - Configurable thresholds

Public functions:
  - create_clinical_decision_support(model, feature_names, thresholds, condition)
    → callable decision_support(patient_data) → dict
  - batch_decision_support(model, X, feature_names, thresholds) → pd.DataFrame
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.decision_support")


# =============================================================================
# CLINICAL DECISION SUPPORT SYSTEM
# =============================================================================


def create_clinical_decision_support(
    model,
    feature_names: List[str],
    thresholds: Optional[Dict[str, float]] = None,
    target_condition: str = "the disease",
) -> Callable[[Union[Dict, pd.DataFrame]], Dict[str, Any]]:
    """
    Create a clinical decision support callable based on the model.

    Parameters
    ----------
    model            : trained sklearn estimator (with predict_proba if possible)
    feature_names    : feature names in the order expected by the model
    thresholds       : custom decision thresholds:
                         'action_required' (default 0.7) — intervention recommended
                         'follow_up'       (default 0.4) — follow-up recommended
    target_condition : name of the predicted medical condition (for messages)

    Returns
    -------
    callable decision_support(patient_data: dict | pd.DataFrame) → dict
    """
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    if thresholds is None:
        thresholds = {
            "action_required": 0.7,
            "follow_up":       0.4,
        }

    print("\n" + "═" * 58)
    print("  CLINICAL DECISION SUPPORT SYSTEM")
    print("═" * 58)
    print(f"  Predicted condition       : {target_condition}")
    print(f"  Action required threshold : {thresholds['action_required']:.0%}")
    print(f"  Follow-up threshold       : {thresholds['follow_up']:.0%}")
    print(f"  Features used             : {len(feature_names)}")

    # Pre-compute model feature importances (shared across all predictions)
    _feature_importances: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        _feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        _feature_importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    def decision_support(
        patient_data: Union[Dict[str, Any], pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Assess risk for a patient and provide a clinical recommendation.

        Parameters
        ----------
        patient_data : dict {feature: value} or pd.DataFrame (1 or N rows)

        Returns
        -------
        dict with keys:
            error, risk_score, recommendation, details, action,
            risk_factors, risk_level, confidence
        """
        # Prepare input DataFrame
        if isinstance(patient_data, dict):
            missing = [f for f in feature_names if f not in patient_data]
            if missing:
                return {
                    "error":   True,
                    "message": f"Missing features: {', '.join(missing)}",
                }
            df = pd.DataFrame([{f: patient_data.get(f, 0) for f in feature_names}])
        elif isinstance(patient_data, pd.DataFrame):
            df = patient_data.reindex(columns=feature_names, fill_value=0)
        else:
            return {"error": True, "message": "Unsupported data format."}

        # Predict risk score
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)
                risk_score = float(proba[0, 1]) if proba.shape[1] == 2 else float(proba[0].max())
            else:
                risk_score = float(model.predict(df)[0])
        except Exception as exc:
            logger.error("Patient prediction error: %s", exc)
            return {"error": True, "message": f"Prediction error: {exc}"}

        # Risk category
        if risk_score >= thresholds["action_required"]:
            risk_level     = "High"
            recommendation = "Clinical action required"
            details        = (f"High risk of {target_condition} "
                              f"({risk_score:.1%}). Intervention recommended.")
            action         = "Urgent specialist consultation recommended."
        elif risk_score >= thresholds["follow_up"]:
            risk_level     = "Moderate"
            recommendation = "Follow-up recommended"
            details        = (f"Moderate risk of {target_condition} "
                              f"({risk_score:.1%}). Active monitoring.")
            action         = "Schedule a short-term follow-up (3–6 months)."
        else:
            risk_level     = "Low"
            recommendation = "Low risk"
            details        = (f"Low risk of {target_condition} "
                              f"({risk_score:.1%}). Standard monitoring.")
            action         = "Regular follow-up per standard protocol."

        # Top-3 dominant risk factors
        risk_factors: List[str] = []
        if _feature_importances is not None and len(_feature_importances) == len(feature_names):
            patient_row = df.iloc[0]
            weighted = [
                (feature_names[i], float(_feature_importances[i]) * float(patient_row.iloc[i]))
                for i in range(len(feature_names))
            ]
            weighted.sort(key=lambda x: abs(x[1]), reverse=True)
            risk_factors = [f"{n} ({v:+.4f})" for n, v in weighted[:3]]

        return {
            "error":          False,
            "risk_score":     round(risk_score, 4),
            "risk_level":     risk_level,
            "recommendation": recommendation,
            "details":        details,
            "action":         action,
            "risk_factors":   risk_factors,
        # NOTE: This "confidence" score ONLY measures how far the model is from the decision threshold (0.5).
        # It is NOT the true probability of being correct. A poorly-calibrated model can easily claim 90% confidence 
        # while being completely wrong. For a true medical certainty metric, the model would need proper calibration 
        # (e.g., Platt scaling or Isotonic regression). Treat this purely as a distance proxy, nothing more.
        "confidence":     round(abs(risk_score - 0.5) * 2, 4),  # Distance to threshold, not calibrated certainty
            "confidence":     round(abs(risk_score - 0.5) * 2, 4),  # decision-boundary distance, not calibrated uncertainty
        }

    return decision_support


# =============================================================================
# BATCH EVALUATION
# =============================================================================


def batch_decision_support(
    model,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    target_condition: str = "the disease",
) -> pd.DataFrame:
    """
    Apply clinical decision support to an entire DataFrame.

    Parameters
    ----------
    model            : trained sklearn estimator
    X                : feature DataFrame (N patients)
    feature_names    : feature names (default: X.columns)
    thresholds       : decision thresholds
    target_condition : predicted medical condition

    Returns
    -------
    pd.DataFrame with columns:
        risk_score, risk_level, recommendation, action, confidence
    """
    if feature_names is None:
        feature_names = X.columns.tolist()

    if thresholds is None:
        thresholds = {"action_required": 0.7, "follow_up": 0.4}

    # Batch predictions
    X_aligned = X.reindex(columns=feature_names, fill_value=0)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_aligned)
        risk_scores = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    else:
        risk_scores = model.predict(X_aligned).astype(float)

    # Categorization
    risk_levels = np.where(
        risk_scores >= thresholds["action_required"], "High",
        np.where(risk_scores >= thresholds["follow_up"], "Moderate", "Low")
    )
    recommendations = np.where(
        risk_scores >= thresholds["action_required"], "Action required",
        np.where(risk_scores >= thresholds["follow_up"], "Follow-up recommended", "Low risk")
    )
    confidence = np.abs(risk_scores - 0.5) * 2

    result_df = pd.DataFrame({
        "risk_score":     np.round(risk_scores, 4),
        "risk_level":     risk_levels,
        "recommendation": recommendations,
        "confidence":     np.round(confidence, 4),
    }, index=X.index)

    # Summary statistics
    print(f"\n  Decision support — {len(result_df)} patients")
    for level in ["High", "Moderate", "Low"]:
        n = (result_df["risk_level"] == level).sum()
        pct = 100 * n / len(result_df)
        print(f"    {level:<10} : {n:>4} ({pct:.1f}%)")

    return result_df
