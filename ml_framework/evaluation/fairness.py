"""
fairness.py — Algorithmic fairness audit for the ML model.

Fairness metrics computed:
  - Statistical Parity (selection rate per group)
  - Equal Opportunity (TPR per group)
  - Predictive Parity (PPV per group)
  - Equalized Odds
  - 80% Rule (EEOC directive)
  - Disparate Impact Ratio

Automatic recommendations are generated based on detected disparities.
Visualizations delegated to visualization.evaluation.fairness_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from ml_framework.visualization.evaluation.fairness_plots import plot_fairness_metrics
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.fairness")


# ──────────────────────────────────────────────────────────────────────────────
# FAIRNESS AUDIT
# ──────────────────────────────────────────────────────────────────────────────


def fairness_audit(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_attributes: Dict | List,
    threshold: float = 0.50,
    disparity_threshold: float = 0.20,
    verbose: bool = True,
) -> Dict:
    """
    Complete fairness audit for each sensitive attribute.

    Parameters
    ----------
    model                : trained model
    X_test               : test features (DataFrame)
    y_test               : test target
    sensitive_attributes : dict {attr_name: col_name} or list of column names
    threshold            : classification threshold for probabilities
    disparity_threshold  : threshold for flagging a disparity (default 0.20)
    verbose              : detailed output

    Returns
    -------
    dict — audit results per sensitive attribute
    """
    section_header("MODEL FAIRNESS AUDIT")

    # Predictions
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = y_pred.astype(float)

    # Normalize sensitive attributes
    if isinstance(sensitive_attributes, list):
        attr_names = sensitive_attributes
    elif isinstance(sensitive_attributes, dict):
        attr_names = list(sensitive_attributes.keys())
    else:
        raise TypeError("sensitive_attributes must be a list or a dict.")

    analysis_df = pd.DataFrame({
        "true":  y_test.values,
        "pred":  y_pred,
        "score": y_proba,
    }, index=y_test.index)

    # Attach sensitive attributes
    for attr in attr_names:
        col = attr if isinstance(sensitive_attributes, list) else sensitive_attributes[attr]
        if isinstance(col, int):
            analysis_df[attr] = X_test.iloc[:, col].values
        elif col in X_test.columns:
            analysis_df[attr] = X_test[col].values
        else:
            logger.warning("Attribute '%s' not found in X_test — skipped.", attr)

    audit_results: Dict = {}

    for attr in attr_names:
        if attr not in analysis_df.columns:
            continue

        print(f"\n  Sensitive attribute: {attr}")
        group_metrics = _compute_group_metrics(analysis_df, attr)
        disparities   = _compute_disparities(group_metrics)

        if verbose:
            _print_group_metrics(group_metrics, disparities, attr, disparity_threshold)

        plot_fairness_metrics(group_metrics, attr)

        audit_results[attr] = {
            "group_metrics": group_metrics,
            "disparities":   disparities,
        }

    _print_recommendations(audit_results, disparity_threshold)

    return audit_results


# ──────────────────────────────────────────────────────────────────────────────
# GROUP METRICS
# ──────────────────────────────────────────────────────────────────────────────


def _compute_group_metrics(analysis_df: pd.DataFrame, attr: str) -> Dict:
    """Compute TPR, FPR, PPV, TNR, selection_rate per group."""
    group_metrics: Dict = {}

    for value in analysis_df[attr].unique():
        grp = analysis_df[analysis_df[attr] == value]
        n   = len(grp)

        tp = int(((grp["pred"] == 1) & (grp["true"] == 1)).sum())
        fp = int(((grp["pred"] == 1) & (grp["true"] == 0)).sum())
        tn = int(((grp["pred"] == 0) & (grp["true"] == 0)).sum())
        fn = int(((grp["pred"] == 0) & (grp["true"] == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Sensitivity / Equal Opportunity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Precision / Predictive Parity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Specificity
        sel = (tp + fp)  / n if n > 0 else 0.0            # Statistical Parity

        group_metrics[value] = {
            "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr":            round(tpr, 4),
            "fpr":            round(fpr, 4),
            "ppv":            round(ppv, 4),
            "tnr":            round(tnr, 4),
            "selection_rate": round(sel, 4),
        }

    return group_metrics


def _compute_disparities(group_metrics: Dict) -> Dict:
    """Compute inter-group disparities.

    Reports BOTH the absolute gap AND the ratio for each metric.
    The previous code reported only the relative gap (max - min) / max, which
    conflates the two and is unstable when the max approaches zero.

    Medical context thresholds (stricter than EEOC general industry):
      absolute gap > 0.10 → clinically significant disparity
      ratio < 0.80        → EEOC 80% Rule violation (legal threshold)
    """
    metrics_keys = ["tpr", "ppv", "selection_rate"]
    disparities: Dict = {}

    MEDICAL_ABS_THRESHOLD = 0.10  # WHO / medical ethics: 10% absolute gap is material

    for key in metrics_keys:
        vals = [m[key] for m in group_metrics.values() if m["n"] > 0]
        if len(vals) < 2:
            continue
        max_v, min_v = max(vals), min(vals)
        abs_gap = max_v - min_v
        ratio   = min_v / max_v if max_v > 0 else 1.0
        disparities[f"{key}_abs_gap"] = round(float(abs_gap), 4)
        disparities[f"{key}_ratio"]   = round(float(ratio),   4)
        # Flag if either criterion is violated (AND logic: both needed for full compliance)
        disparities[f"{key}_flagged"]  = bool(abs_gap > MEDICAL_ABS_THRESHOLD or ratio < 0.90)

    # 80% Rule (Disparate Impact Ratio) — EEOC guideline: min_rate / max_rate >= 0.80
    sel_vals = [m["selection_rate"] for m in group_metrics.values() if m["n"] > 0]
    if len(sel_vals) >= 2:
        dir_ratio = min(sel_vals) / max(sel_vals) if max(sel_vals) > 0 else 1.0
        disparities["disparate_impact_ratio"] = round(float(dir_ratio), 4)
        disparities["eighty_percent_rule"]    = bool(dir_ratio >= 0.80)

    return disparities


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────────────────────────


def _print_group_metrics(group_metrics, disparities, attr, disp_thresh):
    metrics_df = pd.DataFrame(group_metrics).T[["n", "selection_rate", "tpr", "ppv", "tnr"]]
    print(metrics_df.to_string())

    print("\n  Disparities (absolute gap | ratio | medical threshold > 0.10):")
    for base_key in ["tpr", "ppv", "selection_rate"]:
        abs_gap = disparities.get(f"{base_key}_abs_gap")
        ratio   = disparities.get(f"{base_key}_ratio")
        flagged = disparities.get(f"{base_key}_flagged", False)
        if abs_gap is None:
            continue
        flag = "⚠️" if flagged else "✅"
        print(f"    {base_key:<20} abs_gap={abs_gap:.4f}  ratio={ratio:.4f}  {flag}")

    if "eighty_percent_rule" in disparities:
        status = "Compliant" if disparities["eighty_percent_rule"] else "⚠️  Non-compliant"
        print(f"    80% Rule                  : {status}")
    if "disparate_impact_ratio" in disparities:
        print(f"    Disparate Impact Ratio    : {disparities['disparate_impact_ratio']:.4f}  (threshold: ≥ 0.80)")


def _print_recommendations(audit_results: Dict, disp_thresh: float) -> None:
    section_header("FAIRNESS RECOMMENDATIONS")
    has_issues = False

    for attr, res in audit_results.items():
        disp   = res.get("disparities", {})
        flagged_metrics = [k.replace("_flagged", "") for k, v in disp.items()
                           if k.endswith("_flagged") and v is True]

        if flagged_metrics:
            has_issues = True
            print(f"\n  Attribute '{attr}' — significant disparities (abs_gap > 0.10):")
            for base in flagged_metrics:
                abs_gap = disp.get(f"{base}_abs_gap", 0)
                ratio   = disp.get(f"{base}_ratio", 1)
                print(f"    • {base}: abs_gap={abs_gap:.3f}  ratio={ratio:.3f}")

            if "selection_rate" in flagged_metrics:
                print("    → Rebalance the data (re-sampling, class weighting)")
            if "tpr" in flagged_metrics:
                print("    → Apply Equal Opportunity correction (group-adaptive threshold)")
            if "ppv" in flagged_metrics:
                print("    → Calibrate model separately per group")

        if not disp.get("eighty_percent_rule", True):
            print(f"\n  ⚠️  80% Rule violated for '{attr}' — high legal/ethical risk.")

    if not has_issues:
        print("  ✅ No significant disparity detected. Model appears fair.")
