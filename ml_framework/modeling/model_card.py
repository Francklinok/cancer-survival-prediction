"""
model_card.py — Model Card generation (Google 2019 standard).

A Model Card is a transparency document that accompanies any model
deployed to production. It answers:
  - What is this model designed to do?
  - What are its actual performance metrics?
  - What are its biases and limitations?
  - Who can use it and in which contexts?

Reference: Mitchell et al., "Model Cards for Model Reporting" (FAccT 2019)

Public functions:
  - generate_model_card(model, X_train, X_test, y_train, y_test, ...) → dict
  - save_model_card(card, path, filename)                              → str
  - print_model_card(card)                                             → None
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.model_card")


# =============================================================================
# MODEL CARD GENERATION
# =============================================================================


def generate_model_card(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "Medical Model",
    target_description: str = "Treatment response",
    intended_use: str = "Clinical decision support",
    out_of_scope: str = "Do not use for definitive diagnosis",
    limitations: Optional[List[str]] = None,
    sensitive_attrs: Optional[List[str]] = None,
    version: str = "1.0",
) -> Dict[str, Any]:
    """
    Generate a complete Model Card following the Google (2019) standard.

    Parameters
    ----------
    model              : trained sklearn estimator
    X_train, y_train   : training data
    X_test, y_test     : test data
    model_name         : name of the model
    target_description : description of the target variable
    intended_use       : intended use case
    out_of_scope       : explicitly excluded uses
    limitations        : list of known limitations (auto-generated if None)
    sensitive_attrs    : sensitive attributes to monitor
    version            : model version string

    Returns
    -------
    dict — complete Model Card (JSON-serializable)
    """
    section_header("MODEL CARD GENERATION")

    n_classes  = int(y_test.nunique())
    y_pred     = model.predict(X_test)
    has_proba  = hasattr(model, "predict_proba")

    metrics: Dict[str, Any] = {
        "accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro":    round(float(f1_score(y_test, y_pred, average="macro",    zero_division=0)), 4),
    }

    if has_proba:
        if n_classes == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
            metrics["pr_auc"]  = round(float(average_precision_score(y_test, y_proba)), 4)
        else:
            y_proba = model.predict_proba(X_test)
            try:
                metrics["roc_auc_ovr"] = round(
                    float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")), 4
                )
            except Exception:
                pass

    try:
        clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        clf_report_clean = {
            str(k): {
                metric: round(float(v), 4) if isinstance(v, float) else v
                for metric, v in vals.items()
            }
            if isinstance(vals, dict) else round(float(vals), 4)
            for k, vals in clf_report.items()
        }
    except Exception:
        clf_report_clean = {}

    counts      = pd.Series(y_train).value_counts().to_dict()
    imbalance   = float(max(counts.values()) / max(min(counts.values()), 1))
    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else []

    top_features: Dict[str, float] = {}
    if hasattr(model, "feature_importances_") and feature_names:
        pairs = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )[:10]
        top_features = {k: round(float(v), 4) for k, v in pairs}

    elif hasattr(model, "coef_") and feature_names:
        coef = model.coef_
        imp  = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).flatten()
        if len(imp) == len(feature_names):
            pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:10]
            top_features = {k: round(float(v), 4) for k, v in pairs}

    if limitations is None:
        limitations = [
            f"Trained on {len(X_train)} patients — external generalization not validated.",
            f"Class imbalance (ratio {imbalance:.1f}×) — interpret Recall with caution.",
            "Prospective clinical validation required before deployment.",
            "Real-world performance may differ on external cohorts.",
        ]

    try:
        hyperparams = {
            k: v for k, v in model.get_params().items()
            if k not in {"verbose", "silent", "callbacks"}
        }
    except Exception:
        hyperparams = {}

    card: Dict[str, Any] = {
        "model_details": {
            "name":            model_name,
            "type":            type(model).__name__,
            "version":         version,
            "date":            datetime.now().date().isoformat(),
            "target":          target_description,
            "hyperparameters": {k: str(v) for k, v in list(hyperparams.items())[:15]},
        },
        "intended_use": {
            "primary_use":   intended_use,
            "primary_users": "Clinicians, medical data scientists",
            "out_of_scope":  out_of_scope,
        },
        "data_summary": {
            "n_train":                  int(len(X_train)),
            "n_test":                   int(len(X_test)),
            "n_features":               int(X_train.shape[1]),
            "feature_names_sample":     feature_names[:10],
            "class_distribution_train": {str(k): int(v) for k, v in counts.items()},
            "class_imbalance_ratio":    round(imbalance, 2),
        },
        "performance": {
            "metrics_test":          metrics,
            "classification_report": clf_report_clean,
        },
        "top_features": top_features,
        "limitations":  limitations,
        "ethical_considerations": {
            "sensitive_attributes": sensitive_attrs or [],
            "fairness_note":        "Fairness audit required before any clinical deployment.",
            "bias_warning":         "Biases in training data propagate to model predictions.",
            "gdpr_note":            "Medical data processing — GDPR compliance required.",
        },
        "usage_guidelines": {
            "recommended_threshold": 0.5,
            "monitoring":            "Monitor monthly PSI. Retrain if PSI > 0.25.",
            "update_policy":         "Revalidate every 6 months or upon detected drift.",
            "minimum_requirements":  {
                "input_features": int(X_train.shape[1]),
                "expected_range": "standardized data (same preprocessing pipeline)",
            },
        },
    }

    print(f"  Model          : {type(model).__name__} v{version}")
    print(f"  Accuracy       : {metrics.get('accuracy', 'N/A')}")
    print(f"  ROC-AUC        : {metrics.get('roc_auc', metrics.get('roc_auc_ovr', 'N/A'))}")
    print(f"  Top features   : {list(top_features.keys())[:5]}")
    print(f"  Limitations    : {len(limitations)} declared")
    print("Model Card generated.")

    return card


# =============================================================================
# SAVING
# =============================================================================


def save_model_card(
    card: Dict[str, Any],
    path: Union[str, Path] = "./reports/",
    filename: str = "model_card",
) -> str:
    """
    Save the Model Card as JSON and Markdown.

    Parameters
    ----------
    card     : dict returned by generate_model_card()
    path     : destination directory
    filename : base filename (without extension)

    Returns
    -------
    str — path to the created JSON file
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    fp_json = save_dir / f"{filename}.json"
    with open(fp_json, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False, default=str)

    # Markdown
    fp_md = save_dir / f"{filename}.md"
    md = _card_to_markdown(card)
    fp_md.write_text(md, encoding="utf-8")

    print(f"Model Card → {fp_json}")
    print(f"Model Card → {fp_md}")
    return str(fp_json)


def _card_to_markdown(card: Dict[str, Any]) -> str:
    """Convert a Model Card dict to a readable Markdown document."""
    details = card.get("model_details", {})
    perf    = card.get("performance", {}).get("metrics_test", {})
    data    = card.get("data_summary", {})
    use     = card.get("intended_use", {})

    lines = [
        f"# Model Card — {details.get('name', 'Model')}",
        "",
        "## General Information",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Type** | {details.get('type', '?')} |",
        f"| **Version** | {details.get('version', '1.0')} |",
        f"| **Date** | {details.get('date', '?')} |",
        f"| **Target** | {details.get('target', '?')} |",
        "",
        "## Intended Use",
        f"- **Primary use** : {use.get('primary_use', '?')}",
        f"- **Out of scope** : {use.get('out_of_scope', '?')}",
        "",
        "## Performance",
        "| Metric | Value |",
        "|--------|-------|",
    ] + [
        f"| {k} | {v} |" for k, v in perf.items()
    ] + [
        "",
        "## Data",
        f"- **Training** : {data.get('n_train', '?')} observations",
        f"- **Test** : {data.get('n_test', '?')} observations",
        f"- **Features** : {data.get('n_features', '?')}",
        f"- **Class imbalance** : {data.get('class_imbalance_ratio', '?')}×",
        "",
        "## Top Features",
    ] + [
        f"- `{k}` : {v}" for k, v in card.get("top_features", {}).items()
    ] + [
        "",
        "## Limitations",
    ] + [
        f"- {lim}" for lim in card.get("limitations", [])
    ] + [
        "",
        "## Ethical Considerations",
        f"- {card.get('ethical_considerations', {}).get('fairness_note', '')}",
        f"- {card.get('ethical_considerations', {}).get('bias_warning', '')}",
        "",
        "*Auto-generated — ml_framework v2.0.0*",
    ]

    return "\n".join(lines)


# =============================================================================
# DISPLAY
# =============================================================================


def print_model_card(card: Dict[str, Any]) -> None:
    """Display the Model Card in a formatted console output."""
    section_header("MODEL CARD")

    details = card.get("model_details", {})
    perf    = card.get("performance", {}).get("metrics_test", {})
    data    = card.get("data_summary", {})

    print(f"  Model    : {details.get('name')} ({details.get('type')} v{details.get('version')})")
    print(f"  Date     : {details.get('date')}")
    print(f"  Target   : {details.get('target')}")
    print()
    print(f"  Data     : {data.get('n_train')} train | {data.get('n_test')} test | {data.get('n_features')} features")
    print()
    print("  Performance:")
    for k, v in perf.items():
        print(f"    {k:<20} : {v}")
    print()
    print("  Top Features:")
    for k, v in list(card.get("top_features", {}).items())[:5]:
        print(f"    {k:<30} : {v:.4f}")
    print()
    print("  Limitations:")
    for lim in card.get("limitations", []):
        print(f"• {lim}")
