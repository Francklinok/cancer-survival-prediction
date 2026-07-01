"""
documentation.py — Automatic model documentation generation.

Produces complete technical and clinical documentation for an ML model:
  - Metadata (type, version, date, audience)
  - Input feature descriptions
  - Training data descriptive statistics
  - Hyperparameters
  - Performance metrics
  - Usage guide with code example
  - Limitations and usage precautions
  - Maintenance and update section

Public functions:
  - create_model_documentation(model, X, y, ...) → str (Markdown)
  - save_documentation(doc_str, path, filename)  → str (file path)
  - print_documentation_summary(model, metrics)  → None
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sklearn
import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.documentation")


# =============================================================================
# DOCUMENTATION GENERATION
# =============================================================================


def create_model_documentation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]] = None,
    model_name: str = "Predictive Model",
    target_description: str = "target variable",
    model_purpose: str = "Medical decision support",
    audience: str = "Clinicians and medical teams",
    limitations: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    version: str = "1.0.0",
) -> str:
    """
    Generate complete model documentation in Markdown format.

    Parameters
    ----------
    model              : trained sklearn estimator
    X                  : training data (features)
    y                  : training data (target)
    feature_names      : feature names (default: X.columns)
    model_name         : name of the model
    target_description : description of the target variable
    model_purpose      : purpose of the model
    audience           : intended audience for the documentation
    limitations        : list of known limitations
    metrics            : performance metrics dict {metric: value}
    version            : model version string

    Returns
    -------
    str — documentation formatted as Markdown
    """
    print("\n  Generating model documentation...")

    if feature_names is None and hasattr(X, "columns"):
        feature_names = X.columns.tolist()

    model_type = model.__class__.__name__

    try:
        hyperparams = model.get_params()
        skip_keys = {"verbose", "silent", "nthread", "callbacks"}
        hyperparams = {k: v for k, v in hyperparams.items() if k not in skip_keys}
    except Exception:
        hyperparams = {"Not available": "This model does not expose its hyperparameters"}

    importance_section = ""
    if feature_names:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            top = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:10]
            importance_section = "\n## Top 10 Features by Importance\n\n"
            importance_section += "| Feature | Importance |\n|---------|------------|\n"
            for fname, fval in top:
                importance_section += f"| `{fname}` | {fval:.4f} |\n"
        elif hasattr(model, "coef_"):
            coef = model.coef_
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).flatten()
            if len(imp) == len(feature_names):
                top = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:10]
                importance_section = "\n## Top 10 Coefficients (absolute value)\n\n"
                importance_section += "| Feature | |Coefficient| |\n|---------|-------------|\n"
                for fname, fval in top:
                    importance_section += f"| `{fname}` | {fval:.4f} |\n"

    metrics_section = ""
    if metrics:
        metrics_section = "\n## Performance Metrics\n\n"
        metrics_section += "| Metric | Value |\n|--------|-------|\n"
        for k, v in metrics.items():
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            metrics_section += f"| {k} | {v_str} |\n"

    data_stats = ""
    if hasattr(X, "describe") and not X.empty:
        desc = X.describe().T
        preview = desc.head(5).to_markdown() if hasattr(desc, "to_markdown") else str(desc.head(5))
        data_stats = f"\n### Descriptive Statistics (first 5 features)\n\n```\n{preview}\n```\n"

    target_dist = ""
    if hasattr(y, "value_counts"):
        vc = y.value_counts()
        target_dist = "\n### Target Variable Distribution\n\n```\n" + str(vc) + "\n```\n"

    if limitations is None:
        limitations = [
            "This model is a decision-support tool — it does not replace clinical judgment.",
            "Performance may vary for sub-populations absent from the training data.",
            "The model must be periodically re-evaluated (data drift, new cohorts).",
            "Predictions must be interpreted in a complete clinical context.",
        ]

    limitations_md = "\n".join(f"- {lim}" for lim in limitations)

    params_md = ""
    shown = list(hyperparams.items())[:10]
    for k, v in shown:
        params_md += f"- `{k}`: {v}\n"
    if len(hyperparams) > 10:
        params_md += f"- *... and {len(hyperparams) - 10} additional parameters*\n"

    features_md = ""
    if feature_names:
        shown_feats = feature_names[:15]
        for i, f in enumerate(shown_feats, 1):
            features_md += f"{i}. `{f}`\n"
        if len(feature_names) > 15:
            features_md += f"... and {len(feature_names) - 15} additional features\n"

    doc = f"""# {model_name}

## General Information

| Field | Value |
|-------|-------|
| **Model name** | {model_name} |
| **Type** | {model_type} |
| **Version** | {version} |
| **Creation date** | {datetime.now().strftime("%Y-%m-%d")} |
| **ML framework** | scikit-learn {sklearn.__version__} |
| **Purpose** | {model_purpose} |
| **Intended audience** | {audience} |

## Description

This model was developed for the prediction of **{target_description}**.
It uses a **{model_type}** algorithm analyzing {len(feature_names) if feature_names else "multiple"} features
to generate predictions with associated probabilities.

## Input Features

{features_md or "_No features documented_"}

## Training Data
{data_stats}
{target_dist}

## Hyperparameters

{params_md or "_Not available_"}
{metrics_section}
{importance_section}

## Limitations and Usage Precautions

{limitations_md}

## Usage Guide

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load("path/to/{model_type.lower()}.pkl")

# Prepare the data
X_new = pd.DataFrame({{
    # feature_1: value_1,
    # feature_2: value_2,
    # ...
}})

# Predict
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)  # per-class probabilities

print(f"Prediction: {{predictions}}")
print(f"Risk probability: {{probabilities[:, 1]:.2f}}")
```

## Maintenance and Updates

It is recommended to periodically re-evaluate model performance, especially if:

- The input data distribution changes significantly (data drift)
- New cohorts or populations become available
- Performance degrades over time (concept drift)
- New clinically relevant factors are identified

**Recommended re-evaluation frequency**: every 6 months, or at each major clinical protocol update.

---
*Auto-generated on {datetime.now().strftime("%Y-%m-%d at %H:%M")} — ml_framework v2.0.0*
"""

    print("Documentation generated successfully.")
    return doc


# =============================================================================
# DOCUMENTATION SAVING
# =============================================================================


def save_documentation(
    doc_str: str,
    path: Union[str, Path] = "./reports/",
    filename: str = "model_documentation",
) -> str:
    """
    Save the documentation as a Markdown file.

    Parameters
    ----------
    doc_str  : Markdown documentation string
    path     : destination directory
    filename : file name (without extension)

    Returns
    -------
    path of the created file
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    fp = save_dir / f"{filename}.md"

    fp.write_text(doc_str, encoding="utf-8")
    logger.info("Documentation saved → %s", fp)
    print(f"Documentation saved: {fp}")
    return str(fp)


# =============================================================================
# PRINTED SUMMARY
# =============================================================================


def print_documentation_summary(
    model,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a condensed summary of the model documentation.

    Parameters
    ----------
    model   : trained sklearn estimator
    metrics : performance metrics dict
    """
    print("\n" + "═" * 58)
    print("  MODEL SUMMARY")
    print("═" * 58)
    print(f"  Type           : {model.__class__.__name__}")

    try:
        params = model.get_params()
        n = min(4, len(params))
        items = list(params.items())[:n]
        for k, v in items:
            print(f"  {k:<20} : {v}")
        if len(params) > n:
            print(f"  ... ({len(params) - n} additional parameters)")
    except Exception:
        pass

    if metrics:
        print("\n  Metrics:")
        for k, v in list(metrics.items())[:8]:
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k:<25} : {v_str}")

    print("═" * 58)
