"""
pipeline_utils.py — Pipeline utilities (state display, validation, summary).

Helper module for inspecting and displaying the intermediate states
of the MedicalMLPipeline:
  - log_step            : formatted pipeline step header
  - show_state_table    : display a pipeline state element as a table
  - validate_data       : basic dataset validation
  - get_results_summary : structured pipeline results summary
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("ml_framework.pipeline_utils")


# =============================================================================
# STEP LOGGING
# =============================================================================


def log_step(
    step_name: str,
    execution_log: Optional[List[Dict]] = None,
    width: int = 55,
) -> None:
    """
    Print a formatted header for a pipeline step and record it in the log.

    Parameters
    ----------
    step_name     : name of the pipeline step
    execution_log : log list to append to (optional)
    width         : separator line width
    """
    print(f"\n{'─' * width}")
    print(f"  {step_name}")
    print(f"{'─' * width}")

    if execution_log is not None:
        execution_log.append({
            "step":      step_name,
            "timestamp": pd.Timestamp.now(),
        })


# =============================================================================
# STATE DISPLAY
# =============================================================================


def show_state_table(
    state: Dict[str, Any],
    table_name: str,
    n: int = 5,
) -> None:
    """
    Display a pipeline state element as a table.

    Supports:
      - pd.DataFrame        → head(n)
      - dict of DataFrames  → head(n) of each value
      - simple dict         → key/value table
      - tuple of DataFrames → head(n) of each element
      - other               → repr()

    Parameters
    ----------
    state      : pipeline state dict (self.state)
    table_name : key in state to display
    n          : number of rows to show
    """
    if table_name not in state:
        print(f"  '{table_name}' not found in state.")
        return

    obj = state[table_name]
    print(f"\n  {'═' * 50}")
    print(f"  {table_name.upper()}")
    print(f"  {'═' * 50}")

    if isinstance(obj, pd.DataFrame):
        print(obj.head(n).to_string())

    elif isinstance(obj, dict):
        if all(isinstance(v, pd.DataFrame) for v in obj.values()):
            for k, df in obj.items():
                print(f"\n  {k}")
                print(df.head(n).to_string())
        else:
            display_df = pd.DataFrame(
                [(k, v) for k, v in obj.items()],
                columns=["Key", "Value"],
            )
            print(display_df.to_string(index=False))

    elif isinstance(obj, tuple):
        for i, item in enumerate(obj):
            print(f"\n  [{i}]")
            if isinstance(item, pd.DataFrame):
                print(item.head(n).to_string())
            else:
                print(repr(item))

    elif isinstance(obj, (list, pd.Series)):
        print(pd.Series(obj).head(n).to_string())

    else:
        print(repr(obj))


# =============================================================================
# DATA VALIDATION
# =============================================================================


def validate_data(
    state: Dict[str, Any],
    target_column: str,
    max_missing_pct: float = 50.0,
) -> None:
    """
    Validate the dataset loaded in the pipeline state.

    Checks:
      - Dataset is not empty
      - Target column is present
      - Global missing value rate is within threshold

    Parameters
    ----------
    state           : pipeline state dict
    target_column   : expected target column name
    max_missing_pct : missing value alert threshold (%)

    Raises
    ------
    ValueError if dataset is empty or target column is missing.
    """
    data = state.get("data_original") or state.get("df_work") or state.get("df_raw")

    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        raise ValueError("Empty dataset — load data before validation.")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    missing_pct = 100.0 * data.isnull().sum().sum() / (data.size or 1)

    print(f"  Dataset validation: {data.shape[0]} observations × {data.shape[1]} columns")
    print(f"  Missing values: {missing_pct:.2f}%")

    if missing_pct > max_missing_pct:
        logger.warning(
            "High missing value rate: %.1f%% (threshold: %.1f%%)",
            missing_pct, max_missing_pct,
        )
        print(f" Warning: {missing_pct:.1f}% missing values")


# =============================================================================
# RESULTS SUMMARY
# =============================================================================


def get_results_summary(
    state: Dict[str, Any],
    execution_log: Optional[List[Dict]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Generate a structured summary of pipeline results.

    Parameters
    ----------
    state         : pipeline state dict (self.state)
    execution_log : list of execution log entries (optional)

    Returns
    -------
    dict or str if the pipeline has not been run
    """
    if not state.get("evaluation_results") and not state.get("final_metrics"):
        return "Pipeline not executed or evaluation unavailable."

    best_model   = state.get("best_model", "N/A")
    final_ds     = state.get("final_dataset")
    feat_count   = (final_ds.shape[1] - 1) if isinstance(final_ds, pd.DataFrame) else 0
    final_metrics = state.get("final_metrics", {})
    eval_results  = state.get("evaluation_results", {})

    model_scores = {}
    for name, res in eval_results.items():
        if isinstance(res, dict):
            score = res.get("cv_mean") or res.get("test_score") or res.get("roc_auc")
            if score is not None:
                model_scores[name] = round(float(score), 4)

    summary = {
        "best_model":         best_model,
        "dataset_shape":      tuple(final_ds.shape) if isinstance(final_ds, pd.DataFrame) else None,
        "features_selected":  feat_count,
        "final_metrics":      {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in final_metrics.items()},
        "model_scores":       model_scores,
        "n_models_evaluated": len(eval_results),
    }

    if execution_log:
        summary["execution_steps"] = [e["step"] for e in execution_log]

    return summary
