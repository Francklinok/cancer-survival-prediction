"""
Track model performance over time and save the results to disk.

This module records a few standard metrics for a trained model, stores them
in a simple history file, and warns when the model seems to be drifting.
The plotting logic lives in the monitoring visualization module.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_framework.visualization.monitoring.performance_plots import plot_performance_history

logger = logging.getLogger("ml_framework.monitoring")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRACKING
# ──────────────────────────────────────────────────────────────────────────────


def track_model_performance_over_time(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_id: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    storage_path: Optional[str] = None,
    alert_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate a model on test data and keep a running history of its scores.

    Args:
        model: A trained model object.
        X_test: Test features.
        y_test: Ground-truth labels for the test set.
        model_id: Optional name for the model run. A timestamp-based name is used if not provided.
        metrics: Metrics to compute. If omitted, a small default set is used.
        storage_path: Folder where the history CSV file should be saved.
        alert_threshold: Minimum drop that should be treated as a notable change.
        verbose: Whether to print the current results to the console.

    Returns:
        A dataframe containing the recorded performance history.
    """
    if model_id is None:
        model_id = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            metrics.append("roc_auc")

    y_pred      = model.predict(X_test)
    performance = _compute_performance(model, X_test, y_test, y_pred, metrics, model_id)

    if verbose:
        _print_performance(performance)

    history_df = _save_history(performance, model_id, storage_path)

    if len(history_df) > 1:
        plot_performance_history(history_df, metrics)
        _detect_performance_drift(history_df, metrics, alert_threshold)

    return history_df


# ──────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────


def _compute_performance(
    model, X_test, y_test, y_pred, metrics: List[str], model_id: str
) -> Dict:
    """Compute the requested metrics for one model run."""
    perf = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id":  model_id,
    }

    for metric in metrics:
        try:
            if metric == "accuracy":
                perf[metric] = float(accuracy_score(y_test, y_pred))
            elif metric == "precision":
                perf[metric] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            elif metric == "recall":
                perf[metric] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            elif metric == "f1":
                perf[metric] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            elif metric == "roc_auc" and hasattr(model, "predict_proba"):
                y_proba       = model.predict_proba(X_test)[:, 1]
                perf[metric]  = float(roc_auc_score(y_test, y_proba))
        except Exception as exc:
            logger.warning("Metric '%s' could not be computed: %s", metric, exc)

    return perf


# ──────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────────


def _save_history(
    performance: Dict, model_id: str, storage_path: Optional[str]
) -> pd.DataFrame:
    """Append the new result to the history file and return the full history."""
    if not storage_path:
        return pd.DataFrame([performance])

    os.makedirs(storage_path, exist_ok=True)
    history_file = os.path.join(storage_path, f"{model_id}_history.csv")

    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat(
            [history_df, pd.DataFrame([performance])], ignore_index=True
        )
    else:
        history_df = pd.DataFrame([performance])

    history_df.to_csv(history_file, index=False)
    logger.info("History saved: %s", history_file)
    return history_df


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────────────────────────


def _print_performance(perf: Dict) -> None:
    """Print the latest performance values in a simple format."""
    print(f"\n  Model '{perf['model_id']}' performance ({perf['timestamp']}):")
    for k, v in perf.items():
        if k not in ("timestamp", "model_id"):
            print(f"    {k:<20} : {v:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTION
# ──────────────────────────────────────────────────────────────────────────────


def _detect_performance_drift(
    history_df: pd.DataFrame,
    metrics: List[str],
    threshold: float,
) -> None:
    """Check whether recent scores show a clear drop or a downward trend."""
    print("\n  ─── Performance Drift Analysis ───")
    has_alert = False

    # Gather p-values so we can adjust them across several metrics.
    trend_pvals: List[float] = []
    trend_meta: List[dict]   = []

    for metric in metrics:
        if metric not in history_df.columns:
            continue
        vals = history_df[metric].dropna().values
        if len(vals) < 5:
            if len(vals) >= 2:
                # Warn that n < 5 has no statistical power for trend detection
                print(f"  {metric}: only {len(vals)} observations — "
                      f"too few for Mann-Kendall (need ≥ 5). Sudden-drop check only.")
            continue

        # Check whether scores are moving up or down over time.
        try:
            from scipy.stats import kendalltau
            tau, p_raw = kendalltau(np.arange(len(vals)), vals)
            trend_pvals.append(p_raw)
            trend_meta.append({"metric": metric, "tau": tau, "vals": vals})
        except Exception as exc:
            logger.warning("Mann-Kendall failed for '%s': %s", metric, exc)

    # Adjust the p-values to account for testing several metrics.
    n_tests = len(trend_pvals)
    for i, (p_raw, meta) in enumerate(zip(trend_pvals, trend_meta)):
        p_adj     = min(p_raw * n_tests, 1.0)  # Bonferroni
        metric    = meta["metric"]
        tau       = meta["tau"]
        vals      = meta["vals"]
        alpha_adj = 0.05

        if p_adj < alpha_adj and tau < 0:
            print(f"  {metric}: significant declining trend "
                  f"(Mann-Kendall τ={tau:.3f}, p_adj={p_adj:.4f}, Bonferroni n={n_tests}).")
            has_alert = True
        elif p_adj < alpha_adj and tau > 0:
            print(f" {metric}: significant improving trend "
                  f"(Mann-Kendall τ={tau:.3f}, p_adj={p_adj:.4f}).")

        # Also look for a sharp drop, even when there are too few points for trend analysis.
        diffs = np.diff(vals)
        if any(abs(d) > threshold for d in diffs[-3:]):
            print(f"{metric}: abrupt change detected (Δ > {threshold}).")
            has_alert = True

    # For short histories, only the sudden-drop check is reliable.
    for metric in metrics:
        if metric not in history_df.columns:
            continue
        vals = history_df[metric].dropna().values
        if len(vals) < 5 and len(vals) >= 2:
            diffs = np.diff(vals)
            if any(abs(d) > threshold for d in diffs[-3:]):
                print(f" {metric}: abrupt change detected (Δ > {threshold}).")
                has_alert = True

    if not has_alert:
        print("No significant performance drift detected.")
