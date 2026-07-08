"""
visualization/evaluation/model_plots.py
— Core model evaluation plots.

Public functions:
  plot_confusion_matrix(y_test, y_pred, class_names)
  plot_roc_curve(y_test, y_proba, auc_score)
  plot_precision_recall(y_test, y_proba)
  plot_calibration(model, X_test, y_test, n_bins)
  plot_score_distribution(y_proba, y_test)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("ml_framework.visualization.model_plots")


def plot_confusion_matrix(
    y_test,
    y_pred,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Dual confusion matrix: raw counts + row-normalized proportions.

    Parameters
    ----------
    y_test      : true labels
    y_pred      : predicted labels
    class_names : optional list of class label strings
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = class_names or list(np.unique(y_test))
   
    row_sums = cm.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        logger.warning(
            "plot_confusion_matrix: %d class(es) have zero support in y_test — "
            "their normalized row will show as 0 rather than NaN.",
            int(np.sum(row_sums == 0)),
        )
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float), where=row_sums != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (counts)")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (normalized)")

    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    if len(np.unique(y_test)) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"  Sensitivity (Recall)  : {sens:.4f}")
        print(f"  Specificity           : {spec:.4f}")
        print(f"  PPV (Precision)       : {ppv:.4f}")
        print(f"  NPV                   : {npv:.4f}")


def plot_roc_curve(y_test, y_proba, auc_score: float = 0.0) -> None:
    """
    ROC curve with AUC annotation and optimal threshold marker.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    opt_idx = int(np.argmax(tpr - fpr))

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.plot(fpr[opt_idx], tpr[opt_idx], "ro", markersize=10,
             label=f"Optimal threshold = {thresholds[opt_idx]:.3f}")
    plt.fill_between(fpr, tpr, alpha=0.07, color="darkorange")
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_precision_recall(y_test, y_proba) -> None:
    """
    Precision-Recall curve with Average Precision annotation.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap       = average_precision_score(y_test, y_proba)
    baseline = float(pd.Series(y_test).mean()) if hasattr(y_test, "__len__") else 0.5

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where="post", color="steelblue", lw=2,
             label=f"AP = {ap:.4f}")
    plt.fill_between(recall, precision, step="post", alpha=0.12, color="steelblue")
    plt.axhline(baseline, color="gray", linestyle="--", lw=1.5, label=f"Baseline = {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_calibration(model, X_test, y_test, n_bins: int = 10) -> None:
    """
    Calibration curve: predicted probability vs observed proportion.
    """
    try:
        prob_true, prob_pred = calibration_curve(
            y_test, model.predict_proba(X_test)[:, 1], n_bins=n_bins
        )
        plt.figure(figsize=(7, 6))
        plt.plot(prob_pred, prob_true, marker="o", color="steelblue", lw=2, label="Model")
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed proportion")
        plt.title("Calibration Curve", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        logger.warning("Calibration plot failed: %s", exc)


def plot_score_distribution(y_proba: np.ndarray, y_test) -> None:
    """
    Histogram of prediction scores separated by true class.
    """
    df_vis = pd.DataFrame({"score": y_proba, "class": y_test})
    plt.figure(figsize=(9, 5))
   .
    bin_edges = np.histogram_bin_edges(df_vis["score"], bins=30)
    for cls in sorted(df_vis["class"].unique()):
        subset = df_vis[df_vis["class"] == cls]["score"]
        plt.hist(subset, bins=bin_edges, alpha=0.5, label=f"Class {cls}", edgecolor="white")
    plt.xlabel("Prediction score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution by Class", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
