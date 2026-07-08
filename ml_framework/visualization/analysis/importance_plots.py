"""
visualization/analysis/importance_plots.py
— Feature importance exploration charts (pre-modeling).

Public functions:
  plot_combined_importance(imp_df, target_col)
  plot_business_insights_bars(df, strong_cats, target_col)
  plot_leakage_risk(leak_df)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.importance_plots")


def plot_combined_importance(imp_df: pd.DataFrame, target_col: str) -> None:
    """
    Two-panel chart: Borda score bar (left) + per-method rank heatmap (right).

    Compatible with both the new (borda_score / *_rank) and legacy
    (combined_score / pearson / cramers_v) output formats of
    ``feature_importance_exploration()``.

    Parameters
    ----------
    imp_df     : DataFrame from feature_importance_exploration()
    target_col : target column name (for title)
    """
    if imp_df.empty:
        return

    # ── Score column: support both new and legacy format ─────────────────────
    if "borda_score" in imp_df.columns:
        score_col  = "borda_score"
        score_label = "Borda Score — mean rank (assoc | MI | RF)"
    else:
        score_col  = "combined_score"
        score_label = "Combined Score"

    # ── Detail columns: rank cols preferred, fall back to raw ────────────────
    rank_cols = [c for c in ["assoc_rank", "mi_rank", "rf_rank"] if c in imp_df.columns]
    raw_cols  = [c for c in ["pearson", "cramers_v", "mutual_info", "rf_importance"]
                 if c in imp_df.columns]
    detail_cols = rank_cols if rank_cols else raw_cols

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(imp_df) * 0.5 + 1.2)))

    # Left panel — Borda / combined score bar
    axes[0].barh(
        imp_df.index[::-1], imp_df[score_col][::-1],
        color="steelblue", edgecolor="white",
    )
    axes[0].set_title(score_label, fontsize=10)
    axes[0].set_xlabel("Score (0 = least, 1 = most important)")
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(axis="x", alpha=0.3)

    # Right panel — per-metric rank heatmap
    if detail_cols:
        col_labels = (
            ["Assoc rank", "MI rank", "RF rank"] if rank_cols
            else detail_cols
        )
        detail_data = imp_df[detail_cols].rename(columns=dict(zip(detail_cols, col_labels))).round(3)
        if rank_cols:
            # Rank columns are already normalised to [0, 1] by construction.
            heatmap_kwargs = dict(cmap="Blues", vmin=0, vmax=1)
        else:
            
            has_negative = bool((detail_data < 0).to_numpy().any())
            if has_negative:
                bound = float(detail_data.abs().to_numpy().max() or 1.0)
                heatmap_kwargs = dict(cmap="RdBu_r", vmin=-bound, vmax=bound, center=0)
            else:
                heatmap_kwargs = dict(cmap="Blues", vmin=0, vmax=float(detail_data.to_numpy().max() or 1.0))
        sns.heatmap(
            detail_data,
            annot=True, fmt=".2f", ax=axes[1],
            linewidths=0.5, **heatmap_kwargs,
        )
        axes[1].set_title(
            "Per-metric rank (0=lowest, 1=highest)" if rank_cols
            else "Score detail by method"
        )

    plt.suptitle(
        f"Exploratory Feature Importance → {target_col}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_business_insights_bars(
    df: pd.DataFrame,
    strong_cats: List[Tuple[str, float]],
    target_col: str,
) -> None:
    """
    Stacked bar charts for top categorical associations with target.

    Parameters
    ----------
    df          : source DataFrame
    strong_cats : list of (col_name, cramers_v) tuples, sorted descending
    target_col  : target column name
    """
    if not strong_cats:
        return

    fig, axes = plt.subplots(1, len(strong_cats), figsize=(len(strong_cats) * 5, 4))
    if len(strong_cats) == 1:
        axes = [axes]

    for ax, (col, cv) in zip(axes, strong_cats):
        ct = pd.crosstab(df[col], df[target_col], normalize="index")
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
        ax.set_title(f"{col}  (V={cv:.3f})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=7)

    plt.suptitle(f"Top Categorical Associations → {target_col}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_leakage_risk(leak_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of leakage **suspicion scores**.

    Bar length = numeric suspicion score (sum of signal weights).
    Bar color   = risk tier: HIGH (red) | MEDIUM (orange) | LOW (blue).

    Parameters
    ----------
    leak_df : DataFrame returned by ``leakage_exploration()``.
              Expected columns: score, risk  (indexed by column name).
    """
    if leak_df.empty:
        return

    if "score" in leak_df.columns:
        values = leak_df["score"]
        xlabel = "Suspicion score (sum of signal weights)"
    else:
        values = pd.Series([1] * len(leak_df), index=leak_df.index)
        xlabel = ""

    risk_colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#3498db"}
    colors = [risk_colors.get(r, "#95a5a6") for r in leak_df["risk"]]

    # Sort by score descending for display
    order   = values.sort_values(ascending=True).index
    vals    = values.reindex(order)
    cols_   = [risk_colors.get(leak_df.loc[c, "risk"], "#95a5a6") for c in order]

    fig, ax = plt.subplots(figsize=(9, max(3, len(leak_df) * 0.55 + 1.2)))
    bars = ax.barh(range(len(order)), vals.values, color=cols_, edgecolor="white", height=0.6)

    # Annotate score + risk tier on each bar
    for i, (bar, col_name) in enumerate(zip(bars, order)):
        tier  = leak_df.loc[col_name, "risk"]
        score = vals.iloc[i]
        ax.text(
            score + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{tier}  ({score})",
            va="center", ha="left", fontsize=8, color="black",
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(0, vals.max() * 1.45)
    ax.set_title("Leakage Suspicion Score by Column", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="grey", linewidth=0.5)

    patches = [
        mpatches.Patch(color="#e74c3c", label="HIGH  — multiple signals"),
        mpatches.Patch(color="#f39c12", label="MEDIUM — one strong signal"),
        mpatches.Patch(color="#3498db", label="LOW   — keyword only"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.set_title(
        "Leakage Suspicion Score by Column\n"
        "(heuristic — validate with domain knowledge)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
