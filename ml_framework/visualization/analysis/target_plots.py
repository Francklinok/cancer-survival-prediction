"""
visualization/analysis/target_plots.py
— Visualizations for target variable analysis.

Public functions:
  plot_target_analysis(df, target_col)
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger("ml_framework.visualization.target_plots")


def plot_target_analysis(df: pd.DataFrame, target_col: str) -> None:
    """
    Complete visual analysis of the target variable.
    - Classification: distribution bar + pie, imbalance ratio
    - Regression: histogram + Q-Q plot, skewness
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y      = df[target_col].dropna()
    is_cat = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 15

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if is_cat:
        vc  = y.value_counts()
        pct = vc / len(y) * 100

        vc.plot.bar(ax=axes[0], color= sns.color_palette("Set2", len(vc)), edgecolor="white")
        axes[0].set_title("Class Distribution")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=30)
        for p in axes[0].patches:
            axes[0].annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9,
            )

        _PIE_MAX_SLICES = 6
        if len(vc) <= _PIE_MAX_SLICES:
            axes[1].pie(
                pct, labels=vc.index, autopct="%1.1f%%",
                colors=sns.color_palette("Set2", len(vc)),
                wedgeprops={"edgecolor": "white"},
            )
            axes[1].set_title("Proportions")
        else:
            order = pct.sort_values(ascending=True)
            axes[1].barh(
                order.index.astype(str), order.values,
                color=sns.color_palette("Set2", len(order)), edgecolor="white",
            )
            for i, v in enumerate(order.values):
                axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
            axes[1].set_xlabel("% of observations")
            axes[1].set_title(f"Proportions ({len(vc)} classes)")

        majority_pct = pct.max()
        minority_pct = pct.min()
        ratio        = majority_pct / minority_pct if minority_pct > 0 else float("inf")
        print(f"\n  Majority class   : {vc.idxmax()} ({majority_pct:.1f}%)")
        print(f"  Minority class   : {vc.idxmin()} ({minority_pct:.1f}%)")
        print(f"  Imbalance ratio  : {ratio:.2f}:1")
        imb_lvl = (
            "Balanced"                  if ratio < 1.5
            else "Mild imbalance"       if ratio < 3
            else "Moderate → class_weight recommended" if ratio < 10
            else "Severe → SMOTE / ADASYN required"
        )
        print(f"  Level            : {imb_lvl}")

    else:
        sns.histplot(y, kde=True, ax=axes[0], color="steelblue", edgecolor="white")
        axes[0].axvline(y.mean(),   color="red",   linestyle="--", label=f"μ={y.mean():.2f}")
        axes[0].axvline(y.median(), color="green", linestyle="-",  label=f"Med={y.median():.2f}")
        axes[0].legend()
        axes[0].set_title("Target Distribution")

        stats.probplot(y, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot (normality)")

        sk = stats.skew(y)
        print(f"\n  Mean      : {y.mean():.4f}  |  Median : {y.median():.4f}")
        print(f"  Std dev   : {y.std():.4f}")
        print(f"  Skewness  : {sk:.4f}  {'→ log transformation recommended' if abs(sk) > 1 else ''}")
        print(f"  Kurtosis  : {stats.kurtosis(y):.4f}")

    plt.suptitle(f"Target Variable Analysis: {target_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

