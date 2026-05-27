"""
visualization/analysis/distributions.py
— Plots for numeric and categorical variable distributions.

Public functions:
  plot_dtypes_pie(df, title)
  plot_column_types_pie(col_type_dict, title)
  plot_numeric_distributions(df, columns, n_cols, base_width, base_height)
  plot_categorical_distributions(df, columns, n_cols, figsize_per_plot, rare_threshold)
  plot_boxplots(df, columns, n_cols)
  plot_violins(df, columns, target_col, n_cols)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger("ml_framework.visualization.distributions")


def plot_dtypes_pie(
    df: pd.DataFrame,
    title: str = "Data Types Distribution",
) -> None:
    """
    Pie chart of the DataFrame's dtype distribution.

    Shows the proportion of each dtype (int64, float64, object, …)
    across all columns — quick overview of the dataset's type composition.

    Parameters
    ----------
    df    : DataFrame to inspect
    title : chart title

    Example
    -------
    >>> from ml_framework.visualization.analysis.distributions import plot_dtypes_pie
    >>> plot_dtypes_pie(df)
    """
    dtype_counts = df.dtypes.value_counts()
    if dtype_counts.empty:
        logger.warning("plot_dtypes_pie: empty DataFrame.")
        return

    plt.figure(figsize=(6, 6))
    plt.pie(
        dtype_counts.values,
        labels=dtype_counts.index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(dtype_counts)),
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_column_types_pie(
    data,
    title: str = "Column Type Breakdown",
) -> None:
    """
    Pie chart of column type breakdown (continuous / discrete / categorical /
    binary / ordinal).

    Accepts three input formats:

    1. **pd.Series** — index = labels, values = counts  ← main use-case ::

           labels = ["Continuous", "Discrete", "Categorical", "Binary", "Ordinal"]
           sizes  = [len(result["continuous_cols"]), ...]
           data   = pd.Series(sizes, index=labels)
           plot_column_types_pie(data)

    2. **dict** mapping label → count (int) or label → list of column names ::

           plot_column_types_pie({"Continuous": 12, "Discrete": 5, "Categorical": 3})
           plot_column_types_pie({"continuous": cont_cols, "discrete": disc_cols})

    3. **pd.DataFrame** — auto-classified into continuous / discrete /
       categorical / binary ::

           plot_column_types_pie(df)

    Parameters
    ----------
    data  : pd.Series | dict | pd.DataFrame
    title : chart title
    """
    # ── Normalise input to (labels, values) ──────────────────────────────────
    if isinstance(data, pd.Series):
        # Direct use-case: pd.Series(sizes, index=labels)
        series = data[data > 0]
        if series.empty:
            logger.warning("plot_column_types_pie: all counts are zero.")
            return
        labels = series.index.astype(str).tolist()
        values = series.values.tolist()

    elif isinstance(data, dict):
        counts: dict = {}
        for k, v in data.items():
            # v can be an int/float count OR a list of column names
            counts[k] = int(v) if isinstance(v, (int, float)) else len(v)
        counts = {k: v for k, v in counts.items() if v > 0}
        if not counts:
            logger.warning("plot_column_types_pie: no columns found.")
            return
        labels = list(counts.keys())
        values = list(counts.values())

    elif isinstance(data, pd.DataFrame):
        num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        raw = {
            "Continuous":  [c for c in num_cols if data[c].nunique() > 10],
            "Discrete":    [c for c in num_cols if data[c].nunique() <= 10],
            "Categorical": data.select_dtypes(include=["object", "category"]).columns.tolist(),
            "Binary":      [c for c in data.columns if data[c].nunique() == 2],
        }
        counts = {k: len(v) for k, v in raw.items() if len(v) > 0}
        if not counts:
            logger.warning("plot_column_types_pie: no columns found.")
            return
        labels = list(counts.keys())
        values = list(counts.values())

    else:
        logger.warning("plot_column_types_pie: unsupported data type %s.", type(data))
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors = sns.color_palette("Set3", len(labels))

    plt.figure(figsize=(4, 4))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 3,
    base_width: int = 5,
    base_height: int = 5,
) -> None:
    """
    Histogram + KDE + mean/median lines + stats box for each numeric column.

    Highlights: skewness, outliers, distribution shape, variance.
    """
    if columns is None:
        columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].nunique() > 2
        ]
    if not columns:
        logger.warning("No numeric columns to visualize.")
        return

    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * base_width, n_rows * base_height))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = df[col].dropna()

        sns.histplot(data, kde=True, ax=ax, color="steelblue", edgecolor="white", alpha=0.75)

        mean_v = data.mean()
        med_v  = data.median()
        ax.axvline(mean_v, color="crimson",     linestyle="--", linewidth=1.5, label=f"μ={mean_v:.2f}")
        ax.axvline(med_v,  color="forestgreen", linestyle="-",  linewidth=1.5, label=f"Med={med_v:.2f}")

        sk = stats.skew(data)
        ku = stats.kurtosis(data)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr     = q3 - q1
        n_out   = int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())

        stats_txt = (
            f"σ={data.std():.2f}  CV={data.std()/mean_v*100:.1f}%\n"
            f"Skew={sk:+.2f}  Kurt={ku:.2f}\n"
            f"Q1={q1:.2f} | Q3={q3:.2f}\n"
            f"IQR Outliers: {n_out} ({n_out/len(data)*100:.1f}%)"
        )
        ax.text(
            0.97, 0.97, stats_txt,
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Variable Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (5, 4),
    rare_threshold: float = 0.02,
) -> None:
    """
    Visualize categorical variables.
    - Pie chart for ≤ 5 categories, bar chart otherwise.
    - Highlights dominant and rare categories (< rare_threshold).
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not columns:
        logger.warning("No categorical columns to visualize.")
        return

    n_rows  = (len(columns) + n_cols - 1) // n_cols
    palette = sns.color_palette("Set2", 20)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax  = axes[i]
        vc  = df[col].value_counts(dropna=True)
        pct = vc / len(df)
        n_cat = len(vc)

        rare_cats = pct[pct < rare_threshold].index.tolist()

        if n_cat <= 5:
            colors = [
                "#e74c3c" if v in rare_cats else palette[j]
                for j, v in enumerate(vc.index)
            ]
            vc.plot.pie(
                autopct="%1.1f%%", ax=ax, startangle=90,
                colors=colors,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            ax.set_ylabel("")
        else:
            top   = vc.head(15)
            colors = [
                "#e74c3c" if v in rare_cats else "steelblue"
                for v in top.index
            ]
            sns.barplot(x=top.values, y=top.index, ax=ax, palette=colors)
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_width())}",
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    va="center", ha="left", fontsize=8,
                )
            ax.set_xlabel("Frequency")
            ax.set_ylabel("")

        rare_info = f" | {len(rare_cats)} rare cat." if rare_cats else ""
        ax.set_title(
            f"{col}  ({n_cat} cat.){rare_info}",
            fontsize=10, fontweight="bold"
        )

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Variable Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 2,
) -> None:
    """Multi-column boxplots with IQR outlier count annotations."""
    if columns is None:
        columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].nunique() > 2
        ]
    if not columns:
        return

    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3 + 1))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        data = df[col].dropna()
        sns.boxplot(
            x=data, ax=axes[i],
            color="lightsteelblue",
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 4},
        )
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr    = q3 - q1
        n_out  = int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())
        color  = "red" if n_out > len(data) * 0.05 else "black"
        axes[i].set_title(
            f"{col}  — {n_out} IQR outliers ({n_out/len(data)*100:.1f}%)",
            fontsize=10, fontweight="bold", color=color,
        )

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots — Visual Outlier Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_violins(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    n_cols: int = 2,
) -> None:
    """
    Violin plots for numeric variables.
    If target_col is provided, decomposes by class (violin × target).

    Highlights: distribution shape, multimodality, inter-class differences.
    """
    if columns is None:
        columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col and df[c].nunique() > 2
        ]
    if not columns:
        return

    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        if target_col and target_col in df.columns:
            sns.violinplot(
                x=target_col, y=col, data=df, ax=ax,
                palette="Set2", inner="quartile",
            )
            ax.set_title(f"{col} by {target_col}", fontsize=10, fontweight="bold")
            ax.tick_params(axis="x", rotation=20)
        else:
            sns.violinplot(y=df[col].dropna(), ax=ax, color="mediumpurple", inner="quartile")
            ax.set_title(col, fontsize=10, fontweight="bold")

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    title = (
        f"Violin Plots by Class ({target_col})" if target_col
        else "Violin Plots — Distribution Shape"
    )
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
