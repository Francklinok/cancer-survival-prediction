"""
eda.py — Exploratory Data Analysis (EDA) — PURELY VISUAL & DESCRIPTIVE.

Principle
---------
This module contains ONLY visual and descriptive exploration.
No hypothesis tests, no p-values, no statistical decisions.
All formal statistical testing lives in statistical_analysis.py.

Structure
---------
  1.  Numeric distributions          → plot_distributions()
  2.  Categorical distributions      → explore_categorical_distributions()
  3.  Boxplots                        → box_plot()
  4.  Violin plots                    → violin_plot()
  5.  Target variable analysis        → analyze_target()
  6.  Bivariate visual analysis       → bivariate_analysis()
  7.  Pattern & anomaly detection     → pattern_anomaly_analysis()
  8.  Group effect visualization      → group_effect_analysis()
  9.  Residual anomaly detection      → residual_anomaly_analysis()
  10. Full orchestrator               → run_eda()

Note
----
Correlation matrix + VIF         → correlation_matrix.py
Normality + formal tests          → statistical_analysis.py
Class imbalance                   → class_imbalance.py
Feature→target diagnostics        → diagnostic_analysis.py
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# ── Visualization layer ───────────────────────────────────────────────────────
from ml_framework.visualization.analysis.distributions import (
    plot_numeric_distributions,
    plot_categorical_distributions,
    plot_boxplots,
    plot_violins,
)
from ml_framework.visualization.analysis.target_plots import plot_target_analysis
from ml_framework.visualization.analysis.bivariate_plots import (
    plot_numeric_vs_numeric,
    plot_numeric_vs_categorical,
    plot_categorical_vs_categorical,
)

from ml_framework.utils.display_utils import section_header
from ml_framework.utils.statistical_utils import cramers_v_from_series
from ml_framework.utils.column_utils import get_numeric_columns, get_categorical_columns

logger = logging.getLogger("ml_framework.eda")


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS (thin wrappers kept for backward compat within module)
# ──────────────────────────────────────────────────────────────────────────────

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    return cramers_v_from_series(x, y)


def _num_cols(df: pd.DataFrame, exclude: Optional[str] = None) -> List[str]:
    return get_numeric_columns(df, exclude=[exclude] if exclude else None)


def _cat_cols(df: pd.DataFrame, exclude: Optional[str] = None) -> List[str]:
    return get_categorical_columns(df, exclude=[exclude] if exclude else None)

# ──────────────────────────────────────────────────────────────────────────────
# 1. NUMERIC DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 3,
    base_width: int = 5,
    base_height: int = 4,
) -> None:
    """
    Histogram + KDE + mean/median lines for each numeric column.
    Includes a descriptive text summary (skew direction, variance level).
    No statistical tests — purely visual + descriptive labels.
    """
    if columns is None:
        columns = _num_cols(df)
    plot_numeric_distributions(df, columns, n_cols, base_width, base_height)
    _describe_distributions(df, columns)


def _describe_distributions(df: pd.DataFrame, columns: List[str]) -> None:
    """Descriptive (non-inferential) summary of each numeric distribution."""
    section_header("DISTRIBUTION DESCRIPTION")
    for col in columns:
        data = df[col].dropna()
        sk = float(stats.skew(data))
        ku = float(stats.kurtosis(data))
        cv = data.std() / data.mean() * 100 if data.mean() != 0 else 0

        skew_txt = (
            "symmetric"
            if abs(sk) < 0.5
            else f"{'right-skewed ↗' if sk > 0 else 'left-skewed ↙'} (moderate)"
            if abs(sk) < 1
            else f"strongly {'right' if sk > 0 else 'left'}-skewed → transformation likely needed"
        )
        kurt_txt = (
            "bell-shaped tails"
            if abs(ku) < 1
            else "heavy tails → outliers likely"
            if ku > 1
            else "light tails (flat distribution)"
        )
        var_txt = (
            "low spread" if cv < 15
            else "moderate spread" if cv < 50
            else "high spread → scaling recommended"
        )
        print(f"  • {col:<28} skew={sk:+.2f}  {skew_txt}")
        print(f"    {'':28} kurt={ku:+.2f}  {kurt_txt}")
        print(f"    {'':28} CV={cv:.1f}%    {var_txt}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. CATEGORICAL DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────

def explore_categorical_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (5, 4),
    rare_threshold: float = 0.02,
) -> None:
    """
    Bar charts + dominant category + rare category flags.
    Purely visual — no chi-square, no p-values.
    """
    section_header("CATEGORICAL DISTRIBUTIONS")
    if columns is None:
        columns = _cat_cols(df)

    for col in (columns or []):
        vc  = df[col].value_counts(dropna=True)
        pct = vc / len(df)
        rare_cats    = pct[pct < rare_threshold].index.tolist()
        dominant     = vc.idxmax()
        dominant_pct = pct.max() * 100
        print(
            f"  • {col:<30} dominant='{dominant}' ({dominant_pct:.1f}%)"
            + (f"  rare categories: {rare_cats[:3]}" if rare_cats else "")
        )

    plot_categorical_distributions(df, columns, n_cols, figsize_per_plot, rare_threshold)


# ──────────────────────────────────────────────────────────────────────────────
# 3. BOXPLOTS
# ──────────────────────────────────────────────────────────────────────────────

def box_plot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 2,
) -> None:
    """Boxplots for all numeric columns — visual outlier detection."""
    if columns is None:
        columns = _num_cols(df)
    plot_boxplots(df, columns, n_cols)


# ──────────────────────────────────────────────────────────────────────────────
# 4. VIOLIN PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def violin_plot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    n_cols: int = 2,
) -> None:
    """
    Violin plots — distribution shape + spread per group if target provided.
    Reveals bimodality, skew, and between-group visual differences.
    """
    if columns is None:
        columns = _num_cols(df, exclude=target_col)
    plot_violins(df, columns, target_col, n_cols)


# ──────────────────────────────────────────────────────────────────────────────
# 5. TARGET VARIABLE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_target(df: pd.DataFrame, target_col: str) -> None:
    """
    Visual analysis of the target variable:
    class balance bar chart, pie chart, value counts.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    section_header(f"TARGET VARIABLE ANALYSIS — {target_col}")
    plot_target_analysis(df, target_col)


# ──────────────────────────────────────────────────────────────────────────────
# 6. BIVARIATE VISUAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def bivariate_analysis(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
) -> None:
    """
    Visual bivariate exploration — feature × target (3 cases).

    6a. Num × Num target   → scatter plots with trend line
    6b. Num × Cat target   → boxplots by class (visual spread comparison)
    6c. Cat × Cat target   → crosstab heatmaps + Cramér's V ranking (visual only)

    Cramér's V is used here purely to RANK and SELECT which pairs to plot —
    it is NOT presented as a formal test result. Formal chi-square testing
    belongs in statistical_analysis.categorical_association_tests().
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")

    target_is_cat = (
        not pd.api.types.is_numeric_dtype(df[target_col])
        or df[target_col].nunique() <= 15
    )

    num_cols = _num_cols(df, exclude=target_col)[:top_n]
    cat_cols = _cat_cols(df, exclude=target_col)[:top_n]

    # 6a. Num × Num
    if num_cols and pd.api.types.is_numeric_dtype(df[target_col]):
        section_header("6a. BIVARIATE — Numerics × Numeric Target")
        plot_numeric_vs_numeric(df, num_cols, target_col)

    # 6b. Num × Categorical target
    if num_cols and target_is_cat:
        section_header("6b. BIVARIATE — Numerics × Categorical Target")
        plot_numeric_vs_categorical(df, num_cols, target_col)

    # 6c. Cat × Cat — rank by Cramér's V to choose which to plot, no formal test output
    if cat_cols and target_is_cat:
        section_header("6c. BIVARIATE — Categoricals × Categorical Target")
        ranked = []
        for col in cat_cols:
            try:
                v = _cramers_v(df[col].dropna(), df[target_col].dropna())
                ranked.append((col, v))
            except Exception:
                pass
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_cats = [c for c, _ in ranked[:6]]
        print("  Visual ranking by association strength (Cramér's V — for plot selection only):")
        for col, v in ranked[:top_n]:
            bar = "█" * int(v * 20)
            print(f"    {col:<30} V={v:.3f}  {bar}")
        print("  → For formal chi-square tests: use statistical_analysis.categorical_association_tests()")
        if top_cats:
            assoc_df = pd.DataFrame(ranked, columns=["feature", "cramers_v"]).set_index("feature")
            plot_categorical_vs_categorical(df, top_cats, target_col, assoc_df)


# ──────────────────────────────────────────────────────────────────────────────
# 7. PATTERN & ANOMALY DETECTION (VISUAL)
# ──────────────────────────────────────────────────────────────────────────────

def pattern_anomaly_analysis(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    n_cols: int = 3,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Visual detection of patterns, residual anomalies, and structural breaks.

    Detections
    ----------
    1. Extreme outliers (IQR + Z-score dual-method) — plotted on strip charts
    2. Bimodal / multimodal distributions — KDE peak detection
    3. Floor/ceiling effects — mass concentration at min/max bounds
    4. Zero-inflation — spike at zero beyond expected frequency
    5. Digit preference / rounding patterns — last-digit frequency chart
    6. Summary anomaly table

    Parameters
    ----------
    df               : DataFrame
    target_col       : if provided, color anomaly points by target class
    n_cols           : subplot columns
    zscore_threshold : Z-score cutoff for extreme outlier flag (default 3.0)
    iqr_multiplier   : IQR fence multiplier (default 1.5)

    Returns
    -------
    pd.DataFrame — one row per column with anomaly flags and counts
    """
    section_header("PATTERN & ANOMALY DETECTION")

    num_cols = _num_cols(df, exclude=target_col)
    if not num_cols:
        print("  No numeric columns to analyze.")
        return pd.DataFrame()

    records = []

    # ── Per-column anomaly detection ──────────────────────────────────────────
    for col in num_cols:
        s = df[col].dropna()
        n = len(s)
        if n < 4:
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        n_iqr_out = int(((s < lo) | (s > hi)).sum())

        mu, sd = s.mean(), s.std()
        n_z_out = int((((s - mu) / sd).abs() > zscore_threshold).sum()) if sd > 0 else 0

        # Floor / ceiling effect: > 20% of values at exact min or max
        floor_pct = (s == s.min()).sum() / n * 100
        ceil_pct  = (s == s.max()).sum() / n * 100
        floor_flag = floor_pct > 20
        ceil_flag  = ceil_pct  > 20

        # Zero-inflation: > 30% zeros for a non-binary column
        zero_pct  = (s == 0).sum() / n * 100
        zero_flag = zero_pct > 30 and s.nunique() > 2

        ku = float(stats.kurtosis(s))
        sk = float(stats.skew(s))

        bimodal_flag = ku < -0.5 and abs(sk) < 0.3

        records.append({
            "column":        col,
            "n":             n,
            "iqr_outliers":  n_iqr_out,
            "iqr_pct":       round(n_iqr_out / n * 100, 2),
            "z_outliers":    n_z_out,
            "z_pct":         round(n_z_out / n * 100, 2),
            "floor_pct":     round(floor_pct, 1),
            "ceil_pct":      round(ceil_pct, 1),
            "zero_pct":      round(zero_pct, 1),
            "floor_effect":  floor_flag,
            "ceil_effect":   ceil_flag,
            "zero_inflated": zero_flag,
            "bimodal_hint":  bimodal_flag,
            "skewness":      round(sk, 3),
            "kurtosis":      round(ku, 3),
        })

    anomaly_df = pd.DataFrame(records)
    if anomaly_df.empty:
        return anomaly_df

    # ── Print anomaly summary ─────────────────────────────────────────────────
    print(f"\n  {'Column':<28} {'IQR out':>8} {'Z out':>6} {'Floor%':>7} "
          f"{'Zero%':>6} {'Bimodal':>8} {'Flags'}")
    print("  " + "─" * 80)
    for _, row in anomaly_df.iterrows():
        flags = []
        if row["iqr_pct"] > 10:  flags.append("outliers")
        if row["floor_effect"]:   flags.append("floor")
        if row["ceil_effect"]:    flags.append("ceiling")
        if row["zero_inflated"]:  flags.append("0️zero-infl.")
        if row["bimodal_hint"]:   flags.append("bimodal?")
        flag_str = "  ".join(flags) if flags else " clean"
        print(
            f"  {row['column']:<28} {row['iqr_pct']:>7.1f}% {row['z_pct']:>5.1f}% "
            f"{row['floor_pct']:>6.1f}% {row['zero_pct']:>5.1f}% "
            f"{'yes' if row['bimodal_hint'] else 'no':>8}   {flag_str}"
        )

    # ── Visual: strip plots for top anomalous columns ─────────────────────────
    flagged = anomaly_df[
        (anomaly_df["iqr_pct"] > 5) |
        anomaly_df["floor_effect"] |
        anomaly_df["ceil_effect"]  |
        anomaly_df["zero_inflated"]
    ]["column"].tolist()[:9]

    if flagged:
        n_r = (len(flagged) - 1) // n_cols + 1
        fig, axes = plt.subplots(n_r, n_cols, figsize=(n_cols * 4, n_r * 3))
        axes = np.array(axes).flatten()

        for ax, col in zip(axes, flagged):
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            colors = ["#e74c3c" if (v < lo or v > hi) else "#3498db" for v in s]

            if target_col and target_col in df.columns:
                palette = sns.color_palette("Set2", df[target_col].nunique())
                tgt = df.loc[s.index, target_col]
                classes = tgt.unique()
                for i, cls in enumerate(classes):
                    mask = tgt == cls
                    ax.scatter(
                        range(mask.sum()), s[mask], alpha=0.4,
                        s=8, color=palette[i], label=str(cls)
                    )
                ax.legend(fontsize=6, loc="upper right")
            else:
                ax.scatter(range(len(s)), s.values, c=colors, alpha=0.4, s=8)

            ax.axhline(lo, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axhline(hi, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(labelsize=7)

        for ax in axes[len(flagged):]:
            ax.set_visible(False)

        fig.suptitle("Anomaly Strip Charts — Red = IQR outlier", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show()

    # ── Visual: KDE overlays for bimodal candidates ───────────────────────────
    bimodal_cols = anomaly_df[anomaly_df["bimodal_hint"]]["column"].tolist()[:6]
    if bimodal_cols:
        n_r = (len(bimodal_cols) - 1) // n_cols + 1
        fig, axes = plt.subplots(n_r, n_cols, figsize=(n_cols * 4, n_r * 3))
        axes = np.array(axes).flatten()
        for ax, col in zip(axes, bimodal_cols):
            s = df[col].dropna()
            s.plot.kde(ax=ax, color="#2ecc71", linewidth=2)
            ax.hist(s, bins=30, density=True, alpha=0.25, color="#2ecc71")
            ax.set_title(f"{col}\n(bimodal candidate)", fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes[len(bimodal_cols):]:
            ax.set_visible(False)
        fig.suptitle("KDE — Bimodal / Multimodal Distribution Candidates", fontsize=11)
        plt.tight_layout()
        plt.show()

    print("\n  → For formal outlier tests use statistical_analysis.variance_analysis()")
    return anomaly_df


# ──────────────────────────────────────────────────────────────────────────────
# 8. GROUP EFFECT VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def group_effect_analysis(
    df: pd.DataFrame,
    target_col: str,
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
    top_n: int = 8,
) -> None:
    """
    Visual comparison of distributions across target groups.

    Charts produced
    ---------------
    1. Overlapping KDE per group — reveals distribution shift between classes
    2. Mean ± 95% CI bar chart — visual effect size (no formal test)
    3. Heatmap of group means (z-scored) — global group profile
    4. Stacked bar charts for categorical features by group

    This is purely visual. For formal group comparison tests
    (t-test, Mann-Whitney, ANOVA, Kruskal-Wallis), use
    statistical_analysis.compare_groups().

    Parameters
    ----------
    df           : DataFrame
    target_col   : grouping variable (categorical or low-cardinality numeric)
    num_features : numeric features to plot (auto-selected if None)
    cat_features : categorical features to plot (auto-selected if None)
    top_n        : max features per chart type
    """
    section_header(f"GROUP EFFECT VISUALIZATION — by '{target_col}'")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")

    groups = df[target_col].dropna().unique()
    palette = sns.color_palette("Set2", len(groups))

    if num_features is None:
        # Rank by between-group mean difference (visual heuristic only)
        num_features = _num_cols(df, exclude=target_col)
        ranked = []
        for col in num_features:
            group_means = df.groupby(target_col)[col].mean()
            spread = float(group_means.max() - group_means.min())
            ranked.append((col, spread))
        ranked.sort(key=lambda x: x[1], reverse=True)
        num_features = [c for c, _ in ranked[:top_n]]

    if cat_features is None:
        cat_features = _cat_cols(df, exclude=target_col)[:top_n]

    # ── 1. Overlapping KDE per group ──────────────────────────────────────────
    if num_features:
        n_cols = 3
        n_r    = (len(num_features) - 1) // n_cols + 1
        fig, axes = plt.subplots(n_r, n_cols, figsize=(n_cols * 4.5, n_r * 3.5))
        axes = np.array(axes).flatten()

        for ax, col in zip(axes, num_features):
            for i, grp in enumerate(groups):
                subset = df.loc[df[target_col] == grp, col].dropna()
                if len(subset) > 5:
                    subset.plot.kde(ax=ax, label=str(grp), color=palette[i],
                                    linewidth=1.8, alpha=0.85)
            ax.set_title(col, fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for ax in axes[len(num_features):]:
            ax.set_visible(False)

        fig.suptitle(f"Distribution by Group — {target_col}", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()

    # ── 2. Mean ± 95% CI bar chart ────────────────────────────────────────────
    if num_features:
        if len(num_features) > 3:
            logger.info(
                "group_effect_analysis: Mean±CI panel shows only the first 3 of "
                "%d selected numeric features (%s) — see the KDE/heatmap panels "
                "above for the full set.",
                len(num_features), num_features[:3],
            )
        group_stats = (
            df.groupby(target_col)[num_features]
            .agg(["mean", "std", "count"])
        )
        fig, axes = plt.subplots(1, min(len(num_features), 3),
                                  figsize=(min(len(num_features), 3) * 4, 4))
        if len(num_features) == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        for ax, col in zip(axes, num_features[:3]):
            means  = df.groupby(target_col)[col].mean()
            sems   = df.groupby(target_col)[col].sem() * 1.96  # 95% CI
            ax.bar(means.index.astype(str), means.values,
                   yerr=sems.values, capsize=5, color=palette[:len(means)],
                   alpha=0.8, edgecolor="white")
            ax.set_title(col, fontsize=9)
            ax.set_xlabel(target_col, fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle("Mean ± 95% CI by Group (visual only — no formal test)", fontsize=11)
        plt.tight_layout()
        plt.show()

    # ── 3. Group profile heatmap (z-scored means) ─────────────────────────────
    if num_features and len(groups) >= 2:
        group_means = df.groupby(target_col)[num_features].mean()
        # Z-score columns for comparability
        z_means = (group_means - group_means.mean()) / (group_means.std() + 1e-8)
        fig, ax = plt.subplots(figsize=(max(8, len(num_features) * 0.7), len(groups) + 1))
        sns.heatmap(
            z_means, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5,
            cbar_kws={"label": "Z-score of group mean"},
        )
        ax.set_title(f"Group Profile Heatmap (z-scored) — {target_col}", fontsize=11)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.show()

    # ── 4. Stacked bar for categorical features ────────────────────────────────
    if cat_features:
        if len(cat_features) > 4:
            logger.info(
                "group_effect_analysis: showing stacked-bar charts for only the "
                "first 4 of %d selected categorical features (%s).",
                len(cat_features), cat_features[:4],
            )
        for col in cat_features[:4]:
            ct = pd.crosstab(df[target_col], df[col], normalize="index") * 100
            ct.plot(kind="bar", stacked=True, figsize=(8, 4),
                    colormap="Set3", edgecolor="white", linewidth=0.5)
            plt.title(f"'{col}' distribution within each {target_col} group", fontsize=10)
            plt.xlabel(target_col)
            plt.ylabel("% within group")
            plt.legend(loc="upper right", fontsize=8, title=col)
            plt.tight_layout()
            plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 9. RESIDUAL ANOMALY DETECTION (VISUAL)
# ──────────────────────────────────────────────────────────────────────────────

def residual_anomaly_analysis(
    df: pd.DataFrame,
    reference_col: str,
    feature_cols: Optional[List[str]] = None,
    n_cols: int = 3,
) -> None:
    """
    Detect and visualize residual anomalies relative to a reference column.

    For each numeric feature, fits a simple OLS trend vs. reference_col and
    plots residuals to reveal non-linear patterns, heteroscedasticity, and
    structural breaks — all visually, without formal tests.

    Parameters
    ----------
    df            : DataFrame
    reference_col : numeric column used as the X axis 
    feature_cols  : columns to residualize (all numeric if None)
    n_cols        : subplot columns
    """
    section_header(f"RESIDUAL ANOMALY ANALYSIS — reference: '{reference_col}'")

    if reference_col not in df.columns:
        raise ValueError(f"Reference column '{reference_col}' not found.")

    if feature_cols is None:
        feature_cols = _num_cols(df, exclude=reference_col)

    x_raw = df[reference_col].dropna()
    feature_cols = [c for c in feature_cols if c != reference_col and c in df.columns][:9]

    if not feature_cols:
        print("  No feature columns to analyze.")
        return

    n_r   = (len(feature_cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_r, n_cols, figsize=(n_cols * 4.5, n_r * 3.5))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, feature_cols):
        common_idx = x_raw.index.intersection(df[col].dropna().index)
        if len(common_idx) < 10:
            ax.set_visible(False)
            continue

        x = x_raw.loc[common_idx].values
        y = df.loc[common_idx, col].values

        # OLS residuals
        slope, intercept = np.polyfit(x, y, 1)
        y_hat   = slope * x + intercept
        resids  = y - y_hat

        # Color extreme residuals
        std_r  = resids.std()
        colors = ["#e74c3c" if abs(r) > 2.5 * std_r else "#3498db" for r in resids]

        ax.scatter(x, resids, c=colors, alpha=0.4, s=8)
        ax.axhline(0,  color="black", linewidth=0.8)
        ax.axhline( 2.5 * std_r, color="#e74c3c", linewidth=0.7, linestyle="--")
        ax.axhline(-2.5 * std_r, color="#e74c3c", linewidth=0.7, linestyle="--")
        ax.set_title(f"{col}\n(residuals vs {reference_col})", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(feature_cols):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Residual plots vs '{reference_col}' — Red = extreme residual (>2.5σ)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.show()
    print("\n  → Heteroscedasticity detected visually → confirm with Breusch-Pagan test")
    print("  Structural breaks detected visually → confirm with Chow test")


# ──────────────────────────────────────────────────────────────────────────────
# 13. FULL EDA ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def run_eda(
    df: pd.DataFrame,
    target_col: str,
    steps: Optional[List[str]] = None,
    top_n: int = 10,
    reference_col: Optional[str] = None,
) -> None:
    """
    Run the complete visual EDA pipeline.

    Steps
    -----
    "distributions"  → plot_distributions() + explore_categorical_distributions()
    "boxplots"       → box_plot()
    "violins"        → violin_plot(target_col=target_col)
    "target"         → analyze_target()
    "bivariate"      → bivariate_analysis()
    "patterns"       → pattern_anomaly_analysis()
    "groups"         → group_effect_analysis()
    "residuals"      → residual_anomaly_analysis() [requires reference_col]
   
    Note
    ----
    Formal statistical tests (normality, group comparison, VIF, effect sizes)
    are NOT included here. Use statistical_analysis.run_statistical_analysis().
    """
    ALL_STEPS = [
        "distributions", "boxplots", "violins", "target",
        "bivariate", "patterns", "groups",
        "importance", "insights", "leakage",
    ]
    if reference_col:
        ALL_STEPS.append("residuals")

    steps = steps or ALL_STEPS

    try:
        print("\n" + "╔" + "═" * 62 + "╗")
        print("║{:^62}║".format("  COMPLETE EDA — VISUAL & DESCRIPTIVE  "))
        print("║{:^62}║".format(f"  Target: {target_col}  "))
        print("║{:^62}║".format(f"  {df.shape[0]:,} rows × {df.shape[1]} columns  "))
        print("╚" + "═" * 62 + "╝\n")
    except UnicodeEncodeError:
        print("\n" + "+" + "=" * 62 + "+")
        print("|{:^62}|".format("  COMPLETE EDA -- VISUAL & DESCRIPTIVE  "))
        print("|{:^62}|".format(f"  Target: {target_col}  "))
        print("|{:^62}|".format(f"  {df.shape[0]:,} rows x {df.shape[1]} columns  "))
        print("+" + "=" * 62 + "+\n")

    step_map = {
        "distributions": lambda: (
            plot_distributions(df),
            explore_categorical_distributions(df),
        ),
        "boxplots":   lambda: box_plot(df),
        "violins":    lambda: violin_plot(df, target_col=target_col),
        "target":     lambda: analyze_target(df, target_col),
        "bivariate":  lambda: bivariate_analysis(df, target_col, top_n=top_n),
        "patterns":   lambda: pattern_anomaly_analysis(df, target_col=target_col),
        "groups":     lambda: group_effect_analysis(df, target_col),
        "residuals":  lambda: residual_anomaly_analysis(df, reference_col) if reference_col else None,
        
    }

    for i, step in enumerate(steps, 1):
        if step not in step_map:
            logger.warning("Unknown EDA step ignored: '%s'", step)
            continue
        print(f"\n{'─' * 64}")
        print(f"  [{i}/{len(steps)}]  {step.upper()}")
        print(f"{'─' * 64}")
        try:
            step_map[step]()
        except Exception as e:
            logger.error("Error at step '%s': %s", step, e, exc_info=True)

    print("\n" + "═" * 64)
    print(" for formal hypothesis tests, effect sizes, and VIF.")
    print("═" * 64 + "\n")
