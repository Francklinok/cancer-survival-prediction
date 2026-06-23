"""
visualization/analysis/diagnostic_plots.py
— Plot functions for diagnostic_analysis.py.

"""

from __future__ import annotations

import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.visualization.diagnostic_plots")


def plot_diagnostic_association(
    top: pd.DataFrame,
    target_col: str,
    top_n: int,
    df: pd.DataFrame,
) -> None:
    """
    Two-panel chart for diagnostic_analysis():

    Left panel  — Horizontal bar chart of association strength per variable.
                  Blue = numeric (|Pearson r|), Coral = categorical (Cramér's V).
    Right panel — Overlapping density histogram of the top numeric variable
                  split by target class, to visualise the distribution shift.

    Parameters
    ----------
    top        : top-N rows of the diagnostic DataFrame (variable, type, association)
    target_col : name of the target column
    top_n      : number of variables shown (for title)
    df         : source DataFrame (needed for the class histogram)
    """
    if top.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(top) * 0.4 + 1)))

    # ── Left: association bar ─────────────────────────────────────────────────
    colors = ["steelblue" if t == "numeric" else "coral" for t in top["type"]]
    axes[0].barh(
        top["variable"][::-1],
        top["association"][::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    axes[0].set_xlabel("Association strength")
    axes[0].set_title(
        f"Top {top_n} variables associated with '{target_col}'\n"
        "(Numeric = |Pearson r|  |  Categorical = Cramér's V)",
        fontsize=10,
    )
    legend_handles = [
        mpatches.Patch(color="steelblue", label="Numeric  |Pearson r|"),
        mpatches.Patch(color="coral",     label="Categorical  Cramér's V"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8)
    axes[0].grid(axis="x", alpha=0.3)

    # ── Right: class density for top numeric variable ─────────────────────────
    top_num = top[top["type"] == "numeric"]["variable"].tolist()[:1]
    ax = axes[1]
    ax.set_title(f"Distribution by class — {target_col}", fontsize=10)

    if top_num:
        col = top_num[0]
        try:
            for cls, grp in df.groupby(target_col)[col]:
                ax.hist(
                    grp.dropna(), bins=20, alpha=0.6,
                    label=f"Class {cls}", density=True,
                )
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        except Exception as exc:
            logger.debug("Class histogram failed for '%s': %s", col, exc)
            ax.set_visible(False)
    else:
        ax.set_visible(False)

    plt.suptitle(
        f"Diagnostic Analysis — '{target_col}'",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_causal_effect(causal_df: pd.DataFrame, target_col: str) -> None:
    """
    Two-panel chart for causal_analysis():

    Left panel  — Cohen's d horizontal bar chart with threshold lines at ±0.5 and ±0.8.
                  Green = significant (p < 0.05), grey = not significant.
    Right panel — -log10(p-value) volcano-style bar chart with the p = 0.05 threshold line.

    Parameters
    ----------
    causal_df  : DataFrame returned by causal_analysis()
                 Expected columns: treatment_var, cohens_d, p_value, significant
    target_col : name of the outcome variable (for chart title)
    """
    if causal_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(causal_df) * 0.5 + 1)))

    colors = ["#2ecc71" if s else "#95a5a6" for s in causal_df["significant"]]

    # ── Left: Cohen's d ───────────────────────────────────────────────────────
    axes[0].barh(
        causal_df["treatment_var"][::-1],
        causal_df["cohens_d"][::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    axes[0].axvline(0,     color="black",  linewidth=0.8)
    axes[0].axvline( 0.5,  color="orange", linestyle="--", alpha=0.8, label="Medium (0.5)")
    axes[0].axvline( 0.8,  color="red",    linestyle="--", alpha=0.8, label="Large  (0.8)")
    axes[0].axvline(-0.5,  color="orange", linestyle="--", alpha=0.8)
    axes[0].axvline(-0.8,  color="red",    linestyle="--", alpha=0.8)
    axes[0].set_xlabel("Cohen's d  (standardized mean difference)")
    axes[0].set_title(f"Effect size on '{target_col}'", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.3)

    sig_patch   = mpatches.Patch(color="#2ecc71", label="Significant  p < 0.05")
    nosig_patch = mpatches.Patch(color="#95a5a6", label="Not significant")
    axes[0].legend(handles=[sig_patch, nosig_patch], fontsize=8)

    # ── Right: -log10(p) ─────────────────────────────────────────────────────
    log_p = -np.log10(causal_df["p_value"].clip(lower=1e-300))
    axes[1].barh(
        causal_df["treatment_var"][::-1],
        log_p[::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    axes[1].axvline(
        -np.log10(0.05), color="red", linestyle="--", alpha=0.8, label="p = 0.05"
    )
    axes[1].set_xlabel("-log10(p-value)")
    axes[1].set_title("Statistical significance", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Causal Analysis — effect on '{target_col}'",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

def rout_cause_plot(rca_df: pd.DataFrame, target_col: str, top_n: int) -> None:
        top = rca_df.head(top_n)
        colors = ["#2ecc71" if s else "#95a5a6" for s in top["significant"]]
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4 + 1)))
        ax.barh(top["variable"][::-1], top["variance_explained"][::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Variance Explained (effect²)")
        ax.set_title(f"Root Cause Analysis — Top {top_n} drivers of '{target_col}'",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

def contribution_analysis_plot(top:pd.DataFrame, target_col:str) -> None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4 + 1)))
        bars = ax.barh(top["feature"][::-1], top["contribution_pct"][::-1],
                       color="steelblue", edgecolor="white")
        ax.set_xlabel("Contribution (%)")
        ax.set_title(f"Contribution Analysis — '{target_col}'",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for bar, pct in zip(bars, top["contribution_pct"][::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        plt.show()

def segment_analysis_plot(df: pd.DataFrame, g_mean:float, target_col: str, segment_col: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.5 + 1)))

        axes[0].barh(df["segment"][::-1], df["mean"][::-1],
                     color="steelblue", edgecolor="white")
        axes[0].axvline(g_mean, color="red", linestyle="--", label=f"Global mean={g_mean:.2f}")
        axes[0].set_xlabel(f"Mean '{target_col}'")
        axes[0].set_title(f"Segment Means — '{segment_col}'", fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="x", alpha=0.3)

        colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in df["cohens_d_vs_rest"]]
        axes[1].barh(df["segment"][::-1], df["cohens_d_vs_rest"][::-1],
                     color=colors[::-1], edgecolor="white")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Cohen's d vs rest (pooled)")
        axes[1].set_title("Segment Effect Size", fontsize=10)
        axes[1].grid(axis="x", alpha=0.3)

        plt.suptitle(f"Segment Analysis — '{segment_col}' × '{target_col}'",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

def shap_global_analysis_plot(top:pd.DataFrame, target_col:str) -> None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4 + 1)))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                color="darkorange", edgecolor="white")
        ax.set_xlabel("|mean(SHAP value)|")
        ax.set_title(f"SHAP Global Analysis — '{target_col}'",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

def variance_decomposition_plot(vd_df: pd.DataFrame, target_col: str) -> None:
        top = vd_df
        colors = ["#2ecc71" if s else "#95a5a6" for s in top["significant"]]
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4 + 1)))
        ax.barh(top["grouping_var"][::-1], top["omega_squared"][::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Omega² (unbiased variance explained)")
        ax.set_title(f"Variance Decomposition — '{target_col}'",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

def anomaly_explanation_plot(top: pd.DataFrame) -> None:
    sig = top[top["significant"]]
    if not sig.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.5 + 1)))
        ax.barh(sig["feature"][::-1], sig["z_shift"][::-1],
                color=["#e74c3c" if z > 0 else "#3498db" for z in sig["z_shift"][::-1]],
                edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Z-score shift (anomaly mean − normal mean) / normal std")
        ax.set_title("Anomaly Explanation — Significant Feature Shifts",
                        fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

def cohort_analysis_Plot(cohort_stats: pd.DataFrame, cohort_col: str, target_col: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(cohort_stats["cohort"], cohort_stats["mean"],
                    marker="o", color="steelblue", linewidth=2, label="Mean")
    axes[0].fill_between(
        cohort_stats["cohort"],
        cohort_stats["mean"] - cohort_stats["std"].fillna(0),
        cohort_stats["mean"] + cohort_stats["std"].fillna(0),
        alpha=0.2, color="steelblue",
    )
    axes[0].set_xlabel(cohort_col)
    axes[0].set_ylabel(f"Mean '{target_col}'")
    axes[0].set_title(f"Cohort Trend — '{target_col}'", fontsize=10)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    colors = ["#2ecc71" if c >= 0 else "#e74c3c"
                for c in cohort_stats["change_vs_first_pct"]]
    axes[1].bar(range(len(cohort_stats)), cohort_stats["change_vs_first_pct"],
                color=colors, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(cohort_stats)))
    axes[1].set_xticklabels(cohort_stats["cohort"], rotation=45, ha="right")
    axes[1].set_ylabel("Change vs first cohort (%)")
    axes[1].set_title("Relative Change per Cohort", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(f"Cohort Analysis — '{cohort_col}' × '{target_col}'",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
     

def propensity_score_matching_plot(controls:pd.DataFrame, treated:pd.DataFrame, outcome_col: str, balance_df: pd.DataFrame, att: float, att_p: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(controls["ps"], controls[outcome_col], alpha=0.3,
                    label="Controls", color="steelblue", s=15)
    axes[0].scatter(treated["ps"], treated[outcome_col], alpha=0.4,
                    label="Treated", color="darkorange", s=15)
    axes[0].set_xlabel("Propensity Score")
    axes[0].set_ylabel(outcome_col)
    axes[0].set_title("Propensity Score Distribution", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    x = np.arange(len(balance_df))
    w = 0.35
    axes[1].bar(x - w/2, balance_df["smd_before"].abs(), w,
                label="Before", color="#e74c3c", alpha=0.8)
    axes[1].bar(x + w/2, balance_df["smd_after"].abs(), w,
                label="After", color="#2ecc71", alpha=0.8)
    axes[1].axhline(0.1, color="black", linestyle="--", alpha=0.7, label="SMD=0.1 threshold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(balance_df["confounder"], rotation=45, ha="right")
    axes[1].set_ylabel("|SMD|")
    axes[1].set_title("Covariate Balance Before/After Matching", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(f"PSM — ATT={att:.4f}  p={att_p:.4f}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
     
def difference_in_differences_plot(pre_value:str, post_value:str, t_pre: float, t_post: float, c_pre: float, c_post: float, did_estimate: float, outcome_col: str) -> None:

    fig, ax = plt.subplots(figsize=(8, 5))
    periods = [str(pre_value), str(post_value)]
    ax.plot(periods, [t_pre, t_post], "o-", color="darkorange",
            linewidth=2, label="Treated", markersize=8)
    ax.plot(periods, [c_pre, c_post], "s--", color="steelblue",
            linewidth=2, label="Control", markersize=8)
    ax.annotate(f"DiD = {did_estimate:.3f}", xy=(1, t_post),
                xytext=(0.65, (t_post + c_post) / 2),
                fontsize=10, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))
    ax.set_ylabel(outcome_col)
    ax.set_title(f"Difference-in-Differences — '{outcome_col}'",
                    fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def regression_discontinuity_plot(local: pd.DataFrame, coeffs: np.ndarray, cutoff: float, rdd_estimate: float, p_val: float, running_col: str, outcome_col: str,y:pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        left  = local[local["treated"] == 0]
        right = local[local["treated"] == 1]
        ax.scatter(left["x_centered"],  left[outcome_col],  alpha=0.4,
                   color="steelblue", s=15, label="Control")
        ax.scatter(right["x_centered"], right[outcome_col], alpha=0.4,
                   color="darkorange", s=15, label="Treated")

        for side_data, color in [(left, "steelblue"), (right, "darkorange")]:
            if len(side_data) < 2:
                continue
            xs = np.linspace(side_data["x_centered"].min(),
                             side_data["x_centered"].max(), 100)
            idx = side_data["treated"].iloc[0]
            ys = coeffs[0] + coeffs[1] * xs + coeffs[2] * idx + coeffs[3] * xs * idx
            ax.plot(xs, ys, color=color, linewidth=2)

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=f"Cutoff={cutoff}")
        ax.annotate(f"RDD={rdd_estimate:.3f}\np={p_val:.3f}",
                    xy=(0, (y.max() + y.min()) / 2),
                    fontsize=9, color="darkred")
        ax.set_xlabel(f"{running_col} (centered at cutoff)")
        ax.set_ylabel(outcome_col)
        ax.set_title(f"Regression Discontinuity — cutoff={cutoff}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

def causal_forest_plot(cate: np.ndarray, feat_imp: pd.Series, ate: float, ci_lo: float, ci_hi: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(cate, bins=30, color="darkorchid", edgecolor="white", alpha=0.8)
    axes[0].axvline(ate, color="red", linestyle="--", linewidth=2, label=f"ATE={ate:.3f}")
    axes[0].axvline(0,   color="black", linestyle=":", linewidth=1)
    axes[0].set_xlabel("CATE (Conditional ATE)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("CATE Distribution", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    top_feat = feat_imp.head(10)
    axes[1].barh(top_feat.index[::-1], top_feat.values[::-1],
                    color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Feature Importance (CATE heterogeneity)")
    axes[1].set_title("Treatment Effect Heterogeneity Drivers", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle(f"Causal Forest — ATE={ate:.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show() 

def double_machine_learning_plot(T_res: np.ndarray, Y_res: np.ndarray, ate: float, ci_lo: float, ci_hi: float, p_val: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(T_res, Y_res, alpha=0.3, s=15, color="steelblue")
    x_line = np.linspace(T_res.min(), T_res.max(), 100)
    axes[0].plot(x_line, ate * x_line, color="red", linewidth=2,
                    label=f"ATE slope = {ate:.3f}")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("T̃  (treatment residual)")
    axes[0].set_ylabel("Ỹ  (outcome residual)")
    axes[0].set_title("DML Partialling-Out Plot", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].barh(["ATE"], [ate], xerr=[[abs(ate - ci_lo)], [abs(ci_hi - ate)]],
                    color=["#2ecc71" if p_val < 0.05 else "#95a5a6"],
                    capsize=8, height=0.4)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Estimate")
    axes[1].set_title(f"DML ATE  p={p_val:.4f}", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle(f"Double Machine Learning — ATE={ate:.4f}  p={p_val:.4f}",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()