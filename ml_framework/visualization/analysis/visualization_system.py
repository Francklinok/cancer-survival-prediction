"""
visualization_system.py — Comprehensive visualization system for outlier analysis.

Provides VisualizationSystem: a class that compares distributions
before/after outlier treatment via:
  - Overlaid KDE curves (original vs treated)
  - Boxplots with highlighted outlier points
  - Correlation matrices (original, treated, absolute difference)
  - PCA projection (original vs treated)
  - Treatment effectiveness scores

Usage:
    from ml_framework.analysis.visualization_system import VisualizationSystem

    viz = VisualizationSystem()
    viz.analyze(df_original, df_treated, outlier_info, effectiveness_metrics)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ml_framework.visualization_system")


class VisualizationSystem:
    """
    Robust visualization system for comparing data before and after
    outlier treatment.

    Parameters
    ----------
    max_cols : maximum number of columns to display per chart (default 4)

    Methods
    -------
    analyze(original, treated, outlier_info, effectiveness_metrics)
    plot_distributions(original, treated, cols)
    plot_boxplots(data, outlier_info, cols)
    plot_correlation(original, treated, cols)
    plot_pca(original, treated, cols)
    plot_effectiveness(metrics)
    """

    def __init__(self, max_cols: int = 4):
        sns.set_theme(style="whitegrid")
        self.max_cols = max_cols

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ORCHESTRATOR
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        original_data: pd.DataFrame,
        treated_data: pd.DataFrame,
        outlier_info: Dict[str, Any],
        effectiveness_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run the complete before/after visual analysis.

        Parameters
        ----------
        original_data         : DataFrame before treatment
        treated_data          : DataFrame after treatment
        outlier_info          : dict {col: {'count': int, 'indices': list}}
        effectiveness_metrics : optional dict of effectiveness metrics per column
        """
        if original_data.empty or treated_data.empty:
            print("  Empty data — visualization not possible.")
            return

        numeric_cols = [
            c for c in original_data.select_dtypes(include=[np.number]).columns
            if c in treated_data.columns
        ]

        if not numeric_cols:
            print("  No valid numeric columns.")
            return

        top_cols = self._get_top_columns(outlier_info, numeric_cols)

        print(f"\n  VisualizationSystem — {len(numeric_cols)} numeric columns")
        print(f"  Columns displayed (top outliers): {top_cols}")

        self.plot_distributions(original_data, treated_data, top_cols)
        self.plot_boxplots(original_data, outlier_info, top_cols)
        self.plot_correlation(original_data, treated_data, numeric_cols)
        self.plot_pca(original_data, treated_data, numeric_cols)

        if effectiveness_metrics:
            self.plot_effectiveness(effectiveness_metrics)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIORITY COLUMN SELECTION
    # ──────────────────────────────────────────────────────────────────────────

    def _get_top_columns(
        self,
        outlier_info: Dict[str, Any],
        numeric_cols: List[str],
    ) -> List[str]:
        """Return the columns with the most outliers (up to max_cols)."""
        if not outlier_info:
            return numeric_cols[: self.max_cols]

        sorted_cols = sorted(
            outlier_info.items(),
            key=lambda x: x[1].get("count", 0) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        valid = [c for c, _ in sorted_cols if c in numeric_cols]
        return valid[: self.max_cols] or numeric_cols[: self.max_cols]

    # ──────────────────────────────────────────────────────────────────────────
    # DISTRIBUTIONS (KDE before / after)
    # ──────────────────────────────────────────────────────────────────────────

    def plot_distributions(
        self,
        original: pd.DataFrame,
        treated: pd.DataFrame,
        cols: List[str],
    ) -> None:
        """
        Overlay KDE curves before and after treatment for each column.

        Parameters
        ----------
        original : original DataFrame
        treated  : treated DataFrame
        cols     : columns to visualize
        """
        cols = [c for c in cols if c in original.columns and c in treated.columns]
        if not cols:
            return

        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes[0]

        for i, col in enumerate(cols):
            orig_s    = original[col].dropna()
            treated_s = treated[col].dropna()

            if not orig_s.empty:
                sns.kdeplot(orig_s,    ax=axes[i], label="Original", fill=True, alpha=0.35, color="steelblue")
            if not treated_s.empty:
                sns.kdeplot(treated_s, ax=axes[i], label="Treated",  fill=True, alpha=0.35, color="coral")

            axes[i].set_title(f"{col}", fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(alpha=0.3)

        fig.suptitle("Distributions before / after treatment", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # BOXPLOTS WITH HIGHLIGHTED OUTLIERS
    # ──────────────────────────────────────────────────────────────────────────

    def plot_boxplots(
        self,
        data: pd.DataFrame,
        outlier_info: Dict[str, Any],
        cols: List[str],
    ) -> None:
        """
        Boxplots with outlier points highlighted in red.

        Parameters
        ----------
        data         : DataFrame
        outlier_info : dict {col: {'indices': list}}
        cols         : columns to visualize
        """
        cols = [c for c in cols if c in data.columns]
        if not cols:
            return

        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes[0]

        for i, col in enumerate(cols):
            sns.boxplot(y=data[col], ax=axes[i], color="lightblue", width=0.4)

            # Highlight outliers in red
            col_info = outlier_info.get(col, {})
            indices  = col_info.get("indices", []) if isinstance(col_info, dict) else []
            valid_idx = [idx for idx in indices if idx in data.index]

            if valid_idx:
                outlier_vals = data.loc[valid_idx, col].dropna()
                axes[i].scatter(
                    np.random.normal(0, 0.04, len(outlier_vals)),
                    outlier_vals,
                    color="red", s=20, alpha=0.7, zorder=5, label=f"{len(valid_idx)} outliers",
                )
                axes[i].legend(fontsize=7)

            n_out = len(valid_idx)
            pct   = 100 * n_out / max(len(data[col].dropna()), 1)
            axes[i].set_title(f"{col}\n({n_out} outliers, {pct:.1f}%)", fontsize=9)
            axes[i].grid(alpha=0.3)

        fig.suptitle("Boxplots — Highlighted Outliers", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # CORRELATION MATRICES
    # ──────────────────────────────────────────────────────────────────────────

    def plot_correlation(
        self,
        original: pd.DataFrame,
        treated: pd.DataFrame,
        cols: List[str],
    ) -> None:
        """
        Display 3 heatmaps: original correlation, treated correlation, absolute difference.

        Parameters
        ----------
        original : original DataFrame
        treated  : treated DataFrame
        cols     : columns to include
        """
        cols = [c for c in cols if c in original.columns and c in treated.columns]
        if len(cols) < 2:
            return

        # Limit to 15 columns for readability
        cols = cols[:15]

        corr_orig    = original[cols].corr()
        corr_treated = treated[cols].corr()
        diff         = (corr_orig - corr_treated).abs()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        kw = dict(cmap="coolwarm", center=0, vmin=-1, vmax=1,
                  linewidths=0.3, annot=len(cols) <= 8, fmt=".1f")
        sns.heatmap(corr_orig,    ax=axes[0], **kw)
        axes[0].set_title("Correlation — Original")

        sns.heatmap(corr_treated, ax=axes[1], **kw)
        axes[1].set_title("Correlation — Treated")

        sns.heatmap(diff, cmap="Reds", ax=axes[2],
                    linewidths=0.3, annot=len(cols) <= 8, fmt=".1f")
        axes[2].set_title("Absolute Difference")

        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # PCA PROJECTION
    # ──────────────────────────────────────────────────────────────────────────

    def plot_pca(
        self,
        original: pd.DataFrame,
        treated: pd.DataFrame,
        cols: List[str],
    ) -> None:
        """
        Overlay PCA projections (2 components) of both datasets.

        Parameters
        ----------
        original : original DataFrame
        treated  : treated DataFrame
        cols     : numeric columns to use
        """
        cols = [c for c in cols if c in original.columns and c in treated.columns]
        if len(cols) < 2:
            return

        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        pca     = PCA(n_components=2)

        X_orig    = imputer.fit_transform(original[cols])
        X_treated = imputer.transform(treated[cols])

        X_orig    = scaler.fit_transform(X_orig)
        X_treated = scaler.transform(X_treated)

        orig_pca    = pca.fit_transform(X_orig)
        treated_pca = pca.transform(X_treated)

        var_exp = pca.explained_variance_ratio_

        plt.figure(figsize=(7, 5))
        plt.scatter(orig_pca[:, 0],    orig_pca[:, 1],    alpha=0.35, s=15,
                    color="steelblue", label="Original")
        plt.scatter(treated_pca[:, 0], treated_pca[:, 1], alpha=0.35, s=15,
                    color="coral",     label="Treated")
        plt.xlabel(f"PC1 ({var_exp[0]:.1%})")
        plt.ylabel(f"PC2 ({var_exp[1]:.1%})")
        plt.title("PCA Projection — Original vs Treated")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # EFFECTIVENESS SCORES
    # ──────────────────────────────────────────────────────────────────────────

    def plot_effectiveness(self, metrics: Dict[str, Any]) -> None:
        """
        Plot a bar chart of treatment effectiveness scores per column.

        Parameters
        ----------
        metrics : dict {col_name: {'overall_score': float, ...}}
        """
        cols_m, scores = [], []

        for col, m in metrics.items():
            if isinstance(m, dict) and col != "overall_effectiveness":
                score = m.get("overall_score", m.get("score", 0))
                cols_m.append(col)
                scores.append(float(score) * 100)

        if not cols_m:
            return

        fig, ax = plt.subplots(figsize=(max(8, len(cols_m) * 0.8), 4))

        colors = ["#2ecc71" if s >= 70 else "#e67e22" if s >= 40 else "#e74c3c" for s in scores]
        ax.bar(cols_m, scores, color=colors, edgecolor="white")

        for i, (c, s) in enumerate(zip(cols_m, scores)):
            ax.text(i, s + 1, f"{s:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0, max(115, max(scores, default=0) * 1.15))
        ax.axhline(70, color="green",  linestyle="--", alpha=0.5, label="Good (70%)")
        ax.axhline(40, color="orange", linestyle="--", alpha=0.5, label="Fair (40%)")
        ax.set_ylabel("Effectiveness score (%)")
        ax.set_title("Outlier treatment effectiveness by column")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=40)
        plt.tight_layout()
        plt.show()
