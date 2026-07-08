"""
visualization/features/dim_reduction_plots.py
— Dimensionality reduction visualizations: PCA, t-SNE, UMAP.

Public functions:
  plot_pca_analysis(X, X_reduced, explained_variance, cumulative_variance,
                    errors, max_components, final_n, reducer, y)
  plot_2d_projection(X_reduced, y, title)
  plot_biplot(X_reduced, components, feature_names, ax)
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.visualization.dim_reduction_plots")


def plot_pca_analysis(
    X: pd.DataFrame,
    X_reduced: np.ndarray,
    explained_variance: np.ndarray,
    cumulative_variance: np.ndarray,
    errors: list,
    max_components: int,
    final_n: int,
    reducer,
    y: Optional[pd.Series] = None,
) -> None:
    """
    Six-panel PCA diagnostic:
      cumulative variance, scree plot, reconstruction error,
      2D scatter, loadings heatmap, biplot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Cumulative variance
    axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                    marker="o", color="steelblue")
    axes[0, 0].axhline(0.90, color="red",   linestyle="--", label="90%")
    axes[0, 0].axhline(0.95, color="green", linestyle="--", label="95%")
    axes[0, 0].axvline(final_n, color="orange", linestyle=":", label=f"n={final_n}")
    axes[0, 0].set_xlabel("Number of Components")
    axes[0, 0].set_ylabel("Cumulative Variance")
    axes[0, 0].set_title("Cumulative Variance — PCA")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.4)

    _SCREE_MAX_SHOWN = 40
    n_shown = min(_SCREE_MAX_SHOWN, len(explained_variance))
    if len(explained_variance) > _SCREE_MAX_SHOWN:
        logger.info(
            "plot_pca_analysis: scree plot shows only the first %d of %d "
            "components (label legibility) — cumulative variance panel "
            "above still reflects all components.",
            _SCREE_MAX_SHOWN, len(explained_variance),
        )
    axes[0, 1].bar(range(1, n_shown + 1),
                   explained_variance[:n_shown], color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 1].set_xlabel("Component")
    axes[0, 1].set_ylabel("Explained Variance")
    title_suffix = f" (first {n_shown} of {len(explained_variance)})" if len(explained_variance) > _SCREE_MAX_SHOWN else ""
    axes[0, 1].set_title(f"Scree Plot{title_suffix}")
    axes[0, 1].grid(True, alpha=0.4)

    # Reconstruction error
    axes[0, 2].plot(range(1, max_components + 1), errors, marker="o",
                    linestyle="--", color="salmon")
    axes[0, 2].axvline(final_n, color="orange", linestyle=":", label=f"n={final_n}")
    axes[0, 2].set_xlabel("Number of Components")
    axes[0, 2].set_ylabel("MSE Reconstruction")
    axes[0, 2].set_title("Reconstruction Error")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.4)

    # 2D projection
    if final_n >= 2:
        scatter_kw = {"s": 40, "alpha": 0.7, "edgecolors": "k", "linewidths": 0.3}
        if y is not None:
            classes = y.unique()
            palette = sns.color_palette("Set2", len(classes))
            for cls, color in zip(classes, palette):
                mask = (y == cls).values
                axes[1, 0].scatter(
                    X_reduced[mask, 0], X_reduced[mask, 1],
                    label=str(cls), color=color, **scatter_kw
                )
            axes[1, 0].legend(fontsize=8)
        else:
            axes[1, 0].scatter(X_reduced[:, 0], X_reduced[:, 1],
                               **scatter_kw, color="steelblue")
        axes[1, 0].set_title("Projection PC1 × PC2")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].grid(True, alpha=0.4)

    # Loadings heatmap
    if X.shape[1] <= 30 and final_n >= 1:
        loadings = pd.DataFrame(
            reducer.components_.T,
            columns=[f"PC{i+1}" for i in range(final_n)],
            index=X.columns,
        )
        sns.heatmap(loadings, annot=final_n <= 6, cmap="coolwarm", fmt=".2f",
                    cbar=True, ax=axes[1, 1], linewidths=0.2)
        axes[1, 1].set_title("Variable Contributions to Components")
        axes[1, 1].tick_params(axis="y", labelsize=7)

    # Biplot
    if final_n >= 2 and X.shape[1] <= 20:
        plot_biplot(X_reduced, reducer.components_, X.columns, axes[1, 2])

    for ax in axes.flatten():
        if not ax.has_data():
            ax.set_visible(False)

    plt.suptitle("PCA Analysis — Dimensionality Reduction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_2d_projection(
    X_reduced: np.ndarray,
    y: Optional[pd.Series],
    title: str = "Projection",
) -> None:
    """
    2D scatter plot for t-SNE / UMAP projections, colored by class if provided.
    """
    plt.figure(figsize=(9, 7))

    if y is not None:
        classes = y.unique()
        palette = sns.color_palette("Set2", len(classes))
        for cls, color in zip(classes, palette):
            mask = (y == cls).values
            plt.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1],
                label=str(cls), color=color,
                alpha=0.7, s=40, edgecolors="k", linewidths=0.3,
            )
        plt.legend(title="Class", fontsize=9)
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    alpha=0.7, s=40, color="steelblue")

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_biplot(
    X_reduced: np.ndarray,
    components: np.ndarray,
    feature_names,
    ax,
) -> None:
    """
    Biplot overlay of PC1/PC2 scatter and loading arrows.
    """
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.4, s=20, color="steelblue")

    score_radius = float(np.max(np.abs(X_reduced[:, :2]))) if X_reduced.size else 1.0
    loading_radius = float(np.max(np.abs(components[:2, :]))) if components.size else 1.0
    scale = (score_radius / loading_radius * 0.8) if loading_radius > 0 else 1.0

    for i, fname in enumerate(feature_names):
        ax.annotate(
            fname,
            xy=(components[0, i] * scale, components[1, i] * scale),
            xytext=(components[0, i] * scale * 1.1, components[1, i] * scale * 1.1),
            fontsize=7, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        )
    ax.set_title("Biplot PC1 × PC2")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
  
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
