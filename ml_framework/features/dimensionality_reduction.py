"""
dimensionality_reduction.py —  Dimensionality reduction.

Methods:
  - PCA  : automatic component selection (90% variance, reconstruction error)
  - t-SNE: non-linear 2D/3D visualization
  - UMAP : fast non-linear projection (if installed)

Features:
  - Automatic standardization before projection
  - Scree plot, cumulative variance, loadings heatmap
  - Annotated 2D/3D projection
  - Returns DataFrame + fitted model for inference

Visualizations delegated to visualization.features.dim_reduction_plots.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ml_framework.visualization.features.dim_reduction_plots import (
    plot_pca_analysis,
    plot_2d_projection,
)

logger = logging.getLogger("ml_framework.dim_reduction")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def dim_reduction(
    X: pd.DataFrame,
    method: str = "pca",
    n_components: Optional[int] = None,
    random_state: int = 42,
    plot: bool = True,
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Dimensionality reduction with automatic component selection.

    Parameters
    ----------
    X            : feature DataFrame (numeric columns)
    method       : 'pca' | 'tsne' | 'umap'
    n_components : number of components (auto-selected if None)
    random_state : random seed
    plot         : display visualizations
    y            : class labels for colored projections

    Returns
    -------
    X_reduced_df : pd.DataFrame — reduced data
    reducer      : fitted reducer model (for inference)
    """
    method = method.lower()
    print(f"\n===== DIMENSIONALITY REDUCTION ({method.upper()}) =====")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.median()))

    if method == "pca":
        return _pca_reduction(X, X_scaled, n_components, random_state, plot, y)
    elif method == "tsne":
        return _tsne_reduction(X, X_scaled, n_components or 2, random_state, plot, y)
    elif method == "umap":
        return _umap_reduction(X, X_scaled, n_components or 2, random_state, plot, y)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: 'pca', 'tsne', 'umap'.")


# =============================================================================
# PCA
# =============================================================================


def _pca_reduction(
    X: pd.DataFrame,
    X_scaled: np.ndarray,
    n_components: Optional[int],
    random_state: int,
    plot: bool,
    y: Optional[pd.Series],
) -> Tuple[pd.DataFrame, PCA]:

    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)
    explained_variance  = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components_90 = int(np.argmax(cumulative_variance >= 0.90)) + 1

    errors         = []
    max_components = min(30, X.shape[1])
    for n in range(1, max_components + 1):
        pca_temp     = PCA(n_components=n, random_state=random_state)
        reduced      = pca_temp.fit_transform(X_scaled)
        reconstructed = pca_temp.inverse_transform(reduced)
        errors.append(float(np.mean((X_scaled - reconstructed) ** 2)))

    optimal_error_n = int(np.argmin(errors)) + 1
    auto_n          = min(n_components_90, optimal_error_n)
    final_n         = n_components if n_components else auto_n

    print(f"  Components for ≥90% variance  : {n_components_90}")
    print(f"  Components at minimum error   : {optimal_error_n}")
    print(f"  Components retained         : {final_n}")

    reducer   = PCA(n_components=final_n, random_state=random_state)
    X_reduced = reducer.fit_transform(X_scaled)

    if plot:
        plot_pca_analysis(
            X, X_reduced, explained_variance, cumulative_variance,
            errors, max_components, final_n, reducer, y,
        )

    X_reduced_df = pd.DataFrame(
        X_reduced,
        columns=[f"PC{i+1}" for i in range(final_n)],
        index=X.index,
    )
    return X_reduced_df, reducer


# =============================================================================
# t-SNE
# =============================================================================


def _tsne_reduction(
    X: pd.DataFrame,
    X_scaled: np.ndarray,
    n_components: int,
    random_state: int,
    plot: bool,
    y: Optional[pd.Series],
) -> Tuple[pd.DataFrame, TSNE]:

    reducer   = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=min(30, len(X) // 4),
        n_iter=1000,
    )
    X_reduced = reducer.fit_transform(X_scaled)

    if plot:
        plot_2d_projection(X_reduced, y, title=f"t-SNE Projection ({n_components}D)")

    cols = [f"TSNE{i+1}" for i in range(X_reduced.shape[1])]
    return pd.DataFrame(X_reduced, columns=cols, index=X.index), reducer


# =============================================================================
# UMAP
# =============================================================================


def _umap_reduction(
    X: pd.DataFrame,
    X_scaled: np.ndarray,
    n_components: int,
    random_state: int,
    plot: bool,
    y: Optional[pd.Series],
) -> Tuple[pd.DataFrame, object]:

    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    except ImportError:
        logger.error("UMAP not installed: pip install umap-learn — falling back to t-SNE.")
        return _tsne_reduction(X, X_scaled, n_components, random_state, plot, y)

    X_reduced = reducer.fit_transform(X_scaled)

    if plot:
        plot_2d_projection(X_reduced, y, title=f"UMAP Projection ({n_components}D)")

    cols = [f"UMAP{i+1}" for i in range(X_reduced.shape[1])]
    return pd.DataFrame(X_reduced, columns=cols, index=X.index), reducer
