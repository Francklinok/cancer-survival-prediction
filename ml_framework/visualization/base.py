"""
visualization/base.py — Shared utilities for all visualization modules.

Provides:
  - configure_plot_style() : apply consistent matplotlib/seaborn theme
  - section_header()       : re-exported from utils.display_utils (canonical source)
  - save_or_show()         : show or save a figure to disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

# section_header lives in utils.display_utils — re-export for backward compat
from ml_framework.utils.display_utils import section_header  # noqa: F401

logger = logging.getLogger("ml_framework.visualization")


def configure_plot_style(
    style: str = "seaborn-v0_8-whitegrid",
    font_scale: float = 1.0,
    palette: str = "Set2",
    figsize: tuple = (10, 6),
    dpi: int = 100,
) -> None:
    """
    Apply a consistent visual style across all framework plots.

    Parameters
    ----------
    style      : matplotlib style name
    font_scale : seaborn font scale factor
    palette    : default seaborn color palette
    figsize    : default figure size (width, height) in inches
    dpi        : figure resolution
    """
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-whitegrid")

    sns.set_theme(
        style="whitegrid",
        font_scale=font_scale,
        palette=palette,
        rc={
            "figure.figsize": list(figsize),
            "figure.dpi": dpi,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        },
    )
    logger.debug("Plot style configured: %s", style)


def save_or_show(
    fig: Optional[plt.Figure] = None,
    save_path: Optional[str] = None,
    tight: bool = True,
) -> None:
    """
    Either display a matplotlib figure or save it to disk.

    Parameters
    ----------
    fig       : figure to save/show (uses plt.gcf() if None)
    save_path : file path to save the figure; if None, calls plt.show()
    tight     : apply tight_layout before rendering
    """
    if fig is None:
        fig = plt.gcf()

    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logger.info("Figure saved to: %s", path)
    else:
        plt.show()
