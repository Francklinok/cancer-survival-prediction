"""
normalization_pipeline.py — Orchestrated normalization pipeline.

Encapsulates the complete normalization chain:
  analyze_column → strategy → apply → evaluate
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from ml_framework.config.config import NormalizationConfig
from ml_framework.preprocessing.apply_normalisation import apply_normalization
from ml_framework.evaluation.normalization_quality import evaluate_normalization_quality

logger = logging.getLogger("ml_framework.normalization_pipeline")


def run_normalization_pipeline(
    df: pd.DataFrame,
    anal_df: pd.DataFrame,
    strategies_df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
    copy_original: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Complete normalization pipeline.

    Parameters
    ----------
    df            : DataFrame to normalize
    anal_df       : result of analyze_column_properties()
    strategies_df : result of suggest_normalization_strategy()
    config        : NormalizationConfig (default if None)
    copy_original : keep a copy of the original data
    verbose       : print pipeline steps

    Returns
    -------
    dict with keys:
        strategies_df, df_original, df_normalized,
        transformation_log, evaluation_df
    """
    if config is None:
        config = NormalizationConfig()

    if verbose:
        print("\n  Starting normalization pipeline...")

    df_original, df_normalized, transformation_log = apply_normalization(
        df, strategies_df, config, copy_original
    )

    evaluation_df = evaluate_normalization_quality(
        df_original, df_normalized, transformation_log
    )

    # Ensure transformation_log is a DataFrame
    if isinstance(transformation_log, dict):
        transformation_log = pd.json_normalize(transformation_log)
    elif not isinstance(transformation_log, pd.DataFrame):
        transformation_log = pd.DataFrame(transformation_log)

    if verbose:
        print(f"Normalization complete — {df_normalized.shape[1]} columns transformed.")

    return {
        "strategies_df":      strategies_df,
        "df_original":        pd.DataFrame(df_original),
        "df_normalized":      pd.DataFrame(df_normalized),
        "transformation_log": transformation_log,
        "evaluation_df":      pd.DataFrame(evaluation_df),
    }
