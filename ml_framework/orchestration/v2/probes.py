"""
probes.py — Computes recommendations UP FRONT, before the plan is built.

Addresses a key timing constraint: some steps (like outliers or scaling) only 
know their skip/include status after running an analysis on real data. 

To solve this, this module runs the exact same cheap diagnostic checks up front. 
Its only job is to inform the Decision Engine. It never duplicates or replaces 
the actual work done later by the modules themselves if they are included.

This logic was previously validated in the Phase 4 preview tests and is now 
promoted to production code for the MedicalMLPipeline.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ml_framework.config.config import NormalizationConfig
from ml_framework.orchestration.v2.contracts import Recommendation
from ml_framework.orchestration.v2.specs.missing import adapt as adapt_missing
from ml_framework.orchestration.v2.specs.outliers import adapt as adapt_outliers
from ml_framework.orchestration.v2.specs.normalize import adapt as adapt_normalize

logger = logging.getLogger("ml_framework.orchestration.v2.probes")


def probe_recommendations(df: pd.DataFrame, target_column: str) -> List[Recommendation]:
    """
    Computes planning recommendations using the raw, post-ingest DataFrame 
    (ctx.df_work right after ingest, before any preprocessing runs). This captures 
    the exact state of the dataset that the adaptive planner needs to evaluate.

    Returns recommendations for the three adaptive topics: missing_values, 
    outlier_processing, and scaling. 

    If a probe fails (e.g., no numeric columns exist), it logs a warning and skips 
    that recommendation instead of crashing. The planner will then safely default 
    to including the corresponding module.
    """
    recommendations: List[Recommendation] = []

    n_missing = int(df.isnull().sum().sum())
    recommendations.append(adapt_missing({"n_missing": n_missing}))

    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_column]
    if not num_cols:
        logger.info("probe_recommendations: no numeric columns — skipping outliers/scaling probes.")
        return recommendations

    try:
        from ml_framework.preprocessing.outlier_detection import identify_outliers
        outliers_raw = identify_outliers(df, columns=num_cols, method="iqr", verbose=False)
        recommendations.append(adapt_outliers(outliers_raw))
    except Exception as e:
        logger.warning("probe_recommendations: outlier probe failed (%s) — skipping.", e)

    try:
        from ml_framework.analysis.column_analysis import analyze_column_properties
        from ml_framework.strategies.normalisation_strategy import suggest_normalization_strategy

        anal_df = analyze_column_properties(df[num_cols], verbose=False)
        strategies_df = suggest_normalization_strategy(anal_df, NormalizationConfig())
        recommendations.append(adapt_normalize(strategies_df))
    except Exception as e:
        logger.warning("probe_recommendations: normalization probe failed (%s) — skipping.", e)

    return recommendations
