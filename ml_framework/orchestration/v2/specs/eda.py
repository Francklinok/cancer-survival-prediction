"""
specs/eda.py — ModuleSpec for exploratory data analysis.

Business logic untouched: wraps analyze_target, bivariate_analysis,
leakage_exploration, diagnose_class_imbalance verbatim, exactly as
EDAStep.run() does in orchestration/pipeline.py.

This module produces TWO decision-relevant artifacts (leakage_report,
class_imbalance_report), each translated by its own dedicated adapter
(leakage_adapter.py, class_imbalance_adapter.py) rather than a single
adapter on this ModuleSpec — a module produces at most one Recommendation
via its own `produces_recommendation`/`adapter` pair, so multi-signal
modules are modeled as ordinary artifact producers and the recommendations
are derived by adapter-only specs downstream (see class_imbalance.py /
leakage.py in this package).
"""

from __future__ import annotations

import logging

import pandas as pd

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec

logger = logging.getLogger("ml_framework.orchestration.v2.specs.eda")


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.analysis.eda import analyze_target, bivariate_analysis
    from ml_framework.visualization.analysis.missing_plots import plot_missing_overview

    df = ctx.df_work
    target = ctx.target_column

    try:
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
        }).query("missing_count > 0").sort_values("missing_pct", ascending=False)
        ctx.missing_report = missing_df
        if not missing_df.empty:
            plot_missing_overview(df, missing_df)
    except Exception as e:
        logger.warning("plot_missing_overview: %s", e)

    if target and target in df.columns:
        try:
            analyze_target(df, target)
        except Exception as e:
            logger.warning("analyze_target: %s", e)
        try:
            bivariate_analysis(df, target)
        except Exception as e:
            logger.warning("bivariate_analysis: %s", e)

    if target and target in df.columns:
        try:
            from ml_framework.diagnostic.data_diagnostic import leakage_exploration
            leakage_df = leakage_exploration(df, target)
            ctx.artifacts["leakage_report"] = leakage_df
            high_risk = leakage_df[leakage_df.get("risk_level", pd.Series()) == "HIGH"] \
                if "risk_level" in leakage_df.columns else pd.DataFrame()
            if not high_risk.empty:
                logger.warning(
                    "Leakage detection: %d high-risk features — review before training.",
                    len(high_risk),
                )
        except Exception as e:
            logger.warning("leakage_exploration: %s", e)

    if target and target in df.columns:
        try:
            from ml_framework.diagnostic.class_imbalance import diagnose_class_imbalance
            imb = diagnose_class_imbalance(df[target], verbose=True, plot=False)
            ctx.class_imbalance_report = imb
            ctx.artifacts["class_imbalance"] = imb
        except Exception as e:
            logger.warning("diagnose_class_imbalance: %s", e)


SPEC = ModuleSpec(
    name="eda",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({"leakage_report", "class_imbalance_report", "missing_report", "df_work"}),
    invoke=invoke,
    cost_hint="medium",
)
