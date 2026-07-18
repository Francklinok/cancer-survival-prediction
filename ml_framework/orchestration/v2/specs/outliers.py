"""
specs/outliers.py — ModuleSpec for outlier detection and treatment.

Business logic untouched: wraps identify_outliers() + OutlierTreatmentSystem
verbatim, exactly as OutliersStep.run() does in orchestration/pipeline.py.

"""

from __future__ import annotations

import logging

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec, Recommendation

logger = logging.getLogger("ml_framework.orchestration.v2.specs.outliers")


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.preprocessing.outlier_detection import identify_outliers
    from ml_framework.preprocessing.outlier_treatment import (
        OutlierTreatmentConfig,
        OutlierTreatmentSystem,
    )

    df = ctx.df_work
    target = ctx.target_column
    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c != target
    ]

    if not num_cols:
        logger.info("OutliersStep: no numeric columns to process.")
        return

    outliers_dict = identify_outliers(
        df,
        columns=num_cols,
        method=ctx.config.data.outlier_method,
        contamination=ctx.config.data.contamination,
        return_mask=True,
        verbose=True,
    )
    ctx.outliers_dict = outliers_dict

    treatment_cfg = OutlierTreatmentConfig(method="winsorize")
    system = OutlierTreatmentSystem(treatment_cfg)
    ctx.df_work = system.apply(df, outliers_dict)

    try:
        from ml_framework.visualization.analysis.visualization_system import VisualizationSystem
        viz = VisualizationSystem(max_cols=4)
        viz.analyze(df, ctx.df_work, outliers_dict)
    except Exception as e:
        logger.warning("VisualizationSystem: %s", e)


def adapt(raw_output: dict) -> Recommendation:
    """
    raw_output: {col_name: {"count": int, "percentage": float, ...}}, the
    return value of identify_outliers() itself (confirmed by audit) — used
    directly, no independent recomputation of outlier bounds.
    """
    total_outliers = sum(int(v.get("count", 0)) for v in raw_output.values())
    affected_cols = [c for c, v in raw_output.items() if int(v.get("count", 0)) > 0]

    return Recommendation(
        topic="outlier_processing",
        required=total_outliers > 0,
        strategy="winsorize" if total_outliers > 0 else None,
        params={"affected_columns": affected_cols} if affected_cols else {},
        reason=f"{total_outliers} outlier value(s) across {len(affected_cols)} column(s)"
               if total_outliers > 0 else "no outliers detected",
        confidence=1.0,
        source_module="identify_outliers",
        raw=raw_output,
    )


SPEC = ModuleSpec(
    name="outliers",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({"df_work", "outliers_dict"}),
    invoke=invoke,
    produces_recommendation=True,
    adapter=adapt,
    cost_hint="high",  
)
