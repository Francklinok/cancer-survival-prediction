"""
specs/clean.py — ModuleSpec for data cleaning and quality control.

Business logic untouched: wraps ml_framework.preprocessing.clean.clean_dataframe
verbatim, exactly as CleanStep.run() does in orchestration/pipeline.py.
"""

from __future__ import annotations

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.preprocessing.clean import clean_dataframe

    n_before = ctx.df_work.shape
    ctx.df_work = clean_dataframe(
        ctx.df_work,
        id_cols=ctx.config.data.id_columns,
        max_missing_ratio=ctx.config.data.max_missing_ratio,
        min_variance=ctx.config.data.min_variance,
    )
    n_after = ctx.df_work.shape
    print(f"  Shape: {n_before} -> {n_after}")


SPEC = ModuleSpec(
    name="clean",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({"df_work"}),
    invoke=invoke,
    cost_hint="low",
)
