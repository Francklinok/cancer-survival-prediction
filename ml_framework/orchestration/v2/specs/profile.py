"""
specs/profile.py — ModuleSpec for data profiling and quality scoring.

Business logic untouched: wraps ml_framework.analysis.data_profiling.dataset_overview
verbatim, exactly as ProfileStep.run() does in orchestration/pipeline.py.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec

logger = logging.getLogger("ml_framework.orchestration.v2.specs.profile")


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.analysis.data_profiling import dataset_overview

    df = ctx.df_work
    try:
        ctx.num_summary, ctx.cat_summary = dataset_overview(df)
    except Exception as e:
        logger.warning("dataset_overview: %s", e)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    ctx.missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)

    total_cells = df.size
    n_miss = int(df.isnull().sum().sum())
    missing_rate = n_miss / total_cells if total_cells > 0 else 0
    quality_score = max(0.0, (1.0 - missing_rate) * 100)

    ctx.artifacts["quality_report"] = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_rate_pct": round(missing_rate * 100, 2),
        "quality_score": round(quality_score, 1),
        "n_duplicates": int(df.duplicated().sum()),
        "columns_with_missing": ctx.missing_report.index.tolist(),
    }

    print(f"  Quality score : {quality_score:.1f}/100")
    print(f"  Missing rate  : {missing_rate*100:.2f}%")
    print(f"  Duplicates    : {ctx.artifacts['quality_report']['n_duplicates']}")


SPEC = ModuleSpec(
    name="profile",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({"num_summary", "cat_summary", "missing_report", "quality_report", "df_work"}),
    invoke=invoke,
    cost_hint="low",
)
