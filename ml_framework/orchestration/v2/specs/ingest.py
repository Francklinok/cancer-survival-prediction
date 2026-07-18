"""
specs/ingest.py — ModuleSpec for data ingestion.

Business logic untouched: wraps ml_framework.services.data_loading.load_data
verbatim, exactly as IngestStep.run() does in orchestration/pipeline.py.
"""

from __future__ import annotations

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.services.data_loading import load_data

    file_path = ctx.artifacts.get("file_path", "")
    df_raw, df_work = load_data(
        file_path,
        target_column=ctx.target_column,
        encoding=ctx.config.data.encoding,
        sep=ctx.config.data.separator,
    )
    ctx.df_raw = df_raw
    ctx.df_work = df_work
    print(f"  Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")


SPEC = ModuleSpec(
    name="ingest",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset(),
    outputs=frozenset({"df_raw", "df_work"}),
    invoke=invoke,
    cost_hint="low",
    parallelizable=False,  # entry point of the DAG
)
