"""
specs/encode.py — ModuleSpec for categorical variable encoding.

Business logic untouched: wraps encode_dataframe() verbatim, exactly as
EncodeStep.run() does in orchestration/pipeline.py.
"""

from __future__ import annotations

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.preprocessing.encoding import encode_dataframe

    df_encoded, encoding_report = encode_dataframe(ctx.df_work, verbose=True)
    ctx.df_encoded = df_encoded
    ctx.df_work = df_encoded
    ctx.encoders = encoding_report
    ctx.artifacts["encoding_report"] = encoding_report
    print(f"  Shape after encoding: {df_encoded.shape}")


SPEC = ModuleSpec(
    name="encode",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({"df_work", "df_encoded", "encoders"}),
    invoke=invoke,
    cost_hint="low",
)
