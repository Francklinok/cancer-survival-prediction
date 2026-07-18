"""
specs/persist.py — ModuleSpec for persisting all artifacts.

Business logic untouched: wraps the inline joblib/json save calls verbatim,
exactly as PersistStep.run() does in orchestration/pipeline.py.

"""

from __future__ import annotations

from typing import Any, Dict

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec


def invoke(ctx: PipelineContext) -> None:
    import joblib
    import json
    import os

    save_dir = ctx.config.report.output_dir
    os.makedirs(save_dir, exist_ok=True)

    saved: Dict[str, str] = {}

    def _save(obj: Any, filename: str) -> None:
        path = os.path.join(save_dir, filename)
        try:
            if filename.endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=2, default=str)
            else:
                joblib.dump(obj, path)
            saved[filename] = path
            size_kb = os.path.getsize(path) / 1024
            print(f"  Saved: {filename:<40} ({size_kb:.1f} KB)")
        except Exception as e:
            import logging
            logging.getLogger("ml_framework.orchestration.v2.specs.persist").warning(
                "Could not save '%s': %s", filename, e
            )

    _save(ctx.best_model_estimator, "final_model.pkl")

    if ctx.scaler_strategies is not None:
        _save(ctx.scaler_strategies, "scaler_strategies.pkl")

    if ctx.transform_log is not None:
        _save(ctx.transform_log, "transform_log.pkl")

    if ctx.encoders:
        _save(ctx.encoders, "encoding_report.pkl")

    if ctx.significant_features:
        _save(ctx.significant_features, "selected_features.pkl")

    if ctx.model_card:
        _save(ctx.model_card, "model_card.json")

    if ctx.monitoring_df is not None:
        ctx.monitoring_df.to_csv(
            os.path.join(save_dir, "monitoring_baseline.csv"),
            index=False,
        )
        saved["monitoring_baseline.csv"] = os.path.join(save_dir, "monitoring_baseline.csv")

    ctx.artifacts["saved_files"] = saved
    print(f"\n  {len(saved)} artifact(s) saved to '{save_dir}/'")


SPEC = ModuleSpec(
    name="persist",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({
        "best_model_estimator", "scaler_strategies", "transform_log",
        "encoders", "significant_features", "model_card", "monitoring_df",
    }),
    outputs=frozenset({"saved_files"}),
    invoke=invoke,
    cost_hint="low",
    parallelizable=False,  
)
