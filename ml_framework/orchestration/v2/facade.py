"""
facade.py — Making v2 a drop-in replacement via MedicalMLPipeline.

We wrapped the new v2 engine inside a thin facade called MedicalMLPipeline. 
The goal? Keep run(), predict(), and get_summary() exactly as they were
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_framework.config.config import FrameworkConfig
from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.artifact_registry import ArtifactRegistry
from ml_framework.orchestration.v2.contracts import DatasetProfile, Event, EventType
from ml_framework.orchestration.v2.dag_builder import build_dag
from ml_framework.orchestration.v2.decision_engine import PipelinePlanner
from ml_framework.orchestration.v2.event_bus import EventBus
from ml_framework.orchestration.v2.execution_engine import ExecutionEngine, ModuleExecutionError
from ml_framework.orchestration.v2.module_registry import ModuleRegistry, default_registry
from ml_framework.orchestration.v2.planning_rules import build_default_rules
from ml_framework.orchestration.v2.profiling import build_dataset_profile

logger = logging.getLogger("ml_framework.orchestration.v2.facade")


class MedicalMLPipeline:
    """
    Public facade for the medical MLOps pipeline, backed by the Decision
    Engine / DAG Builder / Execution Engine stack (v2).

    Parameters
    ----------
    config                    : FrameworkConfig — global configuration
    n_retries                 : automatic retries per module on error (default 0)
    enable_adaptive_skipping  : if True, outliers/normalize are skipped when
                                 their Recommendation says required=False
                                 (Phase 4 adaptive behavior). Default False
                                 reproduces pipeline.py's fixed ALL_STEPS
                                 exactly — see planning_rules.build_default_rules.
    module_registry           : override for testing; defaults to the real
                                 12-module registry.

   """

    ALL_STEPS = [
        "ingest", "profile", "clean", "eda",
        "missing", "outliers", "encode", "normalize",
        "features", "train", "evaluate", "persist",
    ]

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        n_retries: int = 0,
        enable_adaptive_skipping: bool = False,
        module_registry: Optional[ModuleRegistry] = None,
    ) -> None:
        self.config = config or FrameworkConfig()
        self.n_retries = n_retries
        self._adaptive = enable_adaptive_skipping
        self.registry = module_registry or default_registry
        self.planner = PipelinePlanner(
            rules=build_default_rules(enable_adaptive_skipping=enable_adaptive_skipping)
        )
        self.event_bus = EventBus()
        self.context: Optional[PipelineContext] = None
        self._hooks: Dict[str, Dict[str, List[Callable]]] = {}
        self._wire_hooks_to_event_bus()
        self._wire_step_tracking_to_event_bus()
        logger.info("MedicalMLPipeline (v2) initialized.")

    # ── Execution ────────────────────────────────────────────────────────────

    def run(
        self,
        file_path: str,
        target_column: str,
        steps: Optional[List[str]] = None,
    ) -> "MedicalMLPipeline":
        """
        Execute the pipeline on the provided data file.

        Parameters
        ----------
        file_path     : path to the data file (CSV, Excel, Parquet, JSON)
        target_column : name of the target column
        steps         : subset of steps to execute (all by default)

        Returns
        -------
        self — for chaining: pipeline.run(...).get_summary()
        """
        self.config.model.target_column = target_column
        ctx = PipelineContext(config=self.config, target_column=target_column)
        ctx.artifacts["file_path"] = file_path
        self._execute(ctx, steps or self.ALL_STEPS, initial_artifacts={"file_path", "target_column"})
        return self

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        steps: Optional[List[str]] = None,
    ) -> "MedicalMLPipeline":
        """
        Execute the pipeline directly from a DataFrame (skips ingestion).

        Parameters
        ----------
        df            : already loaded DataFrame
        target_column : name of the target column
        steps         : steps to execute (default: all except 'ingest')
        """
        self.config.model.target_column = target_column
        ctx = PipelineContext(config=self.config, target_column=target_column)
        ctx.df_raw = df.copy()
        ctx.df_work = df.copy()
        default_steps = [s for s in self.ALL_STEPS if s != "ingest"]
        self._execute(ctx, steps or default_steps, initial_artifacts={"df_raw", "df_work", "target_column"})
        return self

    def _execute(self, ctx: PipelineContext, steps: List[str], initial_artifacts: set) -> None:
        recommendations: List = []
        remaining_steps = list(steps)

        if self._adaptive and "ingest" in remaining_steps:
            self._run_ingest_only(ctx)
            remaining_steps = [s for s in remaining_steps if s != "ingest"]

        if self._adaptive and ctx.df_work is not None:
            from ml_framework.orchestration.v2.probes import probe_recommendations
            recommendations = probe_recommendations(ctx.df_work, ctx.target_column)

        placeholder_profile = DatasetProfile(problem_type="tabular", n_rows=0, n_columns=0)
        plan = self.planner.plan(
            recommendations=recommendations, profile=placeholder_profile, module_order=remaining_steps
        )

        # df_work/df_raw are already available once ingest has run standalone.
        already_available = initial_artifacts | ({"df_work", "df_raw"} if ctx.df_work is not None else set())
        dag = build_dag(plan, self.registry, initial_artifacts=already_available)
        for spec_name in dag.nodes:
            self.registry.get(spec_name)  

        engine = ExecutionEngine(self.registry, event_bus=self.event_bus, max_workers=4)
    
        self.context = ctx

        try:
            engine.run(dag, ctx, ArtifactRegistry())
        except ModuleExecutionError as exc:
            self._record_step_error(ctx, exc)

    def _run_ingest_only(self, ctx: PipelineContext) -> None:
        """
        Runs the "ingest" module alone, via its own single-node DAG, so
        adaptive planning can probe the resulting df_work before deciding
        the rest of the plan. Set self.context first for the
        same reason as the main run: hooks/step-time tracking read it live.
        """
        self.context = ctx
        plan = self.planner.plan(
            recommendations=[],
            profile=DatasetProfile(problem_type="tabular", n_rows=0, n_columns=0),
            module_order=["ingest"],
        )
        dag = build_dag(plan, self.registry, initial_artifacts={"file_path", "target_column"})
        engine = ExecutionEngine(self.registry, event_bus=self.event_bus, max_workers=1)
        try:
            engine.run(dag, ctx, ArtifactRegistry())
        except ModuleExecutionError as exc:
            self._record_step_error(ctx, exc)

    def _record_step_error(self, ctx: PipelineContext, exc: ModuleExecutionError) -> None:
        import traceback
        ctx.step_errors[exc.module_name] = "".join(
            traceback.format_exception(type(exc.original), exc.original, exc.original.__traceback__)
        )
        logger.error("Pipeline stopped after failure of '%s'.", exc.module_name)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Predict on new data using the fitted pipeline.

        Applies the exact same preprocessing chain used during training:
        normalization (with the fitted strategy), then model inference.
    

        Parameters
        ----------
        X            : raw feature DataFrame (pre-normalization, pre-encoding)
        return_proba : if True, return class probabilities instead of labels

        Returns
        -------
        np.ndarray of predictions or probabilities
        """
        if self.context is None or self.context.best_model_estimator is None:
            raise RuntimeError("Pipeline not yet executed or no model available.")

        ctx = self.context
        model = ctx.best_model_estimator
        X_proc = X.copy()

        if ctx.encoders:
            try:
                from ml_framework.preprocessing.encoding import encode_dataframe
                X_proc, _ = encode_dataframe(X_proc, verbose=False)
            except Exception as e:
                logger.warning("predict: encoding failed (%s) — using raw features.", e)

        if ctx.scaler_strategies is not None:
            try:
                from ml_framework.preprocessing.apply_normalisation import apply_normalization
                _, X_proc, _ = apply_normalization(
                    X_proc,
                    ctx.scaler_strategies,
                    ctx.config.normalization,
                )
            except Exception as e:
                logger.warning("predict: normalization failed (%s) — using unscaled features.", e)

        if ctx.X_train is not None:
            expected_cols = ctx.X_train.columns.tolist()
            for col in expected_cols:
                if col not in X_proc.columns:
                    X_proc[col] = 0
            X_proc = X_proc[expected_cols]

        if return_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(X_proc)
        return model.predict(X_proc)

    # ── Hooks ────────────────────────────────────────────────────────────────

    def add_hook(
        self,
        step_name: str,
        when: str,
        fn: Callable[[PipelineContext], None],
    ) -> "MedicalMLPipeline":
        """
        Add before/after hooks to steps using the EventBus (MODULE_STARTED / MODULE_COMPLETED) 
        """
        assert when in ("before", "after"), "when must be 'before' or 'after'"
        self._hooks.setdefault(step_name, {"before": [], "after": []})
        self._hooks[step_name][when].append(fn)
        return self

    def _wire_hooks_to_event_bus(self) -> None:
        def on_started(event: Event) -> None:
            module = event.payload.get("module")
            for fn in self._hooks.get(module, {}).get("before", []):
                self._safe_hook_call(fn)

        def on_completed(event: Event) -> None:
            module = event.payload.get("module")
            for fn in self._hooks.get(module, {}).get("after", []):
                self._safe_hook_call(fn)

        self.event_bus.subscribe(EventType.MODULE_STARTED, on_started)
        self.event_bus.subscribe(EventType.MODULE_COMPLETED, on_completed)

    def _safe_hook_call(self, fn: Callable[[PipelineContext], None]) -> None:
        try:
            fn(self.context)
        except Exception as e:
            logger.warning("Hook failed: %s", e)

    def _wire_step_tracking_to_event_bus(self) -> None:
        """
        Fills ctx.step_times using MODULE_COMPLETED events. The old pipeline runner used 
        to calculate this during execution, but since the v2 engine tracks time without
        knowing about ctx, this facade acts as a bridge. This ensures get_summary() still 
        gets its data exactly like before
        """
        def on_completed(event: Event) -> None:
            if self.context is None:
                return
            module = event.payload.get("module")
            elapsed = event.payload.get("elapsed_s")
            if module and elapsed is not None:
                self.context.step_times[module] = elapsed

        self.event_bus.subscribe(EventType.MODULE_COMPLETED, on_completed)

    # ── Results access  ─────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of pipeline results."""
        if self.context is None:
            return {"error": "Pipeline not yet executed."}

        ctx = self.context
        random_baseline = 1.0 / ctx.y_test.nunique() if ctx.y_test is not None else None
        f1 = ctx.final_metrics.get("f1_macro", ctx.final_metrics.get("f1"))

        summary: Dict[str, Any] = {
            "best_model":         ctx.best_model,
            "model_scores":       ctx.model_scores,
            "n_features":         ctx.X_train.shape[1] if ctx.X_train is not None else None,
            "n_train":            len(ctx.X_train) if ctx.X_train is not None else None,
            "n_test":             len(ctx.X_test) if ctx.X_test is not None else None,
            "final_metrics":      ctx.final_metrics,
            "step_times":         ctx.step_times,
            "errors":             ctx.step_errors,
            "saved_files":        ctx.artifacts.get("saved_files", {}),
        }

        if f1 is not None and random_baseline is not None:
            summary["gain_over_random"] = round(f1 - random_baseline, 4)
            summary["random_baseline"] = round(random_baseline, 4)

        return summary

    def get_best_model(self) -> Any:
        """Return the sklearn estimator of the best model."""
        if self.context is None:
            raise RuntimeError("Pipeline not yet executed.")
        return self.context.best_model_estimator

    def get_context(self) -> PipelineContext:
        """Return the complete PipelineContext for artifact access."""
        if self.context is None:
            raise RuntimeError("Pipeline not yet executed.")
        return self.context
