"""
pipeline.py - MLOps orchestrator.

  PipelineContext   — typed shared state object across all steps
  PipelineStep      — abstract base class: each step is a self-contained class
  StepRegistry      — step registry with before/after hooks, retry, skip support
  PipelineRunner    — orchestrates without knowing the business logic of each step
  MedicalMLPipeline — configures the runner, declares steps, exposes the API

Each step:
  - has declared inputs/outputs
  - can be executed, skipped, or retried independently
  - uses only framework modules (no inline business logic)
  - produces versioned artifacts in the context

Usage
-----
    from ml_framework.orchestration.pipeline import MedicalMLPipeline

    pipeline = MedicalMLPipeline(config)
    pipeline.run("data.csv", target_column="TreatmentResponse")

    # Partial execution
    pipeline.run("data.csv", target_column="TreatmentResponse",
                 steps=["ingest", "clean", "eda"])

    # Predict on new data
    predictions = pipeline.predict(new_df)

    # Access artifacts
    df_final = pipeline.context.final_dataset
    model    = pipeline.context.best_model_estimator
"""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from ml_framework.config.config import FrameworkConfig

logger = logging.getLogger("ml_framework.pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE CONTEXT — typed state object (replaces raw dict)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineContext:
    """
    Shared state object passed between all pipeline steps.

    Each step reads its inputs and writes its outputs to this object.
    Unlike a raw dict, attributes are declared and typed —
    this catches runtime errors and serves as living documentation.

    Attributes
    ----------
    config               : global framework configuration
    target_column        : name of the target variable
    df_raw               : raw DataFrame as loaded (never modified)
    df_work              : current DataFrame (modified at each step)
    df_encoded           : DataFrame after categorical encoding
    df_normalized        : DataFrame after normalization
    final_dataset        : final dataset ready for training
    X_train, X_test      : train/test features (post-normalization, post-encoding)
    X_train_raw          : train features before normalization (for quality eval)
    y_train, y_test      : train/test targets
    scaler_strategies    : normalization strategy DataFrame (for inference reuse)
    transform_log        : transformation log dict (for inference reuse)
    encoders             : encoding metadata (for inference reuse)
    models               : dict {name: estimator} of all trained models
    best_model           : name of the best model
    best_model_estimator : sklearn estimator of the best model
    model_scores         : dict {name: CV metrics dict}
    final_metrics        : evaluation metrics on the test set
    model_card           : generated Model Card (dict)
    fairness_results     : output of fairness_audit()
    monitoring_df        : performance snapshot from track_model_performance_over_time()
    artifacts            : free dict for any additional artifact
    step_times           : execution time per step
    step_errors          : captured errors per step
    """

    config: FrameworkConfig = field(default_factory=FrameworkConfig)
    target_column: str = ""

    # Data states
    df_raw: Optional[pd.DataFrame] = None
    df_work: Optional[pd.DataFrame] = None
    df_encoded: Optional[pd.DataFrame] = None
    df_normalized: Optional[pd.DataFrame] = None
    final_dataset: Optional[pd.DataFrame] = None

    # Train/test split — set by NormalizeStep (before any fitting)
    X_train: Optional[pd.DataFrame] = None
    X_train_raw: Optional[pd.DataFrame] = None   # pre-normalization, for quality eval
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    # Reusable preprocessing artifacts
    scaler_strategies: Optional[pd.DataFrame] = None
    transform_log: Optional[Dict[str, Any]] = None
    encoders: Dict[str, Any] = field(default_factory=dict)

    # Models
    models: Dict[str, Any] = field(default_factory=dict)
    best_model: Optional[str] = None
    best_model_estimator: Any = None
    model_scores: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    model_card: Dict[str, Any] = field(default_factory=dict)

    # Post-training analysis
    fairness_results: Dict[str, Any] = field(default_factory=dict)
    monitoring_df: Optional[pd.DataFrame] = None

    # Intermediate metadata
    num_summary: Optional[pd.DataFrame] = None
    cat_summary: Optional[pd.DataFrame] = None
    missing_report: Optional[pd.DataFrame] = None
    outliers_dict: Dict[str, Any] = field(default_factory=dict)
    normalization_result: Dict[str, Any] = field(default_factory=dict)
    significant_features: List[str] = field(default_factory=list)
    importance_df: Optional[pd.DataFrame] = None
    new_features: List[str] = field(default_factory=list)
    selection_results: Optional[pd.DataFrame] = None
    class_imbalance_report: Optional[Dict] = None

    # Free artifacts and execution metrics
    artifacts: Dict[str, Any] = field(default_factory=dict)
    step_times: Dict[str, float] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Return a summary of non-null context keys."""
        return {
            k: (v.shape if isinstance(v, (pd.DataFrame, pd.Series)) else type(v).__name__)
            for k, v in self.__dict__.items()
            if v is not None and not isinstance(v, FrameworkConfig)
            and k not in ("step_times", "step_errors", "artifacts")
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. PIPELINE STEP — abstract base class
# ══════════════════════════════════════════════════════════════════════════════


class PipelineStep(ABC):
    """
    Base class for any pipeline step.

    Each concrete step:
      - overrides ``run(ctx)`` with the business logic
      - declares ``name`` (unique identifier)
      - declares ``requires`` (list of required context attributes)
      - can be skipped via ``skip_if(ctx)``

    The runner calls ``execute(ctx)`` which handles validation,
    timing, and errors around ``run(ctx)``.
    """

    name: str = "step"
    requires: List[str] = []

    def skip_if(self, ctx: PipelineContext) -> bool:
        """Return True to skip this step (override as needed)."""
        return False

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Step business logic — modifies ctx in-place."""

    def execute(self, ctx: PipelineContext) -> bool:
        """
        Entry point called by the runner.
        Validates preconditions, executes run(), measures elapsed time.

        Returns True if successful, False if error or skipped.
        """
        if self.skip_if(ctx):
            logger.info("Step '%s': skipped (skip_if=True).", self.name)
            ctx.step_times[self.name] = 0.0
            return True

        missing_reqs = [r for r in self.requires if getattr(ctx, r, None) is None]
        if missing_reqs:
            msg = f"Step '{self.name}': missing prerequisites: {missing_reqs}"
            logger.error(msg)
            ctx.step_errors[self.name] = msg
            return False

        t0 = time.time()
        try:
            self.run(ctx)
            ctx.step_times[self.name] = round(time.time() - t0, 2)
            logger.info("Step '%s' completed in %.2fs.", self.name, ctx.step_times[self.name])
            return True
        except Exception as exc:
            ctx.step_times[self.name] = round(time.time() - t0, 2)
            ctx.step_errors[self.name] = traceback.format_exc()
            logger.error("Step '%s' failed: %s", self.name, exc, exc_info=True)
            return False


# ══════════════════════════════════════════════════════════════════════════════
# 3. STEP REGISTRY — registry + hooks + retry
# ══════════════════════════════════════════════════════════════════════════════


class StepRegistry:
    """
    Pipeline step registry with hook and retry support.

    Methods
    -------
    register(step_cls)           → register a step class (usable as decorator)
    add_hook(name, when, fn)     → add a before/after hook to a step
    get(name) -> PipelineStep    → instantiate the requested step
    all_names() -> List[str]     → list of steps in registration order
    """

    def __init__(self) -> None:
        self._steps: Dict[str, Type[PipelineStep]] = {}
        self._order: List[str] = []
        self._hooks: Dict[str, Dict[str, List[Callable]]] = {}

    def register(self, step_cls: Type[PipelineStep]) -> Type[PipelineStep]:
        """Register a step class (usable as a decorator)."""
        self._steps[step_cls.name] = step_cls
        if step_cls.name not in self._order:
            self._order.append(step_cls.name)
        self._hooks.setdefault(step_cls.name, {"before": [], "after": []})
        return step_cls

    def add_hook(
        self,
        step_name: str,
        when: str,
        fn: Callable[[PipelineContext], None],
    ) -> None:
        """Add a hook to execute before or after a step."""
        assert when in ("before", "after"), "when must be 'before' or 'after'"
        self._hooks.setdefault(step_name, {"before": [], "after": []})
        self._hooks[step_name][when].append(fn)

    def get(self, name: str) -> PipelineStep:
        """Instantiate and return the requested step."""
        if name not in self._steps:
            raise KeyError(f"Unknown step: '{name}'. Available: {self._order}")
        return self._steps[name]()

    def all_names(self) -> List[str]:
        """Return step names in registration order."""
        return list(self._order)

    def get_hooks(self, name: str, when: str) -> List[Callable]:
        return self._hooks.get(name, {}).get(when, [])


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE RUNNER — orchestrates without knowing business logic
# ══════════════════════════════════════════════════════════════════════════════


class PipelineRunner:
    """
    Orchestrates execution of steps registered in a StepRegistry.

    Parameters
    ----------
    registry      : populated StepRegistry
    n_retries     : number of retry attempts on error (default 0)
    stop_on_error : stop immediately if a step fails (default True)
    """

    def __init__(
        self,
        registry: StepRegistry,
        n_retries: int = 0,
        stop_on_error: bool = True,
    ) -> None:
        self.registry = registry
        self.n_retries = n_retries
        self.stop_on_error = stop_on_error

    def run(
        self,
        ctx: PipelineContext,
        steps: Optional[List[str]] = None,
    ) -> PipelineContext:
        """
        Execute a sequence of steps on the given context.

        Parameters
        ----------
        ctx   : PipelineContext shared between all steps
        steps : subset of steps to execute (all if None)

        Returns
        -------
        Updated PipelineContext
        """
        steps_to_run = steps or self.registry.all_names()
        t_global = time.time()
        n_ok = 0
        n_fail = 0

        _sep1 = "=" * 64
        _sep2 = "-" * 56

        print(f"\n{_sep1}")
        print("  MEDICAL ML PIPELINE — STARTING")
        print(f"  Planned steps: {steps_to_run}")
        print(f"{_sep1}")

        for step_name in steps_to_run:
            print(f"\n{_sep2}")
            print(f"  STEP: {step_name.upper()}")
            print(f"{_sep2}")

            for hook in self.registry.get_hooks(step_name, "before"):
                try:
                    hook(ctx)
                except Exception as e:
                    logger.warning("Hook before '%s' failed: %s", step_name, e)

            step = self.registry.get(step_name)
            success = False
            for attempt in range(self.n_retries + 1):
                if attempt > 0:
                    logger.info("Retry %d/%d for '%s'...", attempt, self.n_retries, step_name)
                success = step.execute(ctx)
                if success:
                    break

            if success:
                n_ok += 1
                print(f"  OK  [{ctx.step_times.get(step_name, 0):.2f}s]")
            else:
                n_fail += 1
                print("  FAILED  (see logs)")
                if self.stop_on_error:
                    logger.error("Pipeline stopped after failure of '%s'.", step_name)
                    break

            for hook in self.registry.get_hooks(step_name, "after"):
                try:
                    hook(ctx)
                except Exception as e:
                    logger.warning("Hook after '%s' failed: %s", step_name, e)

        total_time = round(time.time() - t_global, 2)
        self._print_summary(ctx, n_ok, n_fail, total_time)
        return ctx

    def _print_summary(
        self,
        ctx: PipelineContext,
        n_ok: int,
        n_fail: int,
        total_time: float,
    ) -> None:
        """Print the final execution report."""
        print("\n" + "=" * 64)
        print("  PIPELINE EXECUTION REPORT")
        print("=" * 64)
        print(f"  Steps succeeded : {n_ok}")
        print(f"  Steps failed    : {n_fail}")
        print(f"  Total time      : {total_time:.2f}s")
        print()
        print("  Step details:")
        for step_name, t in ctx.step_times.items():
            status = "FAILED" if step_name in ctx.step_errors else "OK"
            print(f"    {step_name:<22} {status:<6}  {t:.2f}s")

        if ctx.step_errors:
            print("\n  Errors:")
            for step_name, err in ctx.step_errors.items():
                print(f"    {step_name}: {err[:200]}...")

        # Insight report
        try:
            from ml_framework.insight.analyze_insight import (
                analyze_data_insights,
                analyze_model_insights,
                generate_pipeline_report,
            )
            data_insights = analyze_data_insights(
                ctx.df_raw if ctx.df_raw is not None else pd.DataFrame(),
                ctx.df_work,
                ctx.final_dataset,
                ctx.target_column,
            )
            model_insights = analyze_model_insights(
                ctx.model_scores,
                ctx.best_model,
            )
            generate_pipeline_report(data_insights, model_insights)
        except Exception as e:
            logger.warning("Insights report not generated: %s", e)

        print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONCRETE STEPS — each step is a self-contained class
# ══════════════════════════════════════════════════════════════════════════════

registry = StepRegistry()


@registry.register
class IngestStep(PipelineStep):
    """
    Step 1 — Data ingestion and validation.

    Input  : ctx.artifacts['file_path'], ctx.target_column
    Output : ctx.df_raw, ctx.df_work
    """

    name = "ingest"
    requires = []

    def run(self, ctx: PipelineContext) -> None:
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


@registry.register
class ProfileStep(PipelineStep):
    """
    Step 2 — Data profiling and quality scoring.

    Input  : ctx.df_work
    Output : ctx.artifacts['quality_report'], ctx.num_summary, ctx.cat_summary
    """

    name = "profile"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.analysis.data_profiling import dataset_overview

        df = ctx.df_work
        try:
            ctx.num_summary, ctx.cat_summary = dataset_overview(df)
        except Exception as e:
            logger.warning("dataset_overview: %s", e)

        # Missing value report
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


@registry.register
class CleanStep(PipelineStep):
    """
    Step 3 — Data cleaning and quality control.

    Input  : ctx.df_work
    Output : ctx.df_work (cleaned)
    """

    name = "clean"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
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


@registry.register
class EDAStep(PipelineStep):
    """
    Step 4 — Exploratory data analysis.

    Input  : ctx.df_work, ctx.target_column
    Output : ctx.artifacts['diagnostic_analysis']
    """

    name = "eda"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.analysis.eda import analyze_target, bivariate_analysis
        from ml_framework.visualization.analysis.missing_plots import plot_missing_overview

        df = ctx.df_work
        target = ctx.target_column

        # Missing value overview (visual)
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

        # Leakage detection
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

        # Class imbalance diagnosis (before any resampling)
        if target and target in df.columns:
            try:
                from ml_framework.diagnostic.class_imbalance import diagnose_class_imbalance
                imb = diagnose_class_imbalance(df[target], verbose=True, plot=False)
                ctx.class_imbalance_report = imb
                ctx.artifacts["class_imbalance"] = imb
            except Exception as e:
                logger.warning("diagnose_class_imbalance: %s", e)


@registry.register
class MissingStep(PipelineStep):
    """
    Step 5 — Missing value imputation.

    Input  : ctx.df_work
    Output : ctx.df_work (imputed)
    """

    name = "missing"
    requires = ["df_work"]

    def skip_if(self, ctx: PipelineContext) -> bool:
        if ctx.df_work is None:
            return False
        n_miss = int(ctx.df_work.isnull().sum().sum())
        return n_miss == 0

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.preprocessing.missing_value import missing_data_handling

        df_imp, report = missing_data_handling(
            ctx.df_work,
            strategy=ctx.config.data.missing_strategy,
        )
        ctx.df_work = df_imp
        ctx.artifacts["missing_imputation_report"] = report
        n_remaining = int(df_imp.isnull().sum().sum())
        print(f"  Missing values after imputation: {n_remaining}")


@registry.register
class OutliersStep(PipelineStep):
    """
    Step 6 — Outlier detection and treatment.

    Input  : ctx.df_work, ctx.target_column
    Output : ctx.df_work (treated), ctx.outliers_dict
    """

    name = "outliers"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
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


@registry.register
class EncodeStep(PipelineStep):
    """
    Step 7 — Categorical variable encoding.

    Input  : ctx.df_work
    Output : ctx.df_encoded, ctx.df_work (updated), ctx.encoders

    """

    name = "encode"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.preprocessing.encoding import encode_dataframe

        df_encoded, encoding_report = encode_dataframe(ctx.df_work, verbose=True)
        ctx.df_encoded = df_encoded
        ctx.df_work = df_encoded
        ctx.encoders = encoding_report
        ctx.artifacts["encoding_report"] = encoding_report
        print(f"  Shape after encoding: {df_encoded.shape}")


@registry.register
class NormalizeStep(PipelineStep):
    """
    Step 8 — Zero-leakage smart normalization.

    Critical ordering:
      1. Split train/test BEFORE any fitting
      2. Fit transformers on X_train only
      3. Apply fitted transformers to X_test (no re-fit)

    Input  : ctx.df_work, ctx.target_column
    Output : ctx.X_train, ctx.X_train_raw, ctx.X_test, ctx.y_train, ctx.y_test,
             ctx.df_normalized, ctx.scaler_strategies, ctx.transform_log,
             ctx.normalization_result
    """

    name = "normalize"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
        from sklearn.model_selection import train_test_split
        from ml_framework.analysis.column_analysis import analyze_column_properties
        from ml_framework.strategies.normalisation_strategy import suggest_normalization_strategy
        from ml_framework.preprocessing.apply_normalisation import apply_normalization
        from ml_framework.evaluation.normalization_quality import evaluate_normalization_quality
        from ml_framework.config.config import NormalizationConfig

        df = ctx.df_work
        target = ctx.target_column
        cfg_norm = ctx.config.normalization

        if not target or target not in df.columns:
            logger.warning("NormalizeStep: target '%s' not in df — normalizing full df.", target)
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            anal_df = analyze_column_properties(df[num_cols], verbose=False)
            strats_df = suggest_normalization_strategy(anal_df, cfg_norm)
            _, df_norm, t_log = apply_normalization(df, strats_df, cfg_norm)
            ctx.df_normalized = df_norm
            ctx.df_work = df_norm
            ctx.scaler_strategies = strats_df
            ctx.transform_log = t_log
            return

        # ── 1. Split BEFORE normalization ─────────────────────────────────────
        X = df.drop(columns=[target])
        y = df[target]
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y,
                test_size=ctx.config.model.test_size,
                random_state=ctx.config.model.random_state,
                stratify=y if y.nunique() <= 20 else None,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y,
                test_size=ctx.config.model.test_size,
                random_state=ctx.config.model.random_state,
            )

        ctx.X_train_raw = X_tr.copy()   # saved for normalization quality eval
        ctx.y_train = y_tr
        ctx.y_test = y_te

        # ── 2. Fit strategy on X_train only ───────────────────────────────────
        num_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()
        anal_df = analyze_column_properties(X_tr[num_cols] if num_cols else X_tr, verbose=False)
        strats_df = suggest_normalization_strategy(anal_df, cfg_norm)
        ctx.scaler_strategies = strats_df

        # ── 3. Apply to train (fit+transform) ─────────────────────────────────
        X_tr_orig, X_tr_norm, t_log = apply_normalization(X_tr, strats_df, cfg_norm)
        ctx.transform_log = t_log

        # ── 4. Apply to test (transform only — no re-fit) ────
        _, X_te_norm, _ = apply_normalization(X_te, strats_df, cfg_norm)

        ctx.X_train = X_tr_norm
        ctx.X_test = X_te_norm

        # ── 5. Normalization quality evaluation ───────────────────────────────
        try:
            eval_df = evaluate_normalization_quality(
                df_original=X_tr_orig,
                df_normalized=X_tr_norm,
                transformation_log=t_log if isinstance(t_log, dict) else {},
            )
            ctx.normalization_result = {
                "strategies_df": strats_df,
                "evaluation_df": eval_df,
                "transform_log": t_log,
            }
            ctx.artifacts["normalization_quality"] = eval_df
        except Exception as e:
            logger.warning("evaluate_normalization_quality: %s", e)

        # Rebuild a full normalized df for downstream steps
        df_full_norm = X_tr_norm.copy()
        df_full_norm[target] = y_tr.values[:len(df_full_norm)]
        ctx.df_normalized = df_full_norm
        ctx.df_work = df_full_norm

        print(f"  Train: {X_tr_norm.shape}  |  Test: {X_te_norm.shape}  (fit on train only)")


@registry.register
class FeatureEngineeringStep(PipelineStep):
    """
    Step 9 — Medical feature engineering + statistical feature selection.

    Input  : ctx.df_work, ctx.target_column, ctx.X_train, ctx.X_test
    Output : ctx.df_work, ctx.final_dataset, ctx.significant_features,
             ctx.new_features, ctx.X_train (filtered), ctx.X_test (filtered)
    """

    name = "features"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.features.feature_engineering import engineer_features
        from ml_framework.features.statistical_feature_selection import (
            statistical_feature_selection,
        )
        from ml_framework.evaluation.feature_importance import (
            create_feature_importance_summary,
        )

        df = ctx.df_work
        target = ctx.target_column

        # Engineering
        df_eng, new_feats = engineer_features(df, target_col=target)
        ctx.new_features = new_feats
        ctx.artifacts["df_engineered"] = df_eng.copy()
        print(f"  {len(new_feats)} new features created.")

        if not target or target not in df_eng.columns:
            ctx.final_dataset = df_eng
            ctx.df_work = df_eng
            return

        X = df_eng.drop(columns=[target])
        y = df_eng[target]

        # Statistical selection
        try:
            sig_feats, sel_results = statistical_feature_selection(X, y)
            ctx.significant_features = sig_feats
            ctx.selection_results = sel_results
        except Exception as e:
            logger.warning("statistical_feature_selection: %s — keeping all features.", e)
            sig_feats = X.columns.tolist()
            ctx.significant_features = sig_feats

        final_cols = list(set(sig_feats) | {target})
        final_cols = [c for c in final_cols if c in df_eng.columns]
        ctx.final_dataset = df_eng[final_cols]
        ctx.df_work = ctx.final_dataset

        # Align the train/test splits with the new feature set
        feat_cols = [c for c in sig_feats if c != target]
        if ctx.X_train is not None:
            train_feat = [c for c in feat_cols if c in ctx.X_train.columns]
            if train_feat:
                ctx.X_train = ctx.X_train[train_feat]
                ctx.X_test = ctx.X_test[[c for c in train_feat if c in ctx.X_test.columns]]

        print(f"  {len(sig_feats)} features selected.")


@registry.register
class TrainStep(PipelineStep):
    """
    Step 10 — Model training with nested cross-validation + Bayesian tuning.

    Input  : ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test
    Output : ctx.models, ctx.best_model, ctx.best_model_estimator,
             ctx.model_scores
    """

    name = "train"
    requires = ["df_work"]

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.modeling.model_trainer import (
            train_models,
            hyperparameter_tuning,
            nested_cv_evaluation,
        )
        from ml_framework.modeling.model_registry import get_model

        # Use the pre-computed split if available; otherwise split from df_work
        if ctx.X_train is not None and ctx.y_train is not None:
            train_state = train_models(
                ctx.X_train, ctx.y_train,
                X_test=ctx.X_test, y_test=ctx.y_test,
                models_to_test=ctx.config.model.models_to_test,
                cv_folds=ctx.config.model.cv_folds,
                random_state=ctx.config.model.random_state,
                n_jobs=ctx.config.model.n_jobs,
            )
        else:
            df = ctx.final_dataset if ctx.final_dataset is not None else ctx.df_work
            target = ctx.target_column
            if not target or target not in df.columns:
                logger.warning("Target '%s' not found — training skipped.", target)
                return
            train_state = train_models(
                df.drop(columns=[target]), df[target],
                models_to_test=ctx.config.model.models_to_test,
                test_size=ctx.config.model.test_size,
                cv_folds=ctx.config.model.cv_folds,
                random_state=ctx.config.model.random_state,
                n_jobs=ctx.config.model.n_jobs,
            )

        ctx.models = train_state.get("models", {})
        ctx.best_model = train_state.get("best_model")
        ctx.model_scores = train_state.get("evaluation_results", {})

        if ctx.X_train is None:
            ctx.X_train = train_state.get("X_train")
            ctx.X_test = train_state.get("X_test")
            ctx.y_train = train_state.get("y_train")
            ctx.y_test = train_state.get("y_test")

        if ctx.best_model and ctx.best_model in ctx.models:
            ctx.best_model_estimator = ctx.models[ctx.best_model]

        # ── Bayesian hyperparameter optimization ──────────────────────────────
        if ctx.config.model.perform_hyperparameter_tuning and ctx.best_model:
            try:
                from ml_framework.optimization.bayesian_optimization import (
                    optimize_hyperparameters_bayesian,
                )

                model_prefix = ctx.best_model.split("_")[0]
                base_estimator = get_model(model_prefix)
                param_space = _get_param_space(model_prefix)

                if param_space:
                    bayes_model, _ = optimize_hyperparameters_bayesian(
                        X=ctx.X_train,
                        y=ctx.y_train,
                        estimator=base_estimator,
                        param_space=param_space,
                        cv=ctx.config.model.cv_folds,
                        n_iter=ctx.config.model.n_iter_bayesian,
                        n_jobs=ctx.config.model.n_jobs,
                        scoring="f1_macro",
                        verbose=0,
                        random_state=ctx.config.model.random_state,
                    )
                    if bayes_model is not None:
                        ctx.best_model_estimator = bayes_model
                        ctx.models[f"{ctx.best_model}_bayesian"] = bayes_model
                        ctx.artifacts["bayesian_best_params"] = bayes_model.get_params()
                        print(f"  Bayesian tuning applied to {ctx.best_model}.")
                    else:
                        # Fallback to RandomizedSearchCV
                        tuned_state = hyperparameter_tuning(
                            train_state,
                            search_type="random",
                            n_iter=ctx.config.model.n_iter_bayesian,
                            cv_folds=ctx.config.model.cv_folds,
                            random_state=ctx.config.model.random_state,
                            n_jobs=ctx.config.model.n_jobs,
                        )
                        best_est = tuned_state.get("best_estimator")
                        if best_est is not None:
                            ctx.best_model_estimator = best_est
                        ctx.artifacts["tuning_results"] = tuned_state.get("tuning_results", {})
            except Exception as e:
                logger.warning("Bayesian tuning failed (%s) — using default best model.", e)


def _get_param_space(model_prefix: str) -> Dict[str, Any]:
    """Return Bayesian search parameter space for the given model prefix."""
    spaces: Dict[str, Dict] = {
        "xgb": {
            "n_estimators":     ("int",   100, 500),
            "max_depth":        ("int",   3,   9),
            "learning_rate":    ("float", 0.01, 0.3, True),
            "subsample":        ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_alpha":        ("float", 0.0, 1.0),
            "reg_lambda":       ("float", 0.5, 5.0),
        },
        "lgb": {
            "n_estimators":     ("int",   100, 500),
            "max_depth":        ("int",   3,   9),
            "learning_rate":    ("float", 0.01, 0.3, True),
            "num_leaves":       ("int",   20,  150),
            "subsample":        ("float", 0.5, 1.0),
            "reg_alpha":        ("float", 0.0, 1.0),
            "reg_lambda":       ("float", 0.0, 5.0),
        },
        "rf": {
            "n_estimators":     ("int",   100, 500),
            "max_depth":        ("int",   3,   20),
            "min_samples_split":("int",   2,   20),
            "max_features":     ("cat",   ["sqrt", "log2"]),
        },
        "gb": {
            "n_estimators":     ("int",   100, 400),
            "max_depth":        ("int",   2,   7),
            "learning_rate":    ("float", 0.01, 0.3, True),
            "subsample":        ("float", 0.6, 1.0),
        },
        "lr": {
            "C":                ("float", 0.001, 10.0, True),
            "max_iter":         ("int",   200, 2000),
        },
        "svm": {
            "C":                ("float", 0.01, 100.0, True),
            "gamma":            ("cat",   ["scale", "auto"]),
        },
    }
    return spaces.get(model_prefix, {})


@registry.register
class EvaluateStep(PipelineStep):
    """
    Step 11 — Complete evaluation: metrics, learning curves, fairness, model card.

    Input  : ctx.best_model_estimator, ctx.X_test, ctx.y_test,
             ctx.X_train, ctx.y_train
    Output : ctx.final_metrics, ctx.importance_df, ctx.model_card,
             ctx.fairness_results, ctx.monitoring_df
    """

    name = "evaluate"
    requires = ["X_test", "y_test"]

    def skip_if(self, ctx: PipelineContext) -> bool:
        return ctx.best_model_estimator is None

    def run(self, ctx: PipelineContext) -> None:
        from ml_framework.evaluation.evaluation import evaluate_model
        from ml_framework.evaluation.feature_importance import (
            create_feature_importance_summary,
            permutation_feature_importance,
        )
        from ml_framework.evaluation.overfiting_check import (
            check_overfitting,
            compute_learning_curves,
        )
        from ml_framework.evaluation.fairness import fairness_audit
        from ml_framework.evaluation.clinical_report import medical_model_report
        from ml_framework.monitoring.performance_tracking import (
            track_model_performance_over_time,
        )
        from ml_framework.interpretability.model_explainability import (
            interpret_model_with_shap,
        )
        from ml_framework.modeling.model_card import generate_model_card, print_model_card
        from ml_framework.services.decision_support import batch_decision_support

        model = ctx.best_model_estimator
        X_test = ctx.X_test
        y_test = ctx.y_test
        X_train = ctx.X_train
        y_train = ctx.y_train

        # ── Full evaluation metrics ────────────────────────────────────────────
        metrics = evaluate_model(
            model, X_test, y_test,
            X_train=X_train, y_train=y_train,
            threshold=ctx.config.model.threshold,
        )
        ctx.final_metrics = metrics
        print(f"  Test F1-macro  : {metrics.get('f1_macro', metrics.get('f1', 'N/A'))}")
        print(f"  Test Accuracy  : {metrics.get('accuracy', 'N/A')}")

        # ── Overfitting check + learning curves ───────────────────────────────
        try:
            test_acc = metrics.get("accuracy", 0.0)
            overfitting_report = check_overfitting(
                model=model,
                X_train=X_train,
                y_train=y_train,
                test_acc=test_acc,
                threshold=0.10,
            )
            ctx.artifacts["overfitting_report"] = overfitting_report
            severity = overfitting_report.get("severity", "unknown")
            print(f"  Overfitting    : {severity}")
        except Exception as e:
            logger.warning("check_overfitting: %s", e)

        try:
            lc = compute_learning_curves(
                model=model, X=X_train, y=y_train,
                cv=3, scoring="f1_macro", n_jobs=ctx.config.model.n_jobs,
            )
            ctx.artifacts["learning_curves"] = lc
        except Exception as e:
            logger.warning("compute_learning_curves: %s", e)

        # ── Feature importance ────────────────────────────────────────────────
        try:
            imp_df = create_feature_importance_summary(
                model,
                X_test.columns.tolist(),
                top_n=ctx.config.report.top_n_features,
                plot=True,
            )
            ctx.importance_df = imp_df
        except Exception as e:
            logger.warning("create_feature_importance_summary: %s", e)

        try:
            perm_imp = permutation_feature_importance(
                model, X_test, y_test,
                scoring="f1_macro",
                top_n=ctx.config.report.top_n_features,
            )
            ctx.artifacts["permutation_importance"] = perm_imp
        except Exception as e:
            logger.warning("permutation_feature_importance: %s", e)

        # ── SHAP ──────────────────────────────────────────────────────────────
        try:
            interpret_model_with_shap(
                model, X_test,
                feature_names=X_test.columns.tolist(),
                max_display=15,
                plot_type="summary",
                n_dependence_plots=3,
            )
        except Exception as e:
            logger.warning("SHAP: %s", e)

        # ── Fairness audit ────────────────────────────────────────────────────
        try:
            sensitive_attrs = [
                c for c in ctx.config.sensitive_attributes
                if c in X_test.columns
            ]
            if not sensitive_attrs:
                # Fallback: use Age bins if Age column exists
                if "Age" in X_test.columns:
                    X_test_audit = X_test.copy()
                    X_test_audit["Age_group"] = pd.cut(
                        X_test_audit["Age"],
                        bins=[0, 40, 60, 75, 120],
                        labels=["<40", "40-60", "60-75", ">75"],
                    ).astype(str)
                    sensitive_attrs = ["Age_group"]
                else:
                    sensitive_attrs = None

            if sensitive_attrs:
                ctx.fairness_results = fairness_audit(
                    model=model,
                    X_test=X_test_audit if "X_test_audit" in dir() else X_test,
                    y_test=y_test,
                    sensitive_attributes=sensitive_attrs,
                    disparity_threshold=0.20,
                    verbose=True,
                )
        except Exception as e:
            logger.warning("fairness_audit: %s", e)

        # ── Clinical medical report ────────────────────────────────────────────
        try:
            clinical_df = medical_model_report(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_names=X_test.columns.tolist(),
                generate_profiles=True,
                verbose=True,
            )
            ctx.artifacts["clinical_report"] = clinical_df
        except Exception as e:
            logger.warning("medical_model_report: %s", e)

        # ── Clinical decision support (batch) ──────────────────────────────────
        try:
            decision_df = batch_decision_support(
                model=model,
                X=X_test,
                feature_names=X_test.columns.tolist(),
                target_condition=ctx.target_column,
            )
            ctx.artifacts["decision_support"] = decision_df
        except Exception as e:
            logger.warning("batch_decision_support: %s", e)

        # ── Production monitoring baseline ────────────────────────────────────
        try:
            import os
            os.makedirs("artifacts", exist_ok=True)
            ctx.monitoring_df = track_model_performance_over_time(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_id=ctx.best_model or "pipeline_model",
                metrics=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
                storage_path="artifacts/monitoring_history.csv",
                alert_threshold=0.05,
                verbose=True,
            )
        except Exception as e:
            logger.warning("track_model_performance_over_time: %s", e)

        # ── Model Card (Google 2019 standard) ─────────────────────────────────
        try:
            card = generate_model_card(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=f"{ctx.best_model} — Medical Pipeline",
                target_description=ctx.target_column,
                limitations=[
                    "Trained on synthetic data — validate on real clinical cohorts.",
                    "SurvivalMonths may constitute leakage in prospective scenarios.",
                    "Not cleared for independent clinical deployment.",
                ],
                version="1.0",
            )
            ctx.model_card = card
            print_model_card(card)
        except Exception as e:
            logger.warning("generate_model_card: %s", e)


@registry.register
class PersistStep(PipelineStep):
    """
    Step 12 — 

    Input  : ctx.best_model_estimator, ctx.scaler_strategies,
             ctx.transform_log, ctx.encoders, ctx.model_card,
             ctx.significant_features
    Output : artifacts/ directory populated; ctx.artifacts['saved_files']
    """

    name = "persist"
    requires = ["best_model_estimator"]

    def run(self, ctx: PipelineContext) -> None:
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
                logger.warning("Could not save '%s': %s", filename, e)

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


# ══════════════════════════════════════════════════════════════════════════════
# 6. MEDICAL ML PIPELINE — public facade
# ══════════════════════════════════════════════════════════════════════════════


class MedicalMLPipeline:
    """
    Public facade for the medical MLOps pipeline.

    Configures the PipelineRunner with the step registry,
    exposes a simple API, and retains the context after execution.

    Parameters
    ----------
    config    : FrameworkConfig — global configuration (default: standard values)
    n_retries : number of automatic retries per step on error (default 0)

    Usage
    -----
        pipeline = MedicalMLPipeline()
        pipeline.run("data.csv", target_column="TreatmentResponse")

        # Predict on new patients
        predictions = pipeline.predict(new_df)

        # Access results
        model   = pipeline.context.best_model_estimator
        card    = pipeline.context.model_card
        metrics = pipeline.context.final_metrics
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
    ) -> None:
        self.config = config or FrameworkConfig()
        self.runner = PipelineRunner(
            registry=registry,
            n_retries=n_retries,
            stop_on_error=True,
        )
        self.context: Optional[PipelineContext] = None
        logger.info("MedicalMLPipeline initialized.")

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
        self.context = self.runner.run(ctx, steps=steps or self.ALL_STEPS)
        return self

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        steps: Optional[List[str]] = None,
    ) -> "MedicalMLPipeline":
        """
        Execute the pipeline directly from a DataFrame .

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
        self.context = self.runner.run(ctx, steps=steps or default_steps)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Predict on new data using the fitted pipeline.

        Applies the exact same preprocessing chain used during training:

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

        # ── Re-apply encoding ─────────────────────────────────────────────────
        if ctx.encoders:
            try:
                from ml_framework.preprocessing.encoding import encode_dataframe
                X_proc, _ = encode_dataframe(X_proc, verbose=False)
            except Exception as e:
                logger.warning("predict: encoding failed (%s) — using raw features.", e)

        # ── Re-apply normalization (same strategies, no re-fit) ───────────────
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

        # ── Align feature columns ─────────────────────────────────────────────
        if ctx.X_train is not None:
            expected_cols = ctx.X_train.columns.tolist()
            for col in expected_cols:
                if col not in X_proc.columns:
                    X_proc[col] = 0
            X_proc = X_proc[expected_cols]

        if return_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(X_proc)
        return model.predict(X_proc)

    def add_hook(
        self,
        step_name: str,
        when: str,
        fn: Callable[[PipelineContext], None],
    ) -> "MedicalMLPipeline":
        """Add a before/after hook to a step."""
        registry.add_hook(step_name, when, fn)
        return self

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
