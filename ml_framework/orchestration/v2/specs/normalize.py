"""
specs/normalize.py — ModuleSpec for zero-leakage normalization.

Business logic untouched: column analysis, strategy suggestion, and
normalization application are wrapped verbatim from NormalizeStep.run() in
orchestration/pipeline.py. The one deliberate change from that step: the
train/test split now goes through ml_framework.core.dataset_split.split_train_test
instead of an inline sklearn.train_test_split call — the same DRY fix
identified in the architecture audit (core/dataset_split.py already existed
but pipeline.py never called it). split_train_test's stratify-fallback
behavior was extended to match the inline version's try/except exactly
(see core/dataset_split.py) so this is a pure move, not a behavior change.

Critical ordering preserved:
  1. Split train/test BEFORE any fitting
  2. Fit transformers on X_train only
  3. Apply fitted transformers to X_test (no re-fit)
"""

from __future__ import annotations

import logging

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec, Recommendation

logger = logging.getLogger("ml_framework.orchestration.v2.specs.normalize")


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.analysis.column_analysis import analyze_column_properties
    from ml_framework.strategies.normalisation_strategy import suggest_normalization_strategy
    from ml_framework.preprocessing.apply_normalisation import apply_normalization
    from ml_framework.evaluation.normalization_quality import evaluate_normalization_quality
    from ml_framework.core.dataset_split import split_train_test

    df = ctx.df_work
    target = ctx.target_column
    cfg_norm = ctx.config.normalization

    if not target or target not in df.columns:
        logger.warning("normalize: target '%s' not in df — normalizing full df.", target)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        anal_df = analyze_column_properties(df[num_cols], verbose=False)
        strats_df = suggest_normalization_strategy(anal_df, cfg_norm)
        _, df_norm, t_log = apply_normalization(df, strats_df, cfg_norm)
        ctx.df_normalized = df_norm
        ctx.df_work = df_norm
        ctx.scaler_strategies = strats_df
        ctx.transform_log = t_log
        return

    # ── 1. Split BEFORE normalization (via core/dataset_split, not inline) ──
    X_tr, X_te, y_tr, y_te = split_train_test(
        df,
        target=target,
        test_size=ctx.config.model.test_size,
        val_size=0.0,
        stratify=True,
        random_state=ctx.config.model.random_state,
    )

    ctx.X_train_raw = X_tr.copy()
    ctx.y_train = y_tr
    ctx.y_test = y_te

    # ── 2. Fit strategy on X_train only ──────────────────────────────────────
    num_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()
    anal_df = analyze_column_properties(X_tr[num_cols] if num_cols else X_tr, verbose=False)
    strats_df = suggest_normalization_strategy(anal_df, cfg_norm)
    ctx.scaler_strategies = strats_df

    # ── 3. Apply to train (fit+transform) ────────────────────────────────────
    X_tr_orig, X_tr_norm, t_log = apply_normalization(X_tr, strats_df, cfg_norm)
    ctx.transform_log = t_log

    # ── 4. Apply to test (transform only — same strategies, no re-fit) ──────
    _, X_te_norm, _ = apply_normalization(X_te, strats_df, cfg_norm)

    ctx.X_train = X_tr_norm
    ctx.X_test = X_te_norm

    # ── 5. Normalization quality evaluation ──────────────────────────────────
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

    df_full_norm = X_tr_norm.copy()
    df_full_norm[target] = y_tr.values[:len(df_full_norm)]
    ctx.df_normalized = df_full_norm
    ctx.df_work = df_full_norm

    print(f"  Train: {X_tr_norm.shape}  |  Test: {X_te_norm.shape}  (fit on train only)")


def adapt(raw_output) -> Recommendation:
    """
    raw_output: the strategies DataFrame returned by suggest_normalization_strategy()
    itself — columns [column, transform, scaler, winsorize, reason] (confirmed by
    audit).
    """
    if raw_output is None or raw_output.empty:
        return Recommendation(
            topic="scaling",
            required=False,
            reason="no columns to normalize",
            confidence=1.0,
            source_module="suggest_normalization_strategy",
            raw=raw_output,
        )

    active = raw_output[raw_output["scaler"].notna() | raw_output["transform"].notna()] \
        if "scaler" in raw_output.columns else raw_output
    scaler_counts = active["scaler"].value_counts().to_dict() if "scaler" in active.columns else {}
    dominant_scaler = max(scaler_counts, key=scaler_counts.get) if scaler_counts else None

    return Recommendation(
        topic="scaling",
        required=len(active) > 0,
        strategy=dominant_scaler,
        params={"scaler_counts": scaler_counts, "n_columns": len(active)},
        reason=f"{len(active)}/{len(raw_output)} column(s) require scaling/transform",
        confidence=1.0,
        source_module="suggest_normalization_strategy",
        raw=raw_output,
    )


SPEC = ModuleSpec(
    name="normalize",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({
        "df_work", "df_normalized", "X_train", "X_test", "y_train", "y_test",
        "X_train_raw", "scaler_strategies", "transform_log", "normalization_result",
    }),
    invoke=invoke,
    produces_recommendation=True,
    adapter=adapt,
    cost_hint="medium",
)
