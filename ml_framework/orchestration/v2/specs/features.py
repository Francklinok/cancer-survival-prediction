"""
specs/features.py — ModuleSpec for feature engineering + statistical selection.

Business logic untouched: wraps engineer_features() and
statistical_feature_selection() verbatim, exactly as
FeatureEngineeringStep.run() does in orchestration/pipeline.py.
"""

from __future__ import annotations

import logging

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec

logger = logging.getLogger("ml_framework.orchestration.v2.specs.features")


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.features.feature_engineering import engineer_features
    from ml_framework.features.statistical_feature_selection import (
        statistical_feature_selection,
    )

    df = ctx.df_work
    target = ctx.target_column

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

    feat_cols = [c for c in sig_feats if c != target]
    if ctx.X_train is not None:
        train_feat = [c for c in feat_cols if c in ctx.X_train.columns]
        if train_feat:
            ctx.X_train = ctx.X_train[train_feat]
            ctx.X_test = ctx.X_test[[c for c in train_feat if c in ctx.X_test.columns]]

    print(f"  {len(sig_feats)} features selected.")


SPEC = ModuleSpec(
    name="features",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({
        "df_work", "final_dataset", "new_features",
        "significant_features", "selection_results", "X_train", "X_test",
    }),
    invoke=invoke,
    cost_hint="high", 
)
