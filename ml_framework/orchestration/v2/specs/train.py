"""
specs/train.py — ModuleSpec for model training with CV + Bayesian tuning.

Business logic untouched: wraps train_models() + optimize_hyperparameters_bayesian()
verbatim, exactly as TrainStep.run() does in orchestration/pipeline.py
(including the X_test=/y_test= pre-split path fixed in the previous session
to stop passing test_size=0.0 to sklearn's train_test_split).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ml_framework.orchestration.pipeline import PipelineContext, _get_param_space
from ml_framework.orchestration.v2.contracts import ModuleSpec

logger = logging.getLogger("ml_framework.orchestration.v2.specs.train")


def invoke(ctx: PipelineContext) -> None:
    from ml_framework.modeling.model_trainer import train_models, hyperparameter_tuning
    from ml_framework.modeling.model_registry import get_model

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


SPEC = ModuleSpec(
    name="train",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"df_work"}),
    outputs=frozenset({
        "models", "best_model", "best_model_estimator", "model_scores",
        "X_train", "X_test", "y_train", "y_test",
    }),
    invoke=invoke,
    cost_hint="high",  # measured ~28s on the reference dataset
    parallelizable=False,  # CV/Bayesian tuning already uses n_jobs internally
)
