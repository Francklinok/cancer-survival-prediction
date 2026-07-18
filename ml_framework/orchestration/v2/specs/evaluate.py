"""
specs/evaluate.py — ModuleSpec for full model evaluation.

Business logic untouched: wraps evaluate_model, overfitting checks, feature
importance, SHAP, fairness audit, clinical report, monitoring baseline, and
model card generation verbatim, exactly as EvaluateStep.run() does in
orchestration/pipeline.py.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml_framework.orchestration.pipeline import PipelineContext
from ml_framework.orchestration.v2.contracts import ModuleSpec

logger = logging.getLogger("ml_framework.orchestration.v2.specs.evaluate")


def invoke(ctx: PipelineContext) -> None:
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

    metrics = evaluate_model(
        model, X_test, y_test,
        X_train=X_train, y_train=y_train,
        threshold=ctx.config.model.threshold,
    )
    ctx.final_metrics = metrics
    print(f"  Test F1-macro  : {metrics.get('f1_macro', metrics.get('f1', 'N/A'))}")
    print(f"  Test Accuracy  : {metrics.get('accuracy', 'N/A')}")

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

    try:
        sensitive_attrs = [
            c for c in ctx.config.sensitive_attributes
            if c in X_test.columns
        ]
        if not sensitive_attrs:
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
            ],
            version="1.0",
        )
        ctx.model_card = card
        print_model_card(card)
    except Exception as e:
        logger.warning("generate_model_card: %s", e)


SPEC = ModuleSpec(
    name="evaluate",
    version="1.0.0",
    capabilities=frozenset({"tabular"}),
    inputs=frozenset({"best_model_estimator", "X_test", "y_test"}),
    outputs=frozenset({
        "final_metrics", "importance_df", "model_card",
        "fairness_results", "monitoring_df",
    }),
    invoke=invoke,
    cost_hint="high", 
    parallelizable=False,
)
