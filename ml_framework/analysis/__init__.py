# ml_framework/analysis package

from ml_framework.analysis.eda import (
    run_eda,
    plot_distributions,
    explore_categorical_distributions,
    box_plot,
    violin_plot,
    analyze_target,
    bivariate_analysis,
    pattern_anomaly_analysis,
    group_effect_analysis,
    residual_anomaly_analysis,
)
from ml_framework.analysis.statistical_analysis import (
    run_statistical_analysis,
    normality_tests,
    compare_groups,
    correlation_analysis,
    categorical_association_tests,
    variance_analysis,
    multicollinearity_analysis,
    effect_size_analysis,
    feature_association_tests,
)
from ml_framework.analysis.correlation_matrix import (
    plot_correlation_matrix,
    compare_correlation_methods,
    compute_vif,
)

# ── Diagnostic Analysis ───────────────────────────────────────────────────────
from ml_framework.analysis.diagnostic_analysis import (
    ci_str,
    # First-pass linear ranking
    diagnostic_analysis,
    # Root drivers
    root_cause_analysis,
    contribution_analysis,
    # SHAP global (model-agnostic)
    shap_global_analysis,
    # Segment & cohort profiling
    segment_analysis,
    cohort_analysis,
    # Variance & anomaly
    variance_decomposition,
    anomaly_explanation,
    # Causal — observational
    causal_analysis,
    # Causal — bias-corrected
    propensity_score_matching,
    difference_in_differences,
    regression_discontinuity,
    causal_forest_ate,
    double_machine_learning,
)

# ── Data Drift ────────────────────────────────────────────────────────────────
from ml_framework.analysis.data_drift import (
    # Feature drift
    detect_data_drift,
    compute_psi,
    compute_kl_divergence,
    # Target drift
    detect_target_drift,
    # Concept drift
    concept_drift_indicators,
    # Population stability
    population_stability_report,
    monitor_population_stability,
)

# ── Class imbalance ───────────────────────────────────────────────────────────
from ml_framework.diagnostic.class_imbalance import (
    diagnose_class_imbalance,
    rebalance_classes,
)

__all__ = [
    # ── EDA ───────────────────────────────────────────────────────────────────
    "run_eda",
    "plot_distributions",
    "explore_categorical_distributions",
    "box_plot",
    "violin_plot",
    "analyze_target",
    "bivariate_analysis",
    "pattern_anomaly_analysis",
    "group_effect_analysis",
    "residual_anomaly_analysis",
    # ── Statistical Analysis ──────────────────────────────────────────────────
    "run_statistical_analysis",
    "normality_tests",
    "compare_groups",
    "correlation_analysis",
    "categorical_association_tests",
    "variance_analysis",
    "multicollinearity_analysis",
    "effect_size_analysis",
    "feature_association_tests",
    # ── Correlation matrix ────────────────────────────────────────────────────
    "plot_correlation_matrix",
    "compare_correlation_methods",
    "compute_vif",
    # ── Diagnostic Analysis ───────────────────────────────────────────────────
    "ci_str",
    "diagnostic_analysis",
    "root_cause_analysis",
    "contribution_analysis",
    "shap_global_analysis",
    "segment_analysis",
    "cohort_analysis",
    "variance_decomposition",
    "anomaly_explanation",
    # ── Causal Analysis ───────────────────────────────────────────────────────
    "causal_analysis",
    "propensity_score_matching",
    "difference_in_differences",
    "regression_discontinuity",
    "causal_forest_ate",
    "double_machine_learning",
    # ── Data Drift ────────────────────────────────────────────────────────────
    "detect_data_drift",
    "compute_psi",
    "compute_kl_divergence",
    "detect_target_drift",
    "concept_drift_indicators",
    "population_stability_report",
    "monitor_population_stability",
    # ── Class imbalance ───────────────────────────────────────────────────────
    "diagnose_class_imbalance",
    "rebalance_classes",
]
