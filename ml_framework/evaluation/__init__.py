# ml_framework/evaluation package

from ml_framework.evaluation.evaluation import (
    evaluate_model,
    compute_base_metrics,
    check_overfitting,
)
from ml_framework.evaluation.feature_importance import (
    create_feature_importance_summary,
    permutation_feature_importance,
    compare_importance_methods,
)
from ml_framework.evaluation.overfiting_check import (
    compute_learning_curves,
    compute_validation_curve,
)
from ml_framework.evaluation.fairness import fairness_audit
from ml_framework.evaluation.clinical_report import medical_model_report
from ml_framework.evaluation.normalization_quality import (
    evaluate_normalization_quality,
    print_normalization_report,
)

__all__ = [
    # Core evaluation
    "evaluate_model",
    "compute_base_metrics",
    "check_overfitting",
    # Feature importance
    "create_feature_importance_summary",
    "permutation_feature_importance",
    "compare_importance_methods",
    # Overfitting analysis
    "compute_learning_curves",
    "compute_validation_curve",
    # Fairness
    "fairness_audit",
    # Clinical
    "medical_model_report",
    # Normalization quality
    "evaluate_normalization_quality",
    "print_normalization_report",
]
