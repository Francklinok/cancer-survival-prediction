"""
Import from ml_framework.analysis.class_imbalance instead.
"""

from ml_framework.diagnostic.class_imbalance import (  # noqa
    diagnose_class_imbalance,
    rebalance_classes,
)
import  warnings

warnings.warn(
    "ml_framework.analysis.class_imbalance_analysis is deprecated and will be removed in a future version. "
    "Please use ml_framework.diagnostic.class_imbalance instead.",
     DeprecationWarning,
     stackleve = 2
)

# Legacy alias
analyze_class_balance = diagnose_class_imbalance

__all__ = [
    "diagnose_class_imbalance",
    "rebalance_classes",
    "analyze_class_balance",
]
