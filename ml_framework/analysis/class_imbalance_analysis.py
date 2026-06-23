"""
Import from ml_framework.analysis.class_imbalance instead.
"""
from ml_framework.diagnostic.class_imbalance import (  # noqa
    diagnose_class_imbalance,
    rebalance_classes,
)

# Legacy alias
analyze_class_balance = diagnose_class_imbalance

__all__ = [
    "diagnose_class_imbalance",
    "rebalance_classes",
    "analyze_class_balance",
]
