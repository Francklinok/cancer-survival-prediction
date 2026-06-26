"""
normalisation.py — Normalization convenience facade for the features package.

Re-exports the main normalization functions from their canonical locations.

Recommended imports:
    from ml_framework.features.normalisation import run_normalization_pipeline
    from ml_framework.features.normalisation import suggest_normalization_strategy
"""

from ml_framework.strategies.normalisation_strategy import (
    decide_strategy,
    suggest_normalization_strategy,
    validate_input,
)
from ml_framework.preprocessing.apply_normalisation import (
    apply_normalization,
    _apply_transform,
    _apply_scaler,
)
from ml_framework.orchestration.normalization_pipeline import run_normalization_pipeline

__all__ = [
    "suggest_normalization_strategy",
    "decide_strategy",
    "validate_input",
    "apply_normalization",
    "run_normalization_pipeline",
    "_apply_transform",
    "_apply_scaler",
]
