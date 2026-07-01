"""
model.py — Backward-compatibility shim. DEPRECATED.

This module was renamed to model_registry.py.
Update all imports to:
    from ml_framework.modeling.model_registry import get_models, get_model, list_models
"""
import warnings
warnings.warn(
    "ml_framework.modeling.model is deprecated. "
    "Use ml_framework.modeling.model_registry instead.",
    DeprecationWarning,
    stacklevel=2,
)
from ml_framework.modeling.model_registry import get_models, get_model, list_models  # noqa: F401, E402

__all__ = ["get_models", "get_model", "list_models"]
