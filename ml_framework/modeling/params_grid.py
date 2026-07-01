"""
params_grid.py — Backward-compatibility shim. DEPRECATED.

This module was renamed to hyperparameter_grids.py.
Update all imports to:
    from ml_framework.modeling.hyperparameter_grids import get_param_grid, get_all_param_grids
"""
import warnings
warnings.warn(
    "ml_framework.modeling.params_grid is deprecated. "
    "Use ml_framework.modeling.hyperparameter_grids instead.",
    DeprecationWarning,
    stacklevel=2,
)
from ml_framework.modeling.hyperparameter_grids import (  # noqa: F401, E402
    get_param_grid,
    get_all_param_grids,
)

__all__ = ["get_param_grid", "get_all_param_grids"]
