"""
clinical_profiles.py — Backward-compatible re-export.

This module's actual implementation moved to
ml_framework.domain.medical.clinical_profiles. The mechanics here are
domain-neutral (any sklearn-like model + feature_names works) — it moved
alongside the rest of the medical domain pack simply because its naming
and framing (patient risk profiles) is clinical. This module re-exports the
same public names so existing imports keep working unchanged.

For new code, prefer importing from ml_framework.domain.medical directly.
"""

from ml_framework.domain.medical.clinical_profiles import (
    generate_patient_risk_profiles,
    compare_profiles_heatmap,
    profile_to_dataframe,
)

__all__ = [
    "generate_patient_risk_profiles",
    "compare_profiles_heatmap",
    "profile_to_dataframe",
]
