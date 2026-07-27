"""
clinical_report.py — Backward-compatible re-export.

This module's actual implementation moved to
ml_framework.domain.medical.clinical_report, alongside the rest of the
framework's optional medical/clinical domain pack. This module now simply
re-exports the same public names so existing imports keep working
unchanged: ``from ml_framework.evaluation.clinical_report import
medical_model_report`` still works exactly as before.

For new code, prefer importing from ml_framework.domain.medical directly.
"""

from ml_framework.domain.medical.clinical_report import (
    DEFAULT_RISK_THRESHOLDS,
    medical_model_report,
)

__all__ = ["DEFAULT_RISK_THRESHOLDS", "medical_model_report"]
