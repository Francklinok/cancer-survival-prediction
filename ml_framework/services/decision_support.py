"""
decision_support.py — Backward-compatible re-export.

This module's actual implementation moved to
ml_framework.domain.medical.decision_support, since its recommendation
vocabulary ("specialist consultation", "follow-up", target_condition
phrasing) is clinical triage-specific rather than a generic pattern — see
the framework's domain/medical package for the full rationale. This module
re-exports the same public names so existing imports keep working
unchanged.

For new code, prefer importing from ml_framework.domain.medical directly.
"""

from ml_framework.domain.medical.decision_support import (
    batch_decision_support,
    create_clinical_decision_support,
)

__all__ = ["batch_decision_support", "create_clinical_decision_support"]
