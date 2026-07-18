"""
specs/class_imbalance_adapter.py — Adapter translating diagnose_class_imbalance()
output into a Recommendation.

Source: ml_framework.diagnostic.class_imbalance.diagnose_class_imbalance
Raw shape (confirmed by audit): dict with keys severity, imbalance_ratio,
majority_class, minority_class, recommendations (list[str], free text).

No re-analysis happens here — this only maps fields that diagnose_class_imbalance
already computed to the common Recommendation contract. Severity thresholds
("balanced" < 1.5, "mild" < 3, "moderate" < 10, "severe" < 50, else "extreme")
are owned by diagnose_class_imbalance itself and are NOT duplicated here.
"""

from __future__ import annotations

from ml_framework.orchestration.v2.contracts import Recommendation

_SMOTE_SEVERITIES = {"severe", "extreme"}
_WEIGHT_SEVERITIES = {"moderate"}
_NOT_REQUIRED_SEVERITIES = {"balanced", "mild"}


def adapt(raw_output: dict) -> Recommendation:
    severity = str(raw_output.get("severity", "unknown")).lower()
    ratio = raw_output.get("imbalance_ratio")

    if severity in _SMOTE_SEVERITIES:
        strategy = "SMOTE"
    elif severity in _WEIGHT_SEVERITIES:
        strategy = "class_weight"
    else:
        strategy = None

    return Recommendation(
        topic="balancing",
        required=severity not in _NOT_REQUIRED_SEVERITIES,
        strategy=strategy,
        params={"imbalance_ratio": ratio} if ratio is not None else {},
        reason=f"severity={severity}" + (f", ratio={ratio}" if ratio is not None else ""),
        confidence=1.0,
        source_module="diagnose_class_imbalance",
        raw=raw_output,
    )
