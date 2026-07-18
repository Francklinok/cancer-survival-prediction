"""
specs/leakage_adapter.py — Maps leakage_exploration() outputs to a Recommendation.

Takes the raw DataFrame from `leakage_exploration` and translates it into our 
standard contract. It simply counts the rows already flagged as "HIGH" risk 
by the diagnostic module, without redefining any thresholds here.

Note: `required=True` does NOT mean an automatic fix will run (leakage cannot 
be fixed automatically). Instead, it means this warning MUST be surfaced to 
a human or a report before training. The Decision Engine treats this topic 
as purely informational—it never silently inserts an automatic step.
"""

from __future__ import annotations

import pandas as pd

from ml_framework.orchestration.v2.contracts import Recommendation


def adapt(raw_output: pd.DataFrame) -> Recommendation:
    if raw_output is None or raw_output.empty or "risk" not in raw_output.columns:
        return Recommendation(
            topic="leakage_review",
            required=False,
            reason="no leakage signals detected",
            confidence=1.0,
            source_module="leakage_exploration",
            raw=raw_output,
        )

    high_risk = raw_output[raw_output["risk"] == "HIGH"]
    n_high = len(high_risk)

    return Recommendation(
        topic="leakage_review",
        required=n_high > 0,
        strategy="manual_review" if n_high > 0 else None,
        params={"high_risk_columns": high_risk.index.tolist()} if n_high > 0 else {},
        reason=f"{n_high} high-risk column(s)" if n_high > 0 else "no high-risk columns",
        confidence=1.0,
        source_module="leakage_exploration",
        raw=raw_output,
    )
