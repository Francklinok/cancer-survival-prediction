"""
planning_rules.py — Strategy Pattern for pipeline planning.

Each PlanningRule handles ONE decision based on pre-computed Recommendation 
objects. The PipelinePlanner aggregates these verdicts into the final ExecutionPlan.

Rule Families:
1. Always-include: Unconditionally includes standard steps (ingest, clean, etc.) 
   to match legacy pipeline behavior.
2. Recommendation-driven: Dynamically skips steps based on Recommendation.required. 
   Currently, only "missing" is active; outliers/normalize rules exist but are 
   disabled until later phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from ml_framework.orchestration.v2.contracts import DatasetProfile, Recommendation


@dataclass(frozen=True)
class RuleVerdict:
    """
    One rule's decision about one module.

    include        : whether the module should be part of the ExecutionPlan
    reason         : short justification, surfaced in ExecutionPlan.skipped
                     and in audit logs
    triggered_by   : Recommendation.topic that produced this verdict, if any
    """

    module_name: str
    include: bool
    reason: str = ""
    triggered_by: Optional[str] = None


class PlanningRule(ABC):
    """
    Base class for a single planning decision. Subclasses implement
    `evaluate` only — they never touch data, only Recommendation/DatasetProfile.
    """
    module_name: str = ""

    @abstractmethod
    def evaluate(
        self,
        recommendations: Dict[str, Recommendation],
        profile: DatasetProfile,
    ) -> RuleVerdict:
        """recommendations is keyed by Recommendation.topic."""


class AlwaysIncludeRule(PlanningRule):
    """
    Unconditionally includes a module. 

    Applies to all pipeline.py modules that cannot be skipped today (ingest, profile, 
    clean, eda, encode, features, train, evaluate, persist). Using an explicit rule 
    for these modules—rather than hardcoding a default in the planner—keeps the 
    aggregation logic uniform and strictly follows the Strategy Pattern (every module 
    has exactly one rule)
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def evaluate(
        self,
        recommendations: Dict[str, Recommendation],
        profile: DatasetProfile,
    ) -> RuleVerdict:
        return RuleVerdict(module_name=self.module_name, include=True, reason="always included")


class SkipIfNotRequiredRule(PlanningRule):
    """
    Includes a module only if its associated Recommendation has required=True.

    Mirrors legacy skip logic (e.g., skip if n_missing == 0), but reads the verdict 
    directly from the Recommendation instead of recomputing it.

    If no Recommendation is found (e.g., the Analysis Engine didn't run), the rule 
    fails safe and includes the module. Running a step unnecessarily is safer than 
    silently skipping a real preprocessing step due to a missing diagnostic.
    """

    def __init__(self, module_name: str, topic: str) -> None:
        self.module_name = module_name
        self.topic = topic

    def evaluate(
        self,
        recommendations: Dict[str, Recommendation],
        profile: DatasetProfile,
    ) -> RuleVerdict:
        rec = recommendations.get(self.topic)
        if rec is None:
            return RuleVerdict(
                module_name=self.module_name,
                include=True,
                reason=f"no recommendation available for topic '{self.topic}' — including to be safe",
            )
        return RuleVerdict(
            module_name=self.module_name,
            include=rec.required,
            reason=rec.reason,
            triggered_by=rec.topic,
        )


def build_default_rules(enable_adaptive_skipping: bool = False) -> List[PlanningRule]:
    """
    The rule set reproducing pipeline.py's ALL_STEPS behavior exactly when
    enable_adaptive_skipping=False (Phase 2 non-regression default).

    When True, outliers/normalize also become Recommendation-driven — this
    is the Dataset A/B adaptive behavior from the architecture doc, gated
    off until Phase 2/3 are validated end-to-end (architecture doc, Phase 4).
    """
    always = ["ingest", "profile", "clean", "eda", "encode", "features", "train", "evaluate", "persist"]
    rules: List[PlanningRule] = [AlwaysIncludeRule(name) for name in always]

    # "missing" is the one module pipeline.py already skips conditionally today.
    rules.append(SkipIfNotRequiredRule("missing", topic="missing_values"))

    if enable_adaptive_skipping:
        rules.append(SkipIfNotRequiredRule("outliers", topic="outlier_processing"))
        rules.append(SkipIfNotRequiredRule("normalize", topic="scaling"))
    else:
        rules.append(AlwaysIncludeRule("outliers"))
        rules.append(AlwaysIncludeRule("normalize"))

    return rules
