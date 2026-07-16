"""
decision_engine.py — Pipeline Planner: turns Recommendations into an ExecutionPlan.

"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ml_framework.orchestration.v2.contracts import (
    DatasetProfile,
    ExecutionPlan,
    ModuleInvocation,
    Recommendation,
)
from ml_framework.orchestration.v2.planning_rules import PlanningRule, build_default_rules

logger = logging.getLogger("ml_framework.orchestration.v2.decision_engine")


class PipelinePlanner:
    """
    Aggregates PlanningRule verdicts into an ExecutionPlan.

    plan(recommendations, profile, module_order) -> ExecutionPlan

    """

    def __init__(self, rules: Optional[List[PlanningRule]] = None) -> None:
        self.rules = rules if rules is not None else build_default_rules()
        self._rules_by_module: Dict[str, PlanningRule] = {r.module_name: r for r in self.rules}

    def plan(
        self,
        recommendations: List[Recommendation],
        profile: DatasetProfile,
        module_order: List[str],
    ) -> ExecutionPlan:
        rec_by_topic = {rec.topic: rec for rec in recommendations}

        invocations: List[ModuleInvocation] = []
        skipped: List[str] = []

        for module_name in module_order:
            rule = self._rules_by_module.get(module_name)
            if rule is None:
                logger.warning(
                    "No PlanningRule for module '%s' — including by default.", module_name
                )
                invocations.append(ModuleInvocation(module_name=module_name))
                continue

            verdict = rule.evaluate(rec_by_topic, profile)
            if verdict.include:
                invocations.append(
                    ModuleInvocation(module_name=module_name, triggered_by=verdict.triggered_by)
                )
            else:
                skipped.append(module_name)
                logger.info("Planner: skipping '%s' — %s", module_name, verdict.reason)

        return ExecutionPlan(
            invocations=tuple(invocations),
            dataset_profile=profile,
            recommendations=tuple(recommendations),
            skipped=tuple(skipped),
        )
