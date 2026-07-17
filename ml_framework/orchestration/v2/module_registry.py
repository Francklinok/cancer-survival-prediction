"""
module_registry.py — Central, declarative registry of ModuleSpec instances.

Adding a new module = write its specs/<name>.py (invoke + optional adapter +
SPEC) and add one import line below. No other file changes required.
"""

from __future__ import annotations

from typing import Dict, List

from ml_framework.orchestration.v2.contracts import ModuleSpec

from ml_framework.orchestration.v2.specs import (
    ingest,
    profile,
    clean,
    eda,
    missing,
    outliers,
    encode,
    normalize,
    features,
    train,
    evaluate,
    persist,
)


class ModuleRegistry:
    """
    This is a simple, read-only registry to find modules by name. It's built once 
    when imported and acts as the central source of truth. This prevents the Analysis,
    Decision, and DAG engines from importing business modules directly
    """

    def __init__(self, specs: List[ModuleSpec]) -> None:
        self._specs: Dict[str, ModuleSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"Duplicate ModuleSpec name: '{spec.name}'")
            self._specs[spec.name] = spec

    def get(self, name: str) -> ModuleSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown module: '{name}'. Available: {self.names()}")
        return self._specs[name]

    def has(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def all(self) -> List[ModuleSpec]:
        return list(self._specs.values())

    def with_capability(self, capability: str) -> List[ModuleSpec]:
        return [s for s in self._specs.values() if capability in s.capabilities]

    def recommendation_producers(self) -> List[ModuleSpec]:
        return [s for s in self._specs.values() if s.produces_recommendation]


_ALL_SPECS = [
    ingest.SPEC,
    profile.SPEC,
    clean.SPEC,
    eda.SPEC,
    missing.SPEC,
    outliers.SPEC,
    encode.SPEC,
    normalize.SPEC,
    features.SPEC,
    train.SPEC,
    evaluate.SPEC,
    persist.SPEC,
]


default_registry = ModuleRegistry(_ALL_SPECS)
