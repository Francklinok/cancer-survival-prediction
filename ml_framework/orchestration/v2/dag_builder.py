"""
dag_builder.py — Turns an ExecutionPlan into an executable dependency graph.

The DAG Builder NEVER decides whether a module should run — that was
already settled by the Decision Engine (decision_engine.py) when it produced
the ExecutionPlan. This module's only job is to figure out the correct
execution ORDER and PARALLELISM by cross-referencing each included module's
declared ModuleSpec.inputs / ModuleSpec.outputs.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple

from ml_framework.orchestration.v2.contracts import ExecutionPlan
from ml_framework.orchestration.v2.module_registry import ModuleRegistry


class CycleDetectedError(Exception):
    """Raised when the module graph is not a DAG."""


class MissingInputError(Exception):
    """Raised when a module's declared input is produced by no upstream module
    and is not present in the initial artifact set."""


@dataclass(frozen=True)
class DAGNode:
    module_name: str
    inputs: FrozenSet[str]
    outputs: FrozenSet[str]
    parallelizable: bool


@dataclass(frozen=True)
class ExecutionDAG:
    """
    nodes  : all DAGNode instances, keyed by module_name
    levels : execution order — levels[0] has no dependencies among nodes in
             this DAG, levels[1] depends only on levels[0], etc. 
    """

    nodes: Dict[str, DAGNode]
    levels: Tuple[Tuple[str, ...], ...]

    def is_parallel_level(self, level_index: int) -> bool:
        return all(self.nodes[name].parallelizable for name in self.levels[level_index])


def build_dag(
    plan: ExecutionPlan,
    registry: ModuleRegistry,
    initial_artifacts: Set[str],
) -> ExecutionDAG:
    """
    plan               : the Decision Engine's output — which modules to run
    registry           : source of truth for each module's inputs/outputs
    initial_artifacts  : artifact names already available before any module
                          runs (e.g. {"file_path", "target_column"} supplied
                          by the caller) — NOT produced by any ModuleSpec

    Raises MissingInputError if a module's input is neither produced
    upstream nor present in initial_artifacts. Raises CycleDetectedError if
    the resulting dependency graph is not acyclic.

    """
    included_names = [inv.module_name for inv in plan.invocations]
    nodes: Dict[str, DAGNode] = {}
    for name in included_names:
        spec = registry.get(name)
        nodes[name] = DAGNode(
            module_name=name,
            inputs=spec.inputs,
            outputs=spec.outputs,
            parallelizable=spec.parallelizable,
        )

    # Map each artifact name to every module (within this plan) that writes
    # it
    writers_of: Dict[str, List[str]] = {}
    for name in included_names:
        for output in nodes[name].outputs:
            writers_of.setdefault(output, []).append(name)

    dependencies: Dict[str, Set[str]] = {name: set() for name in nodes}

    # Chain co-writers of the same artifact in plan order (in-place mutation).
    for artifact, writers in writers_of.items():
        for earlier, later in zip(writers, writers[1:]):
            dependencies[later].add(earlier)

    plan_index = {name: i for i, name in enumerate(included_names)}
    for name, node in nodes.items():
        for required_input in node.inputs:
            if required_input in initial_artifacts:
                continue
            all_writers = writers_of.get(required_input, [])
            if not all_writers:
                raise MissingInputError(
                    f"Module '{name}' requires '{required_input}', which is produced by "
                    f"no included module and is not in initial_artifacts."
                )
            candidates = [w for w in all_writers if plan_index[w] < plan_index[name]]
            if not candidates:
                # Every module that actually writes this artifact runs *after* (or at the same time as) `name`.
                # This means `name` would try to read data that doesn't exist yet (or grab stale data 
                # from a past run). This is a textbook cyclic dependency. Instead of silently ignoring 
                # this impossible read, we let the topological sort catch it and throw a CycleDetectedError.
                dependencies[name].add(all_writers[-1])
                continue
            producer = max(candidates, key=lambda w: plan_index[w])
            if producer != name:
                dependencies[name].add(producer)

    levels = _topological_levels(dependencies)
    return ExecutionDAG(nodes=nodes, levels=levels)


def _topological_levels(dependencies: Dict[str, Set[str]]) -> Tuple[Tuple[str, ...], ...]:
    """
    We use Kahn's algorithm but group nodes into levels that can run in parallel.
    A node is placed into a level as soon as all its dependencies are resolved 
    by previous levels. Within each level, we keep the plan's original order 
    so execution remains 100% deterministic.
    """
    remaining = {name: set(deps) for name, deps in dependencies.items()}
    resolved: Set[str] = set()
    levels: List[Tuple[str, ...]] = []
    original_order = list(dependencies.keys())

    while remaining:
        ready = [name for name, deps in remaining.items() if deps <= resolved]
        if not ready:
            raise CycleDetectedError(
                f"Cycle detected among modules: {sorted(remaining.keys())}"
            )
        ready_ordered = [name for name in original_order if name in ready]
        levels.append(tuple(ready_ordered))
        resolved.update(ready_ordered)
        for name in ready_ordered:
            del remaining[name]

    return tuple(levels)
