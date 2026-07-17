"""
It executes the ExecutionDAG. 
Like everything else in the engine, it contains zero business logic.

The modules modify a shared context (ctx) in place. The engine bridges 
this old behavior with our new registry: after a module runs,
it automatically copies its declared outputs from ctx into the ArtifactRegistry. 
This keeps the modules untouched while letting us use the registry for caching, 
checkpoints, and events.

"""

from __future__ import annotations

import logging 
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from ml_framework.orchestration.v2.artifact_registry import ArtifactRegistry
from ml_framework.orchestration.v2.cache_store import CacheStore, compute_cache_key
from ml_framework.orchestration.v2.checkpoint_store import CheckpointStore
from ml_framework.orchestration.v2.contracts import Event, EventType
from ml_framework.orchestration.v2.dag_builder import ExecutionDAG
from ml_framework.orchestration.v2.event_bus import EventBus
from ml_framework.orchestration.v2.module_registry import ModuleRegistry

logger = logging.getLogger("ml_framework.orchestration.v2.execution_engine")


class ModuleExecutionError(Exception):
    """Wraps an exception raised by a module's invoke(), with module context."""

    def __init__(self, module_name: str, original: Exception) -> None:
        self.module_name = module_name
        self.original = original
        super().__init__(f"Module '{module_name}' failed: {original}")


class ExecutionEngine:
    """
    Runs an ExecutionDAG level by level.

    ctx_provider() must return the shared PipelineContext object modules
    mutate in place . It's a callable rather than a
    plain object so callers can construct/reset it lazily.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        event_bus: Optional[EventBus] = None,
        cache_store: Optional[CacheStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self.event_bus = event_bus or EventBus()
        self.cache_store = cache_store
        self.checkpoint_store = checkpoint_store
        self.max_workers = max_workers

    def run(
        self,
        dag: ExecutionDAG,
        ctx: Any,
        artifacts: ArtifactRegistry,
        run_id: Optional[str] = None,
        use_cache: bool = False,
    ) -> ArtifactRegistry:
        completed: Dict[str, str] = {}
        self.event_bus.publish(Event(EventType.PIPELINE_STARTED, {"modules": list(dag.nodes)}))

        try:
            for level_index, level in enumerate(dag.levels):
                if dag.is_parallel_level(level_index) and len(level) > 1:
                    self._run_level_parallel(level, dag, ctx, artifacts, completed, use_cache)
                else:
                    for module_name in level:
                        self._run_one(module_name, dag, ctx, artifacts, completed, use_cache)

                if run_id and self.checkpoint_store:
                    self.checkpoint_store.save(run_id, completed, artifacts.snapshot())
        except ModuleExecutionError as exc:
            self.event_bus.publish(
                Event(EventType.PIPELINE_FAILED, {"module": exc.module_name, "error": str(exc.original)})
            )
            raise

        self.event_bus.publish(Event(EventType.PIPELINE_COMPLETED, {"modules": list(completed)}))
        return artifacts

    def _run_level_parallel(
        self,
        level: tuple,
        dag: ExecutionDAG,
        ctx: Any,
        artifacts: ArtifactRegistry,
        completed: Dict[str, str],
        use_cache: bool,
    ) -> None:
        # ctx is shared and mutated in place by every spec (Phase 1 design) —
        # running truly-independent modules against the SAME ctx concurrently
        # is only safe if they touch disjoint ctx attributes, which
        # ModuleSpec.outputs (declared per-module) guarantees by construction
        # for any level the DAG Builder marked parallel.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one, name, dag, ctx, artifacts, completed, use_cache): name
                for name in level
            }
            for future in futures:
                future.result()  # re-raises ModuleExecutionError from the worker thread

    def _run_one(
        self,
        module_name: str,
        dag: ExecutionDAG,
        ctx: Any,
        artifacts: ArtifactRegistry,
        completed: Dict[str, str],
        use_cache: bool,
    ) -> None:
        spec = self.registry.get(module_name)
        node = dag.nodes[module_name]

        cache_key = None
        if use_cache and self.cache_store is not None:
            input_values = {name: artifacts.get_optional(name) for name in node.inputs}
            cache_key = compute_cache_key(spec.name, spec.version, {}, input_values)
            cached = self.cache_store.get(cache_key)
            if cached is not None:
                self.event_bus.publish(Event(EventType.CACHE_HIT, {"module": module_name}))
                for output_name in node.outputs:
                    if output_name in cached:
                        artifacts.put(output_name, cached[output_name], produced_by=module_name)
                        setattr(ctx, output_name, cached[output_name])
                completed[module_name] = cache_key
                return
            self.event_bus.publish(Event(EventType.CACHE_MISS, {"module": module_name}))

        self.event_bus.publish(Event(EventType.MODULE_STARTED, {"module": module_name}))
        t0 = time.time()

        attempts = spec.max_retries + 1 if spec.retryable else 1
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            if attempt > 0:
                self.event_bus.publish(
                    Event(EventType.RETRY_STARTED, {"module": module_name, "attempt": attempt})
                )
            try:
                spec.invoke(ctx)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 — module errors are data, not control flow here
                last_exc = exc
                logger.warning("Module '%s' attempt %d failed: %s", module_name, attempt + 1, exc)

        elapsed = round(time.time() - t0, 2)

        if last_exc is not None:
            self.event_bus.publish(
                Event(EventType.MODULE_FAILED, {"module": module_name, "error": str(last_exc), "elapsed_s": elapsed})
            )
            raise ModuleExecutionError(module_name, last_exc)

        produced: Dict[str, Any] = {}
        for output_name in node.outputs:
            if hasattr(ctx, output_name):
                value = getattr(ctx, output_name)
                artifacts.put(output_name, value, produced_by=module_name)
                produced[output_name] = value
            elif output_name in getattr(ctx, "artifacts", {}):
                value = ctx.artifacts[output_name]
                artifacts.put(output_name, value, produced_by=module_name)
                produced[output_name] = value

        if use_cache and self.cache_store is not None and cache_key is not None:
            self.cache_store.put(cache_key, produced)

        completed[module_name] = cache_key or ""
        self.event_bus.publish(
            Event(EventType.MODULE_COMPLETED, {"module": module_name, "elapsed_s": elapsed})
        )
        for output_name in produced:
            self.event_bus.publish(
                Event(EventType.ARTIFACT_CREATED, {"module": module_name, "artifact": output_name})
            )
