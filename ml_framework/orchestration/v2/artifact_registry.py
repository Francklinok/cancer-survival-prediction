"""
To keep things clean and prevent modules from stepping on each other's toes, 
they never talk to each other directly. Instead, anything a module produces 
is stored right here under its official name (ModuleSpec.outputs).
When a downstream module needs that data later, it simply grabs it from this 
registry (ModuleSpec.inputs).

By keeping everything in one central spot, the DAG remains the ultimate map of
how data flows through our system

"""

from __future__ import annotations

from typing import Any, Dict, Iterator, KeysView


class ArtifactNotFoundError(KeyError):
    """Raised when a module input is read before it has been produced."""


class ArtifactRegistry:
    """
    This is a simple, named store where everything is indexed by its artifact name
    which is just a string matching the entries in the ModuleSpec.outputs or ModuleSpec.inputs.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._producer: Dict[str, str] = {}  

    def put(self, name: str, value: Any, *, produced_by: str = "") -> None:
        self._store[name] = value
        if produced_by:
            self._producer[name] = produced_by

    def get(self, name: str) -> Any:
        if name not in self._store:
            raise ArtifactNotFoundError(
                f"Artifact '{name}' has not been produced yet."
            )
        return self._store[name]

    def get_optional(self, name: str, default: Any = None) -> Any:
        return self._store.get(name, default)

    def has(self, name: str) -> bool:
        return name in self._store

    def producer_of(self, name: str) -> str:
        return self._producer.get(name, "")

    def keys(self) -> KeysView[str]:
        return self._store.keys()

    def snapshot(self) -> Dict[str, Any]:
        """Shallow copy of the current artifact map for checkpointing/debugging."""
        return dict(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)
