"""
cache_store.py — Avoids recomputing a module's output when nothing relevant changed.

A module's result is reused when the combination of (module name, module
version, resolved params, and a fingerprint of its input artifacts) is
identical to a previous run.

Hashing strategy
-----------------
- Params: stable JSON-ish repr, sorted keys.
- pandas DataFrame / Series inputs: hashed via pandas.util.hash_pandas_object
  (fast, content-based, ignores object identity).
- Everything else: repr() fallback;
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ml_framework.orchestration.v2.cache_store")


def _hash_value(value: Any) -> str:
    try:
        import pandas as pd

        if isinstance(value, (pd.DataFrame, pd.Series)):
            return hashlib.sha256(
                pd.util.hash_pandas_object(value, index=True).values.tobytes()
            ).hexdigest()
    except ImportError:
        pass

    try:
        payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_cache_key(
    module_name: str,
    module_version: str,
    params: Dict[str, Any],
    input_artifacts: Dict[str, Any],
) -> str:
    """
    Deterministic cache key for one module invocation..
    """
    parts = [module_name, module_version]
    parts.append(_hash_value(params))
    for name in sorted(input_artifacts):
        parts.append(f"{name}:{_hash_value(input_artifacts[name])}")
    digest_input = "|".join(parts).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


class CacheStore:
    """
    File-based cache: one pickle per cache key under `cache_dir`.

    get(key)       -> cached value, or None if absent
    put(key, value): persist a value under the given key
    clear()        : wipe the whole cache directory (used by tests / manual reset)
    """

    def __init__(self, cache_dir: str | Path = ".cache/orchestration_v2") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            logger.warning("CacheStore: failed to load '%s' — treating as miss.", key)
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception:
            logger.warning("CacheStore: failed to persist '%s' — continuing without cache.", key)

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink(missing_ok=True)
