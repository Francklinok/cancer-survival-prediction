"""
This module is here to save the day when a long-running pipeline (like SHAP or Bayesian tuning) 
gets cut short. If your Jupyter kernel crashes or you manually interrupt 
a run, checkpoint_store lets you pick up right where you left off.

What it does (and what it doesn't)
  1- Keep it simple: This is absolutely not built for complex, distributed multi-worker
     setups. We deliberately excluded that scope to keep things straightforward and local 
     
  2- Tracks progress: For any given run_id, it remembers which modules finished successfully
    and keeps a snapshot of the artifacts they produced.

  3- No overthinking: When you resume a run, it simply skips the steps that already
    finished (as long as their cache keys still match). It only handles saving and 
    loading—it never tries to be smart or decide what is still valid.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ml_framework.orchestration.v2.checkpoint_store")


class CheckpointStore:
    """
    File-based checkpoint store: one directory per run_id under `base_dir`,
    containing a `state.json` (completed module names + their cache keys)
    and an `artifacts.pkl` (ArtifactRegistry snapshot).
    """

    def __init__(self, base_dir: str | Path = ".checkpoints/orchestration_v2") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        path = self.base_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        run_id: str,
        completed_modules: Dict[str, str],
        artifacts: Dict[str, Any],
    ) -> None:
        """
        completed_modules : {module_name: cache_key_used}
        artifacts          : ArtifactRegistry.snapshot()
        """
        run_dir = self._run_dir(run_id)
        try:
            (run_dir / "state.json").write_text(
                json.dumps({"completed_modules": completed_modules}, indent=2),
                encoding="utf-8",
            )
            with open(run_dir / "artifacts.pkl", "wb") as f:
                pickle.dump(artifacts, f)
        except Exception:
            logger.exception("CheckpointStore: failed to save checkpoint for run '%s'.", run_id)

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns {"completed_modules": {...}, "artifacts": {...}} or None if no
        checkpoint exists for this run_id.
        """
        run_dir = self.base_dir / run_id
        state_path = run_dir / "state.json"
        artifacts_path = run_dir / "artifacts.pkl"
        if not state_path.exists() or not artifacts_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            with open(artifacts_path, "rb") as f:
                artifacts = pickle.load(f)
            return {"completed_modules": state["completed_modules"], "artifacts": artifacts}
        except Exception:
            logger.exception("CheckpointStore: failed to load checkpoint for run '%s'.", run_id)
            return None

    def exists(self, run_id: str) -> bool:
        run_dir = self.base_dir / run_id
        return (run_dir / "state.json").exists() and (run_dir / "artifacts.pkl").exists()

    def clear(self, run_id: str) -> None:
        run_dir = self.base_dir / run_id
        for f in run_dir.glob("*"):
            f.unlink(missing_ok=True)

    def list_runs(self) -> List[str]:
        return [p.name for p in self.base_dir.iterdir() if p.is_dir()]
