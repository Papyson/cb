# io/logging.py

"""
Lightweight, structured logging for CityBuilder RL runs.

Files written
---------------
<run_dir>/
    steps.csv       # per-step rows (append-only)
    episodes.jsonl  # one JSON object per episode (append-only)
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

# -------------------------
# Helpers
# -------------------------

def make_run_dir(base_dir: str | os.PathLike, prefix: str = "run") -> Path:
    """
    Create a timestamped run directory under base_dir, e.g.,
    base/run_2025-10-15_11-22-33/
    """
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run = Path(base_dir) / f"{prefix}_{ts}"
    run.mkdir(parents=True, exist_ok=False)
    return run

def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _to_builtin(obj: Any) -> Any:
    """
    Convert numpy scalars/arrays and dataclasses to JSON/CSV-friendly Python types.
    """
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarry):
        # Keep arrays as plain Python lists for JSON; CSV will stringify
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj

def _flatten(prefix: str, obj: Dict[str, Any], out: Dict[str, Any]) -> None:
    """
    Flatten nested dicts into dotted keys: {"a" {"b": 1}} -> {"a.b": 1}
    """
    for k,v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        v = _to_builtin(v)
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v

def _record_to_dict(record: Dict[str, Any] | Any) -> Dict[str, Any]:
    """
    Accept dict or dataclass-like; convert to a flat dict with JSON-safe values.
    """
    if is_dataclass(record):
        record = asdict(record)
    elif hasattr(record, "__dict__") and not isinstance(record, dict):
        record = {k: v for k, v in vars(record).items() if not k.startswith("_")}
    if not isinstance(record, dict):
        raise TypeError("record must be a dict or dataclass-like")
    flat: Dict[str, Any] = {}
    _flatten("", record, flat)
    return flat

# --------------------------
# Logger
# --------------------------

class EpisodeLogger:
    """
    Structured logger for steps and episodes.

    Usage
    -------
    >>> run_dir = make_run_dir("./logs")
    >>> logger = EpisodeLogger(run_dir)
    >>> logger.start_episode({"episode_id": 1, "catalog_seed": 123, "budget_int": 98765})
    >>> logger.log_step({"episode_id": 1, "step_idx": 0, "item_id": 7, "reward": 1.25,
                        "info": {"pareto": {"distance_chebyshev": 0.18}}})
    >>> logger.end_episode(summary_dict_from_env)
    >>> logger.close()
    """

    def __init__(self, run_dir: str | os.PathLike):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._steps_path = self.run_dir / "steps.csv"
        self._episodes_path = self.run_dir / "episodes.jsonl"

        self._steps_fh = open(self._steps_path, "a", newline="", encoding="utf-8")
        self._episodes_fh = open(self._episodes_path, "a", encoding="utf-8")

        self._csv_writer: Optional[csv.DictWriter] = None
        self._header: Optional[list[str]] = None
        self._episode_meta: Dict[str, Any] = {}

    # ------ Episode Lifecycle -----------

    def start_episode(self, meta: Dict[str, Any]) -> None:
        """ Set episode metadata (merged into each row, without overriding explicit fields)."""
        if not isinstance(meta, dict):
            raise TypeError("meta must be a dict")
        # Make values JSON-Safe
        self._episode_meta = _to_builtin(meta)

    def end_episode(self, summary: Dict[str, Any]) -> None:
        """
        Append one JSON object (already conatins Pareto fields if env.summary() does).
        """
        if not isinstance(summary, dict):
            raise TypeError("summary must be a dict")
        safe = _to_builtin(summary)
        self._episodes_fh.write(json.dumps(safe, separators=(",", ":")) + "\n")
        self._episodes_fh.flush()
        self._episode_meta = {}

    # ------- Step Logging --------------

    def log_step(self, record: Dict[str, Any] | Any) -> None:
        """
        Append a step record to steps.csv. Nested dicts (e.g. info.pareto.*) are flattened.
        The CSV header is fixed on first wwrite; subsequent extra/missing keys are handled safely.
        """

        flat = _record_to_dict(record)

        # Merge meta (do not oveerwrite explicitly provided fields)
        merged = dict(self._episode_meta)
        merged.update(flat)

        # Initialize CSV writer/header on first use
        if self._csv_writer is None:
            # Fix header order deterministically
            self._header = list(merged.keys())
            self._csv_writer = csv.DictWriter(self._steps_fh, fieldnames=self._header, extrasaction="ignore")
            # only write header if new file
            if self._steps_path.stat().st_size == 0:
                self._csv_writer.writeheader()
            else:
                # Conform to fixed header; add missing keys to merged with empty strings
                for k in self._header:
                    if k not in merged:
                        merged[k] = ""

            # Convert all remaining values to builtins
            merged = _to_builtin(merged)
            self._csv_writer.writerow(merged)
            self._steps_fh.flush()

    # ------ Utilities ---------

    def close(self) -> None:
        try:
            self._steps_fh.close()
        finally:
            self._episodes_fh.close()

    def __enter__(self) -> "EpisodeLogger":
        return self
    
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
