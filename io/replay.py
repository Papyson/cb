# io/replay.py
"""
Replay traces for deterministic re-runs.

Replay JSON format
--------------------
{
    "format_version": 1,
    "master_seed": 123456,
    "cfg_fingerpriint": "v1",
    "episode_index": 0,
    "actions": [7, 3, 12, ...],
    "meta": { "git_sha": "...", "note": "optional" }
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

@dataclass
class ReplayTrace:
    format_version: int
    master_seed: int
    cfg_fingerprint: str = "v1"
    episode_index: int = 0
    actions: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict) # free-from provenance

def save_replay(path: str | Path, trace: ReplayTrace) -> None:
    """ Write a replay trace to a JSON file (overwrites if exists)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(trace), f, indent=2, sort_keys=True)

def load_replay(path: str | Path) -> ReplayTrace:
    """ Load a replay trace from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "format_version" not in obj or int(obj["format_version"]) != 1:
        raise ValueError("Unsupported or missing format_version in replay trace")
    return ReplayTrace(**obj)

# --------------------
# Running a replay
# --------------------

def run_replay_single_episode(
        env_factory: Callable[[int, str], Any],
        trace: ReplayTrace,
) -> Dict[str, Any]:
    """
    Run a single-episode replay using an environment factory.

    Parameters
    -----------
    env_factory: callable(master_seed: int, cfg_fingerprint: str) -> env
        Returns a fresh CityBuilderEnv (or compatibile) instance. The env must implement:
        -seed(seed: int) -> None
        -reset() -> (observation, action_mask, info)
        -step(action_id: int) -> StepResult (.reward, .done, .action_mask, .observation, .info)
        -episode_summary() -> dict
    trace: ReplayTrace
        The trace to execute.

    Returns
    ----------
    dict with:
        ok: bool
        error: str | None
        total_reward: float
        num_steps: int
        mismatch_at: int | None
        final_summary: dict
        terminal_pareto: dict | None # if env.step included info["pareto"]
    """

    env = env_factory(int(trace.master_seed), str(trace.cfg_fingerprint))
    env.seed(int(trace.master_seed))

    # Start episode
    _, mask, _ = env.reset()

    total_reward = 0.0
    mismatch_at: Optional[int] = None
    error: Optional[str] = None
    terminal_pareto: Optional[Dict[str, Any]] = None

    for t, a in enumerate(trace.actions):
        # Mask-enforced: illegal action -> mismatch
        if a < 0 or a >= mask.size or not bool (mask[a]):
            mismatch_at = t
            error = f"Illegal action at step {t}: {a}"
            break

        step_res = env.step(int(a))
        total_reward += float(step_res.reward)
        mask = step_res.action_mask

        # Capture Pareto info if this was terminal
        if step_res.done:
            # If trace continues beyond done -> mismatch
            if t < len(trace.actions) - 1:
                mismatch_at = t + 1
                error = "Environment terminated before consuming all actions"
            # Pull terminal Pareto info (attached by env.step on done=True)
            info = step_res.info or {}
            if "pareto" in info and isinstance(info["pareto"], dict):
                terminal_pareto = info["pareto"]
            break

    # If we consumed all actions but env not done, mark soft mismatch
    if mismatch_at is None and not step_res.done:
        error = "Trace exhausted before environment termination"
        mismatch_at = len(trace.actions)

    final = env.episode_summary()   # includes pareto_* fields per current env.py

    return {
        "ok": mismatch_at is None and error is None,
        "error": error,
        "total_reward": total_reward,
        "num_steps": int(final.get("num_steps", 0)),
        "mismatch_at": mismatch_at,
        "final_summary": final,
        "terminal_pareto": terminal_pareto,
    }