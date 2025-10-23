# city_builder_rl/env_client.py
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, Optional

from .config import Config

# Local, in-process env (ensure: pip install -e ./citybuilder_env)
from citybuilder_env import make_env, CityBuilderEnv  # type: ignore


class EnvClient:
    """
    In-process adapter over citybuilder_env that preserves the response shapes
    expected by loops.py:

        reset() -> {"observation": obs, "info": info}
        step(a) -> {"observation": obs_next, "reward": r, "done": done, "info": info}

    Pareto distance is NOT read on step(); it is obtained from env.episode_summary()
    when the episode ends and returned via summary().
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.env: Optional[CityBuilderEnv] = None
        self.env_id: str = "local-env"
        self.episode_id: int = 0

        # Episode accounting
        self._episode_return: float = 0.0
        self._episode_steps: int = 0

        # End-of-episode summary cache
        self._episode_ended: bool = False
        self._last_episode_summary: Optional[Dict[str, Any]] = None

        # Build args (allow overrides via cfg)
        self._seed_used: Optional[int] = getattr(cfg, "SEED", None)
        self._cfg_path_used: Optional[str] = getattr(cfg, "ENV_CFG_PATH", None)
        self._cfg_fingerprint: str = getattr(cfg, "CFG_FINGERPRINT", "v1")

        # Meta
        self.server_meta: Dict[str, Any] = {}

    # ---------------- Lifecycle ----------------
    def init(self) -> Dict[str, Any]:
        """Construct the environment in-process."""
        self.env = make_env(
            seed=int(self._seed_used if self._seed_used is not None else 0),
            cfg_path=self._cfg_path_used,
            cfg_fingerprint=self._cfg_fingerprint,
        )
        self.server_meta = {
            "mode": "local",
            "cfg_path": self._cfg_path_used,
            "cfg_fingerprint": self._cfg_fingerprint,
        }
        return {"env_id": self.env_id, "config": dict(self.server_meta)}

    def seed(self, seed: int) -> Dict[str, Any]:
        """Recreate env with a new master seed for determinism at construction time."""
        self._seed_used = int(seed)
        self.env = make_env(
            seed=self._seed_used,
            cfg_path=self._cfg_path_used,
            cfg_fingerprint=self._cfg_fingerprint,
        )
        self._reset_episode_trackers()
        return {"ok": True, "seed": self._seed_used}

    # ---------------- Env API (dict responses) ----------------
    def reset(self) -> Dict[str, Any]:
        """
        Returns a dict so loops.py can do:
            resp = env.reset()
            obs = resp["observation"]
        """
        assert self.env is not None, "Call init() first"

        self.episode_id += 1
        self._reset_episode_trackers()

        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 3:
            obs, mask, info = out
        elif isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        return {"observation": obs, "info": info or {}}

    def step(self, action: Any) -> Dict[str, Any]:
        """
        Accepts whichever action your env expects (e.g., item_id:int).
        Unwraps StepResult and returns dict with keys: observation, reward, done, info.
        """
        assert self.env is not None, "Call init() first"

        step_out = self.env.step(action)

        # Case 1: StepResult dataclass (your env)
        if is_dataclass(step_out) or (
            hasattr(step_out, "observation")
            and hasattr(step_out, "reward")
            and hasattr(step_out, "done")
        ):
            obs_next = step_out.observation
            reward = float(step_out.reward)
            done = bool(step_out.done)
            info = dict(step_out.info) if step_out.info is not None else {}
            # Surface mask if present
            if getattr(step_out, "action_mask", None) is not None:
                info.setdefault("action_mask", step_out.action_mask)

        # Case 2: Classic Gym 4-tuple
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs_next, reward, done, info = step_out
            reward = float(reward)
            done = bool(done)
            info = {} if info is None else dict(info)

        # Case 3: Gymnasium 5-tuple
        elif isinstance(step_out, tuple) and len(step_out) == 5:
            obs_next, reward, terminated, truncated, info = step_out
            reward = float(reward)
            done = bool(terminated or truncated)
            info = {} if info is None else dict(info)

        else:
            raise RuntimeError("Unexpected env.step() return shape")

        # Accounting
        self._episode_steps += 1
        self._episode_return += float(reward)

        if done:
            self._episode_ended = True  # summary() will pull true episode summary

        return {
            "observation": obs_next,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def summary(self) -> Dict[str, Any]:
        """
        If the episode ended, fetch the real end-of-episode summary from the env
        (including pareto_distance). Otherwise return a lightweight snapshot with
        pareto_distance=0.0 (not available mid-episode).
        """
        if self._episode_ended:
            return self._pull_episode_summary()

        # Mid-episode snapshot (no Pareto info)
        return {
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "steps": self._episode_steps,
            "return": self._episode_return,
            "pareto_distance": 0.0,
            "config": dict(self.server_meta),
            "seed": self._seed_used,
        }

    def close(self):
        if self.env and hasattr(self.env, "close"):
            self.env.close()

    # ---------------- Helpers ----------------
    def _reset_episode_trackers(self):
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_ended = False
        self._last_episode_summary = None

    def _pull_episode_summary(self) -> Dict[str, Any]:
        """
        Call the env's episode_summary() ONCE at episode end and cache the result.
        Normalizes pareto_distance to top-level for the trainer.
        """
        assert self.env is not None, "Call init() first"
        if self._last_episode_summary is None:
            if hasattr(self.env, "episode_summary"):
                raw = self.env.episode_summary()  # expected to be a dict
                pd = None
                if isinstance(raw, dict):
                    pd = raw.get("pareto_distance_chebyshev")
                    if pd is None:
                        metrics = raw.get("metrics")
                        if isinstance(metrics, dict):
                            pd = metrics.get("pareto_distance") or metrics.get("pareto_dist")
                self._last_episode_summary = {
                    "env_id": self.env_id,
                    "episode_id": self.episode_id,
                    "steps": self._episode_steps,
                    "return": self._episode_return,
                    "pareto_distance": float(pd) if pd is not None else 0.0,
                    "config": dict(self.server_meta),
                    "seed": self._seed_used,
                    "raw": raw,  # keep raw for debugging/analysis if needed
                }
            else:
                # Fallback if env lacks episode_summary()
                self._last_episode_summary = {
                    "env_id": self.env_id,
                    "episode_id": self.episode_id,
                    "steps": self._episode_steps,
                    "return": self._episode_return,
                    "pareto_distance": 0.0,
                    "config": dict(self.server_meta),
                    "seed": self._seed_used,
                }
        return self._last_episode_summary
