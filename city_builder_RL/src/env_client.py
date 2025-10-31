# city_builder_rl/env_client.py
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, Optional
from pathlib import Path

from .config import Config

from citybuilder_env import make_env, CityBuilderEnv  # type: ignore
from citybuilder_env.io.logging import EpisodeLogger, make_run_dir  # type: ignore


class EnvClient:
    """
    In-process adapter over citybuilder_env that preserves dict responses for loops.py.
    Also allows attaching the environment's EpisodeLogger.
    """

    def __init__(self, cfg: Config, mode: str = "train"):
        self.cfg = cfg
        self.mode = str(mode)
        self.env: Optional[CityBuilderEnv] = None
        self.env_id: str = "local-env"
        self.episode_id: int = 0

        self._episode_return: float = 0.0
        self._episode_steps: int = 0
        self._episode_ended: bool = False
        self._last_episode_summary: Optional[Dict[str, Any]] = None

        self._seed_used: Optional[int] = getattr(self.cfg, "SEED", None)
        self._cfg_path_used: Optional[str] = getattr(self.cfg, "ENV_CFG_PATH", None)
        self._cfg_fingerprint: str = getattr(self.cfg, "CFG_FINGERPRINT", "v1")

        self.server_meta: Dict[str, Any] = {}

        # NEW: env-native logging
        self._logger: Optional[EpisodeLogger] = None
        self._run_dir: Optional[Path] = None

    # -------------- Lifecycle --------------
    def init(self) -> Dict[str, Any]:
        self.env = make_env(
            seed=int(self._seed_used if self._seed_used is not None else 0),
            cfg_path=self._cfg_path_used,
            cfg_fingerprint=self._cfg_fingerprint,
            mode=self.mode,
        )
        self.server_meta = {
            "mode": self.mode,
            "cfg_path": self._cfg_path_used,
            "cfg_fingerprint": self._cfg_fingerprint,
        }
        # bind logger if already provided
        if self._logger is not None:
            self._bind_logger_to_env()
        return {"env_id": self.env_id, "config": dict(self.server_meta)}

    def seed(self, seed: int) -> Dict[str, Any]:
        self._seed_used = int(seed)
        self.env = make_env(
            seed=self._seed_used,
            cfg_path=self._cfg_path_used,
            cfg_fingerprint=self._cfg_fingerprint,
            mode=self.mode,
        )
        self._reset_episode_trackers()
        # re-bind logger after recreation
        if self._logger is not None:
            self._bind_logger_to_env()
        return {"ok": True, "seed": self._seed_used}

    # -------------- Logging integration (NEW) --------------
    def attach_logger(
        self,
        *,
        run_dir_base: str = "./logs",
        prefix: str = "run",
        run_dir: Optional[str] = None,
        logger: Optional[EpisodeLogger] = None,
    ) -> None:
        """
        Attach the environment's EpisodeLogger.

        - Provide `logger` directly OR
        - Let this method create one:
            - If `run_dir` is given, use it.
            - Else, create a timestamped dir via make_run_dir(run_dir_base, prefix).
        Safe to call before or after init(); will bind immediately if env exists.
        """
        if logger is None:
            if run_dir is None:
                self._run_dir = make_run_dir(run_dir_base, prefix)
            else:
                self._run_dir = Path(run_dir)
            logger = EpisodeLogger(self._run_dir)
        else:
            # try to keep a handle to its path
            self._run_dir = Path(getattr(logger, "run_dir", run_dir or run_dir_base))

        self._logger = logger
        if self.env is not None:
            self._bind_logger_to_env()

    def _bind_logger_to_env(self) -> None:
        """Give the env the logger so it can call start_episode/log_step/end_episode."""
        assert self.env is not None
        if hasattr(self.env, "set_logger"):
            self.env.set_logger(self._logger)  # type: ignore[arg-type]
        else:
            setattr(self.env, "logger", self._logger)

    # -------------- Env API --------------
    def reset(self) -> Dict[str, Any]:
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
        assert self.env is not None, "Call init() first"
        step_out = self.env.step(action)

        if is_dataclass(step_out) or (
            hasattr(step_out, "observation") and hasattr(step_out, "reward") and hasattr(step_out, "done")
        ):
            obs_next = step_out.observation
            reward = float(step_out.reward)
            done = bool(step_out.done)
            info = dict(step_out.info) if step_out.info is not None else {}
            if getattr(step_out, "action_mask", None) is not None:
                info.setdefault("action_mask", step_out.action_mask)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs_next, reward, done, info = step_out
            reward = float(reward); done = bool(done); info = {} if info is None else dict(info)
        elif isinstance(step_out, tuple) and len(step_out) == 5:
            obs_next, reward, terminated, truncated, info = step_out
            reward = float(reward); done = bool(terminated or truncated); info = {} if info is None else dict(info)
        else:
            raise RuntimeError("Unexpected env.step() return shape")

        self._episode_steps += 1
        self._episode_return += float(reward)
        if done:
            self._episode_ended = True

        return {"observation": obs_next, "reward": reward, "done": done, "info": info}

    def summary(self) -> Dict[str, Any]:
        if self._episode_ended:
            return self._pull_episode_summary()
        return {
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "steps": self._episode_steps,
            "return": self._episode_return,
            "pareto_distance": 0.0,
            "config": dict(self.server_meta),
            "seed": self._seed_used,
            "run_dir": str(self._run_dir) if self._run_dir else None,
        }

    def close(self):
        if self.env and hasattr(self.env, "close"):
            self.env.close()

    # -------------- Helpers --------------
    def _reset_episode_trackers(self):
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_ended = False
        self._last_episode_summary = None

    def _pull_episode_summary(self) -> Dict[str, Any]:
        assert self.env is not None, "Call init() first"
        if self._last_episode_summary is None:
            if hasattr(self.env, "episode_summary"):
                raw = self.env.episode_summary()
                pd = None
                if isinstance(raw, dict):
                    pd = raw.get("pareto_distance_chebyshev") or raw.get("pareto_distance")
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
                    "raw": raw,
                    "run_dir": str(self._run_dir) if self._run_dir else None,
                }
            else:
                self._last_episode_summary = {
                    "env_id": self.env_id,
                    "episode_id": self.episode_id,
                    "steps": self._episode_steps,
                    "return": self._episode_return,
                    "pareto_distance": 0.0,
                    "config": dict(self.server_meta),
                    "seed": self._seed_used,
                    "run_dir": str(self._run_dir) if self._run_dir else None,
                }
        return self._last_episode_summary
