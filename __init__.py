# __init__.py

"""
citybuilder_env: Lean, deterministic multi-objective knapsack enviornment.

Public API
-------------
- Env
    CityBuilderEnv, StepResult, make_env
- Config/state types (for power users; the factory shields casual users)
    GeneratorConfig, EpisodeConfig, Catalog, EpisodeState, Advisors
- Services / utilities
    compute_pareto_frontier_and_distance, RecommendationManager, ScoreManager, RewardManager
    DeterministicRNG, hash_uint64
- I/O helpers
    EpisodeLogger, make_run_dir
    ReplayTrace, save_replay, load_replay, run_replay_single_episode
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .env import CityBuilderEnv, StepResult
from .items import GeneratorConfig
from .state import EpisodeConfig, Catalog, EpisodeState, Advisors
from .pareto import compute_pareto_frontier_and_distance

from .utils.rng import DeterministicRNG, hash_uint64
from .recommend import RecommendationManager
from .score import ScoreManager
from .reward import RewardManager

from .io.logging import EpisodeLogger, make_run_dir
from .io.replay import (
    ReplayTrace,
    save_replay,
    load_replay,
    run_replay_single_episode,
)

__all__ = [
    # Core env
    "CityBuilderEnv",
    "StepResult",
    "make_env",

    # Config / state (optional for power users)
    "GeneratorConfig",
    "EpisodeConfig",
    "Catalog",
    "EpisodeState",
    "Advisors",

    # Services
    "compute_pareto_frontier_and_distance",
    "RecommendationManager",
    "ScoreManager",
    "RewardManager",

    # Utils
    "DeterministicRNG",
    "hash_uint64",

    # I0 helpers
    "EpisodeLogger",
    "make_run_dir",
    "ReplayTrace",
    "save_replay",
    "load_replay",
    "run_replay_single_episode",
]

__version__ = "0.1.0"

# ----------------------------------------------------------
# Convenience factory (no gcfg/ecfg required by the agent)
# ----------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        import yaml # PyYAML
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load the environment config."
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    
def _configs_from_yaml(cfg: dict) -> Tuple[GeneratorConfig, EpisodeConfig]:
    """
    Expect the YAML to have top-level sections 'generator' and 'episode'
    whose keys match the dataclasses for GeneratorConfig and EpisodeConfig
    """

    gen = cfg.get("generator", {})
    epi = cfg.get("episode", {})
    gcfg = GeneratorConfig(**gen)
    ecfg = EpisodeConfig(**epi)
    return gcfg, ecfg

def make_env(
        seed: int,
        cfg_path: Optional[str] = None,
        cfg_fingerprint: str = "v1",
) -> CityBuilderEnv:
    """
    Build a CityBuilderEnv without the agent providing gcfg/ecfg.

        Loads both GeneratorConfig and EpisodeConfig from YAML.
        The agent only passes a deterministic 'seed'
        'cfg_path' defaults to './config/default.yaml' if not provided.

    Example
    ---------
    >>> import citybuilder_env as cbe
    >>> env = cbe.make_env(seed=1234) # no gcfg/ecfg arguements
    >>> obs, mask, info = env.reset()
    """

    # Resolving config file
    candidates = []
    if cfg_path:
        candidates.append(Path(cfg_path))
    candidates.append(Path("./config/default.yaml"))
    cfg_file = next((p for p in candidates if p.exists()), None)
    if cfg_file is None:
        raise FileNotFoundError(
            f"Could not locate a config YAML. Tried: {', '.join(str(p) for p in candidates)}"
        )
    
    cfg = _load_yaml(cfg_file)
    gcfg, ecfg = _configs_from_yaml(cfg)

    # Construct env with provided seed and fingerprint
    env = CityBuilderEnv(gcfg=gcfg, ecfg=ecfg, master_seed=int(seed), cfg_fingerprint=str(cfg_fingerprint))
    return env