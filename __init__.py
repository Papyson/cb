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
from .items import GeneratorConfig, MarginalSpec
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
    gen_config_dict = cfg.get("generator", {})
    epi_config_dict = cfg.get("episode", {})

    if 'marginals' in gen_config_dict:
        marginals_data = gen_config_dict.pop('marginals')
        if 'cost' in marginals_data:
            gen_config_dict['cost_marginal'] = marginals_data['cost']
        if 'objectives' in marginals_data:
            gen_config_dict['objective_marginals'] = marginals_data['objectives']

    if 'cost_marginal' in gen_config_dict:
        cost_dict = gen_config_dict['cost_marginal']
        gen_config_dict['cost_marginal'] = MarginalSpec(**cost_dict)

    if 'objective_marginals' in gen_config_dict:
        obj_list_of_dicts = gen_config_dict['objective_marginals']
        spec_keys = ['dist', 'shape', 'scale']
        gen_config_dict['objective_marginals'] = [
            MarginalSpec(**{k: d.get(k) for k in spec_keys if k in d})
            for d in obj_list_of_dicts
        ]

    if 'sigma' in gen_config_dict:
        gen_config_dict['sigma'] = np.array(gen_config_dict['sigma'])
    
    gen_config_dict.pop('seed', None)

    if 'int_scale' not in epi_config_dict and 'int_scale' in gen_config_dict:
        epi_config_dict['int_scale'] = gen_config_dict['int_scale']
        
    gcfg = GeneratorConfig(**gen_config_dict)
    ecfg = EpisodeConfig(**epi_config_dict)
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

    # __init__.py (New Code)

    package_dir = Path(__file__).parent

    # Resolving config file
    candidates = []
    if cfg_path:
        candidates.append(Path(cfg_path))

    default_config_path = package_dir / "config" / "default.yaml"
    candidates.append(default_config_path)

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