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

from .env import CityBuilderEnv, make_env, StepResult  # if StepResult is defined in env.py
from .items import GeneratorConfig, MarginalSpec
from .state import EpisodeConfig, Catalog, EpisodeState, Advisors
from .pareto import compute_pareto_frontier_and_distance
from .utils.rng import DeterministicRNG, hash_uint64
from .recommend import RecommendationManager
from .score import ScoreManager
from .reward import RewardManager
from .io.logging import EpisodeLogger, make_run_dir
from .io.replay import ReplayTrace, save_replay, load_replay, run_replay_single_episode

__all__ = [
    "CityBuilderEnv", "StepResult", "make_env",
    "GeneratorConfig", "EpisodeConfig", "Catalog", "EpisodeState", "Advisors",
    "compute_pareto_frontier_and_distance",
    "RecommendationManager", "ScoreManager", "RewardManager",
    "DeterministicRNG", "hash_uint64",
    "EpisodeLogger", "make_run_dir",
    "ReplayTrace", "save_replay", "load_replay", "run_replay_single_episode",
]
__version__ = "0.1.0"

