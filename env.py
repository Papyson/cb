# env.py
"""
CityBuilderEnv: minimal Gymnasium-style environment orchestrator.

Deterministic catalog per episode via items.generate_catalog and RNG fan-out
One-step loop order
    1. Mask = selection.build_action_mask(...)
    2. validate action (mask-enforced)
    3. snapshot advisors_pre and ref_ids = R_t
    4. score_mgr.apply_selection(...)           # budget & cum scores (ints)
    5. selection.apply_selection(...)           # remove item, step++
    6. reward_mgr.compute_reward(...)           # using advisors_pre & R_t
    7. done = selection.check_done(...)         
    8. if not done: advisors = recommend. RecommednationManager.compute_advisors(...)

Observation schema(simple, stable):
    {
        "costs_int": np.int64[N],
        "objs_int": np.int64[N, K]
        "advisor_freq": np.int64[N],    # 0...K
        "budget_rem_int": int,
        "scores_cum_int": np.int64[K],
        "step_idx": int,
    }

Action space: item_id ∈ {0..N-1} with mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .state import Catalog, EpisodeConfig, EpisodeState, Advisors
from .items import GeneratorConfig, generate_catalog
from .recommend import RecommendationManager
from .score import ScoreManager
from .reward import RewardManager
from . import selection as sel
from .utils.rng import DeterministicRNG, hash_uint64
from .pareto import compute_pareto_frontier_and_distance

@dataclass
class StepResult:
    observation: Dict
    reward: float
    done: bool
    info: Dict
    action_mask: np.ndarray

class CityBuilderEnv:
    """
    Deterministic, lean environment for RL training.

    Parameters
    -----------
    gcfg: GeneratorConfig
        Gaussian-copula generator settings (catalog-level)
    ecfg: EpisodeConfig
        Episode settings (budgeting, beta, limits, etc.)
    master_seed: int
        Root seed controlling the entire episode sequence (catalogs, solvers).
    cfg_fingerprint: str
        A short stable string/hash tying seeds to config (change when gcfg/ecfg change).
    """

    def __init__(self, gcfg: GeneratorConfig, ecfg: EpisodeConfig, master_seed: int, cfg_fingerprint: str = "v1"):
        self.gcfg = gcfg
        self.ecfg = ecfg
        self.cfg_fp = str(cfg_fingerprint)

        self.master = DeterministicRNG(seed=int(master_seed), stream="env")
        self.episode_idx = 0
        self.episode_id = 0

        # Will be set at reset
        self.catalog: Catalog | None = None
        self.costs_int: np.ndarray | None = None
        self.objs_int: np.ndarray | None = None
        self.state: EpisodeState | None = None
        self.rec_mgr: RecommendationManager | None = None

        self._initial_budget_int: int | None = None
        self.score_mgr = ScoreManager(K=self.ecfg.K)
        self.reward_mgr = RewardManager(K=self.ecfg.K, beta=self.ecfg.beta, normalization_mode="max")

    # ----------------------
    # Lifecycle
    # ----------------------

    def seed(self, seed: int) -> None:
        """Reset the master RNG and episode counter."""
        self.master = DeterministicRNG(seed=int(seed), stream="env")
        self.episode_idx = 0
        self.episode_id = 0

    def reset(self) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Start a new episode:
            Generate catalog for episode_idx (deterministic)
            Compute budget from analytic expected cost
            Initialize state and advisors from R_0
        Returns (observation, action_mask, info)
        """
        # 1. Deterministic catalog stream for this episode
        cat_rng = self.master.child("catalog", self.episode_idx, self.cfg_fp)

        # 2. Generate catalog and analytic expected cost
        catalog, expected_cost = generate_catalog(self.gcfg, cat_rng)
        self.catalog = catalog
        self.costs_int = catalog.costs_array()
        self.objs_int = catalog.objs_matrix()
        N = int(catalog.N)

        # 3. Budget (int) from analytic E[cost]
        budget_int = self.ecfg.derive_budget_int(expected_cost)

        # 4. Initialize episode state
        remaining_ids = catalog.ids_array()         #0..N-1 sorted
        scores0 = np.zeros(self.ecfg.K, dtype=np.int64)
        self.state = EpisodeState(
            catalog=catalog,
            budget_rem_int=int(budget_int),
            scores_cum_int=scores0,
            remaining_ids=remaining_ids,
            selected_ids=[],
            step_idx=0,
            advisors=Advisors.empty(self.ecfg.K),       # replaced below
        )

        # 5. Set up RecommendationManager for this episode (deterministic base seed)
        rec_seed = int(hash_uint64(self.master.seed, "ilp", self.episode_idx, self.cfg_fp))
        self.rec_mgr = RecommendationManager(
            K=self.ecfg.K,
            time_limit_ms=self.ecfg.solver_time_limit_ms,
            cache_size=self.ecfg.cache_size,
            budget_bucket=self.ecfg.budget_bucket,
            base_seed=rec_seed,
        )

        # 6. Initial advisors for R_0
        advisors, _ = self.rec_mgr.compute_advisors(
            remaining_ids=self.state.remaining_ids,
            budget_rem_int=self.state.budget_rem_int,
            costs_int=self.costs_int,
            objs_int=self.objs_int,
        )
        self.state.advisors = advisors

        # 7. Prepare score ledger & reward trackers
        self.score_mgr.reset()
        self.reward_mgr.reset()

        # 8. Build first observation & mask
        obs = self._make_observation()
        mask = sel.build_action_mask(self.state, self.costs_int)

        # 9. Episode bookkeeping
        self.episode_id += 1
        self.episode_idx += 1

        info = {
            "episode_id": self.episode_id,
            "catalog_seed": int(cat_rng.seed),
            "budget_int": int(budget_int),
        }
        return obs, mask, info
    
    # ----------------
    # Step
    # ----------------

    def step(self, action_id: int) -> StepResult:
        """
        One environment step. Mask-enforced: illegal actions return no mutation and reward=0
        """
        assert self.state is not None and self.costs_int is not None and self.objs_int is not None and self.rec_mgr is not None

        # 1. Build current mask and validate action
        mask = sel.build_action_mask(self.state, self.costs_int)
        if action_id < 0 or action_id >= mask.size or not bool(mask[action_id]):
            # Illegal action: mask_enforced -> no mutation; zero reward
            obs = self._make_observation()
            info = {"illegal_action": int(action_id)}
            return StepResult(observation=obs, reward=0.0, done=False, info=info, action_mask=mask)
        
        # 2. Snapshot pre-decision context for reward
        advisors_pre = self.state.advisors
        ref_ids = self.state.remaining_ids  # R_t

        # 3. Accounting (ints)
        score_info = self.score_mgr.apply_selection(self.state, self.costs_int, self.objs_int, action_id)

        # 4. Mutate remaining set
        sel.apply_selection(self.state, action_id)

        # 5. Reward (use advisors_pre & R_t)
        reward, details = self.reward_mgr.compute_reward(
            selected_id=action_id,
            remaining_ref_ids=ref_ids,
            objs_int=self.objs_int,
            advisors=advisors_pre,
        )

        # 6. Termination check
        done_check = sel.check_done(self.state, self.ecfg.max_steps)
        done = bool(done_check)

        rec_stats = None
        if not done:
            # 7. Refresh advisors for R_{t+1}
            advisors_next, rec_stats = self.rec_mgr.compute_advisors(
                remaining_ids=self.state.remaining_ids,
                budget_rem_int=self.state.budget_rem_int,
                costs_int=self.costs_int,
                objs_int=self.objs_int,
            )
            self.state.advisors = advisors_next
        else:
            # --- compute pareto metrics at the terminal step ---
            pareto_seed = int(hash_uint64(self.master.seed, "pareto", self.episode_id, self.cfg_fp))
            result = compute_pareto_frontier_and_distance(
                objs_int=self.objs_int,
                costs_int=self.costs_int,
                budget_int=int(self._initial_budget_int or 0),
                agent_scores_int=self.state.scores_cum_int,
                K=self.ecfg.K,
                num_weights=self.ecfg.num_frontier_weights,
                time_limit_ms=self.ecfg.solver_time_limit_ms,
                base_seed=pareto_seed,
            )
            pareto_info = {
                "distance_chebyshev": float(result.distance_chebyshev),
                "frontier_size": int(result.frontier_obj_int.shape[0]),
                "best_index": int(result.best_frontier_idx),
                "norm_mins": result.norm_mins,
                "norm_maxs": result.norm_maxs,
            }
        
        # 8. Build next observation & mask
        obs_next = self._make_observation()
        mask_next = sel.build_action_mask(self.state, self.costs_int)

        info = {
            "cost_int": score_info["cost_int"],
            "budget_after_int": score_info["budget_after_int"],
            "advisor_freq_selected": advisors_pre.freq.get(int(action_id), 0),
            "reward_details": {
                "norm_components": details.norm_components,
                "advisor_freq": details.advisor_freq,
                "max_vec_int": details.max_vec_int,
                "mode": details.mode,
            },
            "done_reason": done_check.reason,
            "recommedation_stats": (None if rec_stats is None else {
                "cache_hit": rec_stats.cache_hit,
                "total_wall_ms": rec_stats.total_wall_ms,
                "per_objective_status": rec_stats.per_objective_status,
            }),
        }
        if pareto_info is not None:
            info["pareto"] = pareto_info

        return StepResult(observation=obs_next, reward=float(reward), done=done, info=info, action_mask=mask_next)
    
    # -----------------
    # Summaries
    # -----------------

    def episode_summary(self) -> Dict:
        """ Light weight end-of-episode summary (call after done=True)."""
        assert self.state is not None and self.catalog is not None
        # --- compute Pareto metrics for the final point ---
        pareto_seed = int(hash_uint64(self.master.seed, "pareto", self.episode_id, self.cfg_fp))
        result = compute_pareto_frontier_and_distance(
            objs_int=self.objs_int,
            costs_int=self.costs_int,
            budget_int=int(self._initial_budget_int or 0),
            agent_scores_int=self.state.scores_cum_int,
            K=self.ecfg.K,
            num_weights=self.ecfg.num_frontier_weights,
            time_limit_ms=self.ecfg.solver_time_limit_ms,
            base_seed=pareto_seed,
        )

        return {
            "episode_id": self.episode_id,
            "final_scores_int": self.state.scores_cum_int.copy(),
            "budget_remaining_int": int(self.state.budget_rem_int),
            "num_steps": int (self.state.step_idx),
            "selected_ids": list(self.state.selected_ids),
            "catalog_name": self.catalog.name,
            "int_scale": int(self.catalog.int_scale),

            # Pareto metrics
            "pareto_distance_chebyshev": float(result.distance_chebyshev),
            "pareto_frontier_size": int(result.frontier_obj_int.shape[0]),
            "pareto_best_index": int(result.best_frontier_idx),
            "pareto_norm_mins": result.norm_mins,
            "pareto_norm_maxs": result.norm_maxs,
        }
    
    # ------------------
    # Internals
    # ------------------

    def _make_observation(self) -> Dict:
        """ Assemble the observation dict the agent will consume."""
        assert self.state is not None and self.costs_int is not None and self.objs_int is not None
        N = self.costs_int.shape[0]
        # Build a dense advisor frequency vector (0..K) for all item_ids
        adv_freq = np.zeros(N, dtype=np.int64)
        for iid, f in self.state.advisors.freq.items():
            adv_freq[int(iid)] = int(f)

        return {
            "costs_int": self.costs_int,                        # (N,)
            "objs_int": self.objs_int,                          # (N, K)
            "advisor_freq": adv_freq,                           # (N,)
            "budget_rem_int": int(self.state.budget_rem_int),
            "scores_cum_int": self.state.scores_cum_int.copy(), # (K,)
            "step_idx": int(self.state.step_idx)
        }
