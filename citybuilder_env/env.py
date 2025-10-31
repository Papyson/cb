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

from dataclasses import dataclass, replace
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path

import numpy as np

from citybuilder_env.state import Catalog, EpisodeConfig, EpisodeState, Advisors
from citybuilder_env.items import GeneratorConfig, MarginalSpec, generate_catalog
from citybuilder_env.recommend import RecommendationManager
from citybuilder_env.score import ScoreManager
from citybuilder_env.reward import RewardManager
from citybuilder_env import selection as sel
from citybuilder_env.utils.rng import DeterministicRNG, hash_uint64
from citybuilder_env.pareto import compute_pareto_frontier_and_distance
from citybuilder_env.io.logging import EpisodeLogger


@dataclass
class StepResult:
    observation: Dict
    reward: float
    done: bool
    info: Dict
    action_mask: np.ndarray


class CityBuilderEnv:
    """
    Deterministic, lean environment for RL training/evaluation.

    Parameters
    ----------
    gcfg : GeneratorConfig
        Gaussian-copula generator settings (catalog-level)
    ecfg : EpisodeConfig
        Episode settings (budgeting, limits, capacity bins, etc.)
    master_seed : int
        Root seed controlling the entire episode sequence (catalogs, solvers).
    cfg_fingerprint : str
        A short stable string/hash tying seeds to config (change when gcfg/ecfg change).
    mode : str
        "train" or "eval" (seed namespacing & logging only; distribution identical)
    """

    def __init__(
        self,
        gcfg: GeneratorConfig,
        ecfg: EpisodeConfig,
        master_seed: int,
        cfg_fingerprint: str = "v1",
        mode: str = "train",
    ):
        self.gcfg = gcfg
        self.ecfg = ecfg
        self.cfg_fp = str(cfg_fingerprint)
        self.mode = str(mode)

        # Namespace the RNG stream by mode for clarity (seed already hashed in make_env)
        self.master = DeterministicRNG(seed=int(master_seed), stream=f"env/{self.mode}")
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
        self.reward_mgr = RewardManager(K=self.ecfg.K, normalization_mode="max")

        self.logger: Optional[EpisodeLogger] = None
        self.cfg_fingerprint = self.cfg_fp

        # Precompute the capacity × antithetic category list
        self._cats: List[Tuple[str, int]] = self._build_categories()

    # ----------------------
    # Lifecycle
    # ----------------------

    def set_logger(self, logger: EpisodeLogger) -> None:
        self.logger = logger

    def seed(self, seed: int) -> None:
        """Reset the master RNG and episode counter."""
        self.master = DeterministicRNG(seed=int(seed), stream=f"env/{self.mode}")
        self.episode_idx = 0
        self.episode_id = 0

    # ----------------------
    # Episode helpers
    # ----------------------

    def _build_categories(self) -> List[Tuple[str, int]]:
        """
        Build the interleaved category list:
        If use_antithetic_in_bin: [(tight,+1),(mid,+1),(loose,+1),(tight,-1),(mid,-1),(loose,-1)]
        Else:                      [(tight,+1),(mid,+1),(loose,+1)]
        """
        bins = [name for (name, _rng) in self.ecfg.category_bins]
        if self.ecfg.use_antithetic_in_bin:
            plus = [(b, +1) for b in bins]
            minus = [(b, -1) for b in bins]
            return plus + minus
        else:
            return [(b, +1) for b in bins]

    def _pick_tau_for_episode(self, bin_name: str) -> float:
        """
        Stratified tau within [lo, hi) for the given bin.
        Uses episode_idx to produce even coverage over time.
        """
        # Find the bin bounds
        bin_map = {name: (float(lo), float(hi)) for (name, (lo, hi)) in self.ecfg.category_bins}
        if bin_name not in bin_map:
            raise KeyError(f"Unknown capacity bin '{bin_name}' in EpisodeConfig.category_bins")
        lo, hi = bin_map[bin_name]
        eps = getattr(self.ecfg, "tau_edge_eps", 1e-3)
        lo = lo + eps
        hi = hi - eps
        hi = max(hi, lo + 1e-9)  # guard tiny intervals

        if self.ecfg.stratify_tau:
            S = max(int(getattr(self.ecfg, "tau_strata", 32)), 1)
            # visits_to_cycle = how many complete cycles of the category list we've done
            visits_to_cycle = self.episode_idx // max(len(self._cats), 1)
            slot = visits_to_cycle % S
            tau = lo + ((slot + 0.5) / S) * (hi - lo)
        else:
            # Deterministic uniform via a per-episode RNG
            tau_rng = self.master.child("tau", bin_name, self.episode_idx, self.cfg_fp)
            tau = float(tau_rng.gen.uniform(low=lo, high=hi))
        # clamp to (lo, hi)
        tau = min(max(tau, lo), hi)
        return tau

    # ----------------------
    # Gym-style API
    # ----------------------

    def reset(self) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Start a new episode:
            0) Choose category (capacity bin, antithetic sign)
            1) Generate catalog (deterministic; honors antithetic sign)
            2) Derive budget from realized mass: W = tau * sum_i w_i
            3) Initialize state and advisors
        Returns (observation, action_mask, info)
        """
        # 0) Category for this episode (interleaved)
        cat_idx = self.episode_idx % len(self._cats)
        bin_name, sign = self._cats[cat_idx]
        tau = self._pick_tau_for_episode(bin_name)

        # 1) Deterministic catalog stream for this episode
        cat_rng = self.master.child("catalog", self.episode_idx, self.cfg_fp)
        catalog, expected_cost, total_cost_float = generate_catalog(self.gcfg, cat_rng, antithetic_sign=sign)

        # Sequential, deterministic name
        ep_no = self.episode_id + 1
        try:
            catalog = replace(catalog, name=f"catalog_{ep_no:04d}")
        except TypeError:
            catalog.name = f"catalog_{ep_no:04d}"

        self.catalog = catalog
        self.costs_int = catalog.costs_array()
        self.objs_int = catalog.objs_matrix()
        N = int(catalog.N)

        # 2) Budget from REALIZED mass (feasibility-safe integerization)
        int_scale = int(catalog.int_scale)
        W_float = float(tau) * float(total_cost_float)
        budget_int = int(np.floor(W_float * int_scale))
        if budget_int < 1:
            budget_int = 1

        # Optional legacy budgeting (explicitly enabled only)
        if getattr(self.ecfg, "use_expected_budget", False):
            budget_int = int(self.ecfg.derive_budget_int(expected_cost))

        self._initial_budget_int = int(budget_int)

        # 3) Initialize episode state
        remaining_ids = catalog.ids_array()  # 0..N-1 sorted
        scores0 = np.zeros(self.ecfg.K, dtype=np.int64)
        self.state = EpisodeState(
            catalog=catalog,
            budget_rem_int=int(budget_int),
            scores_cum_int=scores0,
            remaining_ids=remaining_ids,
            selected_ids=[],
            step_idx=0,
            advisors=Advisors.empty(self.ecfg.K),  # replaced below
        )

        # 4) RecommendationManager (deterministic base seed)
        rec_seed = int(hash_uint64(self.master.seed, "ilp", self.episode_idx, self.cfg_fp))
        self.rec_mgr = RecommendationManager(
            K=self.ecfg.K,
            time_limit_ms=self.ecfg.solver_time_limit_ms,
            cache_size=self.ecfg.cache_size,
            budget_bucket=self.ecfg.budget_bucket,
            base_seed=rec_seed,
        )

        # 5) Initial advisors for R_0
        advisors, _ = self.rec_mgr.compute_advisors(
            remaining_ids=self.state.remaining_ids,
            budget_rem_int=self.state.budget_rem_int,
            costs_int=self.costs_int,
            objs_int=self.objs_int,
        )
        self.state.advisors = advisors

        # 6) Prepare score ledger & reward trackers
        self.score_mgr.reset()
        self.reward_mgr.reset()

        # 7) First observation & mask
        obs = self._make_observation()
        mask = sel.build_action_mask(self.state, self.costs_int)

        # 8) Bookkeeping
        self.episode_id += 1
        self.episode_idx += 1

        info = {
            "episode_id": self.episode_id,
            "mode": self.mode,
            "catalog_seed": int(cat_rng.seed),
            "catalog_name": self.catalog.name,
            "budget_int": int(budget_int),
            "tau": float(tau),
            "capacity_bin": bin_name,
            "antithetic_sign": int(sign),
        }

        if self.logger is not None:
            meta = dict(info)
            meta.update({
                "cfg_fingerprint": self.cfg_fp,
                "K": int(self.ecfg.K),
                "int_scale": int(self.catalog.int_scale),
                "initial_budget_int": int(self._initial_budget_int),
            })
            self.logger.start_episode(meta)

        return obs, mask, info

    def step(self, action_id: int) -> StepResult:
        """
        One environment step. Mask-enforced: illegal actions return no mutation and reward=0
        """
        assert self.state is not None and self.costs_int is not None and self.objs_int is not None and self.rec_mgr is not None

        # 1. Build current mask and validate action
        mask = sel.build_action_mask(self.state, self.costs_int)
        if action_id < 0 or action_id >= mask.size or not bool(mask[action_id]):
            obs = self._make_observation()
            info = {"illegal_action": int(action_id)}
            return StepResult(observation=obs, reward=0.0, done=False, info=info, action_mask=mask)

        # 2. Snapshot pre-decision context for reward
        advisors_pre = self.state.advisors
        ref_ids = self.state.remaining_ids.copy()  # R_t
        budget_before_int = int(self.state.budget_rem_int)

        # 3. Accounting (ints)
        score_info = self.score_mgr.apply_selection(self.state, self.costs_int, self.objs_int, action_id)

        # 4. Mutate remaining set
        sel.apply_selection(self.state, action_id)

        # 5. Reward (use advisors_pre & R_t)
        reward, details = self.reward_mgr.compute_reward(
            selected_id=action_id,
            remaining_ref_ids=ref_ids,
            objs_int=self.objs_int,
        )

        # 6. Termination check
        done_check = sel.check_done(self.state, self.ecfg.max_steps)
        done = done_check.done

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

        # 8. Build next observation & mask
        obs_next = self._make_observation()
        mask_next = sel.build_action_mask(self.state, self.costs_int)

        item_objs_int = self.objs_int[action_id, :].astype(int, copy=True)

        info = {
            "cost_int": score_info["cost_int"],
            "item_objs_int": item_objs_int.tolist(),
            "budget_before_int": budget_before_int,
            "budget_after_int": score_info["budget_after_int"],
            "advisor_freq_selected": advisors_pre.freq.get(int(action_id), 0),
            "reward_details": {
                "norm_components": details.norm_components,
                "max_vec_int": details.max_vec_int,
                "mode": details.mode,
            },
            "done_reason": done_check.reason,
            "recommendation_stats": (None if rec_stats is None else {
                "cache_hit": rec_stats.cache_hit,
                "total_wall_ms": rec_stats.total_wall_ms,
                "per_objective_status": rec_stats.per_objective_status,
            }),
        }

        result = StepResult(observation=obs_next, reward=float(reward), done=done, info=info, action_mask=mask_next)

        if self.logger is not None:
            self.logger.log_step({
                "episode_id": self.episode_id,
                "step_idx": int(self.state.step_idx),
                "action_id": int(action_id),
                "reward": float(reward),
                "info": info,
            })

        if done and self.logger is not None:
            summary = self.episode_summary()
            self.logger.end_episode(summary)

        return result

    # -----------------
    # Summaries
    # -----------------

    def episode_summary(self) -> Dict:
        """Light-weight end-of-episode summary (call after done=True)."""
        assert self.state is not None and self.catalog is not None

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
            "num_steps": int(self.state.step_idx),
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

    def close(self) -> None:
        if self.logger is not None:
            try:
                self.logger.close()
            except Exception:
                pass

    # ------------------
    # Internals
    # ------------------

    def _make_observation(self) -> Dict:
        """Assemble the observation dict the agent will consume."""
        assert self.state is not None and self.costs_int is not None and self.objs_int is not None and self.catalog is not None
        N = self.costs_int.shape[0]
        adv_freq = np.zeros(N, dtype=np.int64)
        for iid, f in self.state.advisors.freq.items():
            adv_freq[int(iid)] = int(f)

        remaining_list_of_dicts = []
        items_by_id = {item.id: item for item in self.catalog.items}
        for item_id in self.state.remaining_ids:
            item = items_by_id[item_id]
            cost_f, obj_f = item.float_view(self.catalog.int_scale)
            remaining_list_of_dicts.append({
                "id": item.id,
                "cost": cost_f,
                "v": obj_f.tolist(),
            })

        return {
            "costs_int": self.costs_int,
            "objs_int": self.objs_int,
            "advisor_freq": adv_freq,
            "budget_rem_int": int(self.state.budget_rem_int),
            "scores_cum_int": self.state.scores_cum_int.copy(),
            "step_idx": int(self.state.step_idx),
            "budget": float(self.state.budget_rem_int) / float(self.catalog.int_scale),
            "initial_budget": float(self._initial_budget_int) / float(self.catalog.int_scale),
            "remaining": remaining_list_of_dicts,
        }


# ----------------------------------------------------------
# Convenience factory (no gcfg/ecfg required by the agent)
# ----------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # PyYAML
    except Exception as e:
        raise RuntimeError("PyYAML is required to load the environment config.") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _configs_from_yaml(cfg: dict) -> Tuple[GeneratorConfig, EpisodeConfig]:
    """
    Expect the YAML to have top-level sections 'generator' and 'episode'
    whose keys match the dataclasses for GeneratorConfig and EpisodeConfig.
    """
    gen_config_dict = cfg.get("generator", {})
    epi_config_dict = cfg.get("episode", {})

    # Map legacy 'marginals' block to explicit fields
    if 'marginals' in gen_config_dict:
        marginals_data = gen_config_dict.pop('marginals')
        if 'cost' in marginals_data:
            gen_config_dict['cost_marginal'] = marginals_data['cost']
        if 'objectives' in marginals_data:
            gen_config_dict['objective_marginals'] = marginals_data['objectives']

    # Coerce MarginalSpec and sigma shape
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

    # Episode int_scale inherits generator int_scale if not present
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
    mode: str = "train",
) -> CityBuilderEnv:
    """
    Build a CityBuilderEnv without the agent providing gcfg/ecfg.

    Loads GeneratorConfig and EpisodeConfig from YAML.
    The agent passes a deterministic 'seed' and a 'mode' ("train"/"eval").

    Example
    -------
    >>> import citybuilder_env as cbe
    >>> env = cbe.make_env(seed=1234, mode="train")
    >>> obs, mask, info = env.reset()
    """
    package_dir = Path(__file__).parent

    # Resolve config file
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

    # Hash the mode into the root seed (Option 2: guarantees disjoint train/eval)
    root_seed = int(hash_uint64(int(seed), str(mode)))

    # Construct env
    env = CityBuilderEnv(
        gcfg=gcfg,
        ecfg=ecfg,
        master_seed=root_seed,
        cfg_fingerprint=str(cfg_fingerprint),
        mode=str(mode),
    )
    return env
