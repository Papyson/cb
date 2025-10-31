# city_builder_rl/features.py
from typing import List, Tuple, Any, Dict
import hashlib
import numpy as np
import torch

class FeatureExtractor:
    """
    Produces fixed-size state features and per-item features from raw observations.
    Uses per-episode pool size N0 to compute remaining_frac = |R_t| / N0.

    Advisor-derived features are HARD-gated by `use_advisors`.
    When advisors are disabled, advisor columns are omitted (not zero-filled),
    so item/state dimensions truly change between modes.
    """

    def __init__(
        self,
        *,
        eps: float,
        initial_budget: float,
        num_objectives: int,
        max_items_hint: int,
        use_advisors: bool = True,       # NO-ADVISOR(False) vs FULL(True)
        use_slate_overlap: bool = True,  # Only meaningful when use_advisors=True
    ):
        self.eps = float(eps)
        self.initial_budget = float(initial_budget)
        self.K = int(num_objectives)
        self.max_items_hint = int(max_items_hint)              # fallback
        self.episode_max_items_hint = int(max_items_hint)      # set per episode
        self.use_advisors = bool(use_advisors)
        self.use_slate_overlap = bool(use_slate_overlap)

        # Pre-compute and expose dims
        self.state_dim, self.item_dim = self._compute_dims()

        # Cache a signature string for checkpoint sanity
        self._signature = self._compute_signature()

    # ------------------------ public metadata ------------------------

    def get_dims(self) -> Tuple[int, int]:
        """Return (state_dim, item_dim)."""
        return self.state_dim, self.item_dim

    def feature_signature(self) -> str:
        """
        A short string capturing the active feature set (mode, K, flags).
        Save this next to model checkpoints; verify on load.
        """
        return self._signature

    # ------------------------ episode control ------------------------

    def set_episode_pool_size(self, n0: int):
        """Call once after /reset with len(observation['remaining'])."""
        self.episode_max_items_hint = max(1, int(n0))

    # ------------------------ internal: dims/signature ------------------------

    def _compute_dims(self) -> Tuple[int, int]:
        """
        Item features (in order):
          [ slack(1), Vnorm(K), ratio(K), var_i(1), ranks(K), (phi_i?1) ]
        State features (in order):
          [ b_frac(1), r_frac(1), tau(1), objective_conflict(1),
            vmax(K), vmean(K), vstd(K),
            (centered advisor_frac_k?K), (slate_overlap?1) ]
        """
        K = self.K

        # Item dim
        base_item = 1 + K + K + 1 + K  # slack + Vnorm + ratio + var + ranks
        item_dim = base_item + (1 if self.use_advisors else 0)  # phi_i

        # State dim
        base_state = 4 + 3*K  # b_frac, r_frac, tau, objective_conflict + (vmax,vmean,vstd)
        add_adv = K if self.use_advisors else 0  # centered advisor_frac_k
        add_overlap = 1 if (self.use_advisors and self.use_slate_overlap) else 0
        state_dim = base_state + add_adv + add_overlap

        return state_dim, item_dim

    def _compute_signature(self) -> str:
        payload = {
            "K": self.K,
            "use_advisors": self.use_advisors,
            "use_slate_overlap": self.use_slate_overlap,
            "state_dim": self.state_dim,
            "item_dim": self.item_dim,
            "order_item": ["slack", "Vnorm[K]", "ratio[K]", "var", "ranks[K]", "phi?"],
            "order_state": ["b_frac", "r_frac", "tau", "objective_conflict",
                            "vmax[K]", "vmean[K]", "vstd[K]",
                            "centered_advisor_frac[K]?", "slate_overlap?"],
        }
        s = repr(payload).encode("utf-8")
        return "feat_" + hashlib.sha1(s).hexdigest()[:12]

    # ------------------------ small utilities ------------------------

    @staticmethod
    def _pooled_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if values.size == 0:
            return (np.zeros(0), np.zeros(0), np.zeros(0))
        vmax = values.max(axis=0)
        vmean = values.mean(axis=0)
        vstd = values.std(axis=0)
        return vmax, vmean, vstd

    @staticmethod
    def _safe_sum_costs(remaining: List[Dict[str, Any]]) -> float:
        if not remaining:
            return 0.0
        return float(np.sum([float(it["cost"]) for it in remaining]))

    @staticmethod
    def _normalize_ranks(column: np.ndarray) -> np.ndarray:
        """
        Returns normalized ranks in [0,1] for a 1D array where higher value -> higher rank.
        If all equal or length <=1, returns zeros.
        """
        n = column.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=float)
        # argsort twice to get ranks; tie-aware using stable sort
        order = np.argsort(column, kind="mergesort")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        # If all equal, variance=0 -> return zeros
        if np.allclose(column, column[0]):
            return np.zeros(n, dtype=float)
        # scale to [0,1]
        if n > 1:
            return ranks.astype(float) / float(n - 1)
        return np.zeros(n, dtype=float)

    @staticmethod
    def _mean_pairwise_corr(mat: np.ndarray) -> float:
        """
        mat: [R, K] values for remaining items.
        Returns mean of upper-triangular Pearson correlations across objectives.
        If degenerate (<3 rows or zero-variance columns), returns 0.0.
        """
        R, K = mat.shape if mat.ndim == 2 else (0, 0)
        if R < 3 or K <= 1:
            return 0.0
        # If any objective column has ~zero variance, corr is ill-defined; return 0
        if np.any(np.isclose(mat.var(axis=0), 0.0)):
            return 0.0
        corr = np.corrcoef(mat, rowvar=False)  # [K,K]
        if not np.all(np.isfinite(corr)):
            return 0.0
        iu = np.triu_indices(K, k=1)
        if iu[0].size == 0:
            return 0.0
        return float(np.mean(corr[iu]))

    # ---------- helpers to read advisor signals from different obs schemas ----------

    def _advisor_state_fraction(
        self, obs: Dict[str, Any], remaining_ids: List[int]
    ) -> np.ndarray:
        """
        Returns length-K vector in [0,1]: fraction of remaining items endorsed per advisor.
        If advisors disabled or not derivable, returns zeros(K).
        """
        if not self.use_advisors:
            return np.zeros(self.K, dtype=float)

        # HTTP-style: obs["advisor_sets"] = {k: [ids...], ...}
        adv_sets = obs.get("advisor_sets")
        if isinstance(adv_sets, dict):
            R = max(1, len(remaining_ids))
            lists = list(adv_sets.values())
            K_eff = len(lists)
            if K_eff == 0:
                return np.zeros(self.K, dtype=float)
            fracs = np.array([len(set(lst).intersection(remaining_ids)) / float(R) for lst in lists], dtype=float)
            if K_eff < self.K:
                fracs = np.pad(fracs, (0, self.K - K_eff))
            elif K_eff > self.K:
                fracs = fracs[: self.K]
            return fracs

        # Env-style: obs["advisor_freq"] = np.int64[N], frequency per item
        adv_freq = obs.get("advisor_freq")
        if isinstance(adv_freq, (list, np.ndarray)):
            if len(remaining_ids) == 0:
                frac_scalar = 0.0
            else:
                adv_freq = np.asarray(adv_freq)
                mask = np.zeros_like(adv_freq, dtype=bool)
                mask[np.array(remaining_ids, dtype=int)] = True
                # fraction of remaining items recommended by >=1 advisor
                frac_scalar = float(np.count_nonzero(adv_freq[mask] > 0)) / float(max(1, len(remaining_ids)))
            return np.full(self.K, frac_scalar, dtype=float)

        # Default: no advisor info
        return np.zeros(self.K, dtype=float)

    def _advisor_item_phi(
        self, obs: Dict[str, Any], item_ids: List[int]
    ) -> np.ndarray:
        """
        Returns length-N vector in [0,1]: per-item advisor "support" (fraction of advisors endorsing it).
        If advisors disabled or not derivable, returns zeros(N).
        """
        N = len(item_ids)
        if not self.use_advisors or N == 0:
            return np.zeros(N, dtype=float)

        adv_sets = obs.get("advisor_sets")
        if isinstance(adv_sets, dict):
            sets = [set(lst) for lst in adv_sets.values()]
            K_eff = len(sets)
            if K_eff == 0:
                return np.zeros(N, dtype=float)
            return np.array([sum(int(i in s) for s in sets) / float(K_eff) for i in item_ids], dtype=float)

        adv_freq = obs.get("advisor_freq")
        if isinstance(adv_freq, (list, np.ndarray)):
            adv_freq = np.asarray(adv_freq, dtype=float)
            K_eff = float(self.K) if self.K > 0 else 1.0
            raw = np.array([adv_freq[int(i)] for i in item_ids], dtype=float)
            return np.clip(raw / max(1.0, K_eff), 0.0, 1.0)

        return np.zeros(N, dtype=float)

    def _advisor_slate_overlap(self, obs: Dict[str, Any]) -> float:
        """
        Jaccard-style overlap: |∩ S_k| / |∪ S_k| over advisor sets at current step.
        Returns 0.0 if not derivable or empty.
        """
        if not self.use_advisors:
            return 0.0
        adv_sets = obs.get("advisor_sets")
        if not isinstance(adv_sets, dict) or len(adv_sets) == 0:
            return 0.0
        sets = [set(v) for v in adv_sets.values()]
        union = set().union(*sets)
        if len(union) == 0:
            return 0.0
        inter = set(sets[0])
        for s in sets[1:]:
            inter &= s
        return float(len(inter)) / float(len(union))

    # ----------------- public API: feature extraction -----------------

    def state_features(self, obs: dict) -> torch.Tensor:
        budget = float(obs["budget"])
        remaining = obs["remaining"]
        R = len(remaining)

        # Matrix of item objective values: [R, K]
        V = np.array([it["v"] for it in remaining], dtype=float) if R > 0 else np.zeros((0, self.K))

        vmax, vmean, vstd = self._pooled_stats(V)

        remaining_ids = [it["id"] for it in remaining]

        # Core scalars
        b_frac = budget / max(1.0, self.initial_budget)
        r_frac = R / max(1.0, self.episode_max_items_hint)

        # Tightness tau = B_t / sum_j C_j
        sum_costs = self._safe_sum_costs(remaining)
        tau = budget / max(self.eps, float(sum_costs))

        # Objective conflict = mean pairwise corr across objectives on remaining items
        objective_conflict = self._mean_pairwise_corr(V) if R > 0 else 0.0

        # Pool stats must be length-K even if V is empty
        def _ensure_len_k(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return np.zeros(self.K, dtype=float)
            if arr.shape[0] == self.K:
                return arr
            if arr.shape[0] < self.K:
                return np.pad(arr, (0, self.K - arr.shape[0]))
            return arr[: self.K]

        vmax = _ensure_len_k(vmax)
        vmean = _ensure_len_k(vmean)
        vstd = _ensure_len_k(vstd)

        # Assemble state vector
        parts = [
            np.array([b_frac, r_frac, tau, objective_conflict], dtype=float),
            vmax, vmean, vstd
        ]

        if self.use_advisors:
            # Centered advisor prevalence vector
            advisor_frac_raw = self._advisor_state_fraction(obs, remaining_ids)  # length K
            adv_mean = float(advisor_frac_raw.mean()) if advisor_frac_raw.size > 0 else 0.0
            advisor_frac_centered = advisor_frac_raw - adv_mean
            parts.append(advisor_frac_centered.astype(float))

            if self.use_slate_overlap:
                overlap = self._advisor_slate_overlap(obs)
                parts.append(np.array([overlap], dtype=float))

        state_vec = np.concatenate(parts).astype(np.float32)
        return torch.from_numpy(state_vec)  # shape: [state_dim]

    def item_features(self, obs: dict) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        budget = float(obs["budget"])
        remaining = obs["remaining"]
        ids = [it["id"] for it in remaining]

        R = len(remaining)
        costs = np.array([it["cost"] for it in remaining], dtype=float) if R > 0 else np.zeros((0,))
        V = np.array([it["v"] for it in remaining], dtype=float) if R > 0 else np.zeros((0, self.K))

        # Normalization across remaining items
        vmax = V.max(axis=0) if V.size > 0 else np.ones(self.K)
        vmax[vmax == 0] = 1.0
        Vnorm = V / (vmax + self.eps) if V.size > 0 else np.zeros((0, self.K))

        # Slack ratio (a.k.a normalized cost vs budget)
        slack = costs / (budget + self.eps) if R > 0 else np.zeros((0,))

        # Efficiency ratios per objective
        ratio = V / (costs[:, None] + self.eps) if R > 0 else np.zeros((0, self.K))

        # Value dispersion across objectives (compute on Vnorm to be scale-free)
        var_i = Vnorm.var(axis=1) if R > 0 else np.zeros((0,))

        # Per-objective normalized ranks in [0,1]
        ranks = []
        if R > 0:
            for k in range(self.K):
                ranks.append(self._normalize_ranks(V[:, k]))
            ranks = np.stack(ranks, axis=1)  # [R, K]
        else:
            ranks = np.zeros((0, self.K))

        # Advisor consensus per item (phi_i) if enabled
        if self.use_advisors:
            phi = self._advisor_item_phi(obs, ids)  # length R
        else:
            phi = np.zeros(R, dtype=float)  # not appended; keeps code simple below

        # Compose item feature matrix in the agreed order
        cols = [
            slack[:, None],   # 1
            Vnorm,            # K
            ratio,            # K
            var_i[:, None],   # 1
            ranks,            # K
        ]
        if self.use_advisors:
            cols.append(phi[:, None])  # +1

        item_x = np.concatenate(cols, axis=1).astype(np.float32) if R > 0 else np.zeros((0, self.item_dim), dtype=np.float32)

        # Feasibility mask (action mask): item is selectable if cost <= budget
        mask = (costs <= budget).astype(np.float32) if R > 0 else np.zeros((0,), dtype=np.float32)

        return torch.from_numpy(item_x), ids, torch.from_numpy(mask)
