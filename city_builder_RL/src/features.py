from typing import List, Tuple, Any
import numpy as np
import torch

class FeatureExtractor:
    """
    Produces fixed-size state features and per-item features from raw observations.
    Uses per-episode pool size N0 to compute remaining_frac = |R_t| / N0.
    """

    def __init__(self, *, eps: float, initial_budget: float, num_objectives: int, max_items_hint: int):
        self.eps = float(eps)
        self.initial_budget = float(initial_budget)
        self.K = int(num_objectives)
        self.max_items_hint = int(max_items_hint)              # fallback
        self.episode_max_items_hint = int(max_items_hint)      # set per episode

    def set_episode_pool_size(self, n0: int):
        """Call once after /reset with len(observation['remaining'])."""
        self.episode_max_items_hint = max(1, int(n0))

    @staticmethod
    def _pooled_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if values.size == 0:
            return (np.zeros(0), np.zeros(0), np.zeros(0))
        vmax = values.max(axis=0)
        vmean = values.mean(axis=0)
        vstd = values.std(axis=0)
        return vmax, vmean, vstd

    def state_features(self, obs: dict) -> torch.Tensor:
        budget = float(obs["budget"])
        remaining = obs["remaining"]
        R = len(remaining)

        V = np.array([it["v"] for it in remaining], dtype=float) if R > 0 else np.zeros((0, self.K))
        vmax, vmean, vstd = self._pooled_stats(V)

        adv = obs.get("advisor_sets", {})
        adv_lists = list(adv.values())
        if len(adv_lists) == 0:
            advisor_frac = np.zeros(self.K, dtype=float)
        else:
            K_eff = len(adv_lists)
            # Each lst is already subset of remaining as per server contract
            advisor_frac = np.array([len(lst) / max(R, 1) for lst in adv_lists], dtype=float)
            if K_eff < self.K:
                advisor_frac = np.pad(advisor_frac, (0, self.K - K_eff))
            elif K_eff > self.K:
                advisor_frac = advisor_frac[: self.K]

        b_frac = budget / max(1.0, self.initial_budget)
        r_frac = R / max(1.0, self.episode_max_items_hint)   # per-episode N0

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

        state_vec = np.concatenate([
            np.array([b_frac, r_frac], dtype=float),
            vmax, vmean, vstd, advisor_frac
        ]).astype(np.float32)
        return torch.from_numpy(state_vec)  # [S = 2 + 4K]

    def item_features(self, obs: dict) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        budget = float(obs["budget"])
        remaining = obs["remaining"]
        ids = [it["id"] for it in remaining]

        costs = np.array([it["cost"] for it in remaining], dtype=float) if len(remaining) > 0 else np.zeros((0,))
        V = np.array([it["v"] for it in remaining], dtype=float) if len(remaining) > 0 else np.zeros((0, self.K))

        vmax = V.max(axis=0) if V.size > 0 else np.ones(self.K)
        vmax[vmax == 0] = 1.0
        Vnorm = V / (vmax + self.eps)

        adv = obs.get("advisor_sets", {})
        r_sets = [set(lst) for lst in adv.values()]
        K_eff = len(r_sets)
        if K_eff == 0:
            phi = np.zeros(len(ids), dtype=float)
        else:
            phi = np.array([sum(int(i in s) for s in r_sets) / float(K_eff) for i in ids], dtype=float)

        ratio = V / (costs[:, None] + self.eps) if len(remaining) > 0 else np.zeros((0, self.K))
        cost_norm = costs / (budget + self.eps) if len(remaining) > 0 else np.zeros((0,))

        item_x = np.concatenate([
            cost_norm[:, None],
            Vnorm,
            phi[:, None],
            ratio
        ], axis=1).astype(np.float32)

        mask = (costs <= budget).astype(np.float32) if len(remaining) > 0 else np.zeros((0,), dtype=np.float32)
        return torch.from_numpy(item_x), ids, torch.from_numpy(mask)
