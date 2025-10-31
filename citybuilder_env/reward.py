# reward.py
"""
RewardManager: compute per-step reward from normalized objective gains ONLY.

Definition
----------
At step t, for selected item i and objectives v_k(i):

    1) Compute per-objective normalization reference over the *pre-removal*
       remaining set R_t:
           m_k(t) = max_{j in R_t} v_k(j)        (or its EMA variant)

    2) Normalize the chosen item's objectives component-wise:
           \tilde v_k(i) = v_k(i) / max(m_k(t), eps)   ∈ [0, 1]

    3) Reward is the sum of normalized gains (no advisor shaping):
           r_t = sum_k \tilde v_k(i)                  ∈ [0, K]

Notes
-----
- Normalization mode:
    * "max":    per-step maximum over R_t (fast, reactive).
    * "ema":    EMA over the per-step maxima (smooth, less reactive).
- All objectives are treated as "maximize".
- This module is intentionally policy-agnostic: advisor signals are NOT used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal

import numpy as np

from citybuilder_env.utils.normalization import (
    remaining_max,
    normalize_item_by_max,
    EMAMaxTracker,
)


@dataclass(frozen=True)
class RewardDetails:
    """Diagnostics for logging/analysis."""
    norm_components: tuple      # length K, each in [0,1]
    max_vec_int: tuple          # reference maxima used for normalization (ints for readability)
    mode: str                   # "max" or "ema"


class RewardManager:
    """
    Compute rewards with direction-agnostic normalization (all maximize).

    Parameters
    ----------
    K : int
        Number of objectives (>=1).
    normalization_mode : {"max", "ema"}
        - "max": use per-step max over the reference remaining set R_t.
        - "ema": track an EMA of the per-step max for smoother normalization.
    ema_decay : float
        Only used if normalization_mode == "ema"; closer to 1.0 = slower updates.
    """

    def __init__(
        self,
        K: int,
        normalization_mode: Literal["max", "ema"] = "max",
        ema_decay: float = 0.9,
    ):
        if K <= 0:
            raise ValueError("K must be positive")
        if normalization_mode not in ("max", "ema"):
            raise ValueError('normalization_mode must be "max" or "ema"')

        self.K = int(K)
        self.mode = normalization_mode
        self._ema_tracker = EMAMaxTracker(K=self.K, decay=ema_decay) if self.mode == "ema" else None

    # ---------- Core API -------------

    def compute_reward(
        self,
        selected_id: int,
        remaining_ref_ids: np.ndarray,
        objs_int: np.ndarray,
    ) -> Tuple[float, RewardDetails]:
        """
        Compute reward for selecting 'selected_id' given a reference remaining set R_t.

        Parameters
        ----------
        selected_id : int
            The chosen item id (should be present in remaining_ref_ids if you use pre-removal R_t).
        remaining_ref_ids : np.ndarray[int64]
            Reference remaining set used to compute per-objective maxima.
            Typically this is the set at decision time (pre-removal).
        objs_int : np.ndarray[int64] shape (N, K)
            Integerized objective matrix for the catalog.

        Returns
        -------
        (reward, details)
            reward : float in [0, K]
            details: RewardDetails(norm_components, max_vec_int, mode)
        """
        if objs_int.ndim != 2 or objs_int.shape[1] != self.K:
            raise ValueError(f"objs_int must be (N, {self.K})")

        # Degenerate reference set -> zero reward, zero maxima
        if remaining_ref_ids.size == 0:
            zeros = tuple(0.0 for _ in range(self.K))
            maxs  = tuple(0 for _ in range(self.K))
            return 0.0, RewardDetails(norm_components=zeros, max_vec_int=maxs, mode=self.mode)

        # 1) Reference maxima (either per-step max or EMA of that max)
        ref_max = remaining_max(objs_int, remaining_ref_ids)  # int64[K]
        v = objs_int[int(selected_id), :].astype(np.int64, copy=False)

        if self.mode == "ema":
            assert self._ema_tracker is not None
            ref_max = self._ema_tracker.update(ref_max)       # float[K]
            # Normalize with EMA (float vector)
            denom = np.maximum(ref_max.astype(np.float64, copy=False), 1e-12)
            norm = np.clip(v.astype(np.float64) / denom, 0.0, 1.0)
            # For readability in logs, round EMA maxes to nearest int
            max_vec_for_details = tuple(int(x) for x in np.round(ref_max).astype(np.int64).tolist())
        else:
            # "max" mode: normalize by integer max (fast and simple)
            norm = normalize_item_by_max(v, ref_max, eps=1e-12)   # float[K] in [0,1]
            max_vec_for_details = tuple(int(x) for x in ref_max.tolist())

        # 2) Reward: sum of normalized gains (no advisor shaping)
        reward = float(np.sum(norm, dtype=np.float64))

        details = RewardDetails(
            norm_components=tuple(float(x) for x in norm.tolist()),
            max_vec_int=max_vec_for_details,
            mode=self.mode,
        )
        return reward, details

    # -------- Utilities ----------------

    def theoretical_bounds(self) -> Tuple[float, float]:
        """
        Return (min_reward, max_reward) for current K.
        """
        lo = 0.0
        hi = float(self.K)
        return lo, hi

    def reset(self) -> None:
        """
        Reset internal trackers (for EMA mode) at episode start.
        """
        if self._ema_tracker is not None:
            self._ema_tracker = EMAMaxTracker(
                K=self.K,
                decay=self._ema_tracker.decay,
                init_eps=self._ema_tracker.init_eps,
            )
