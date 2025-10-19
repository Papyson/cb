# reward.py

"""
RewardManager: compute per-step reward from normalized objective gains
and advisor-frequency shaping.

Reward definition
----------------------
At step t, for selected item i:
    1. Compute per-objective normalized gains
        \tilde v_k(i) = v_k(i) / max_{j in remaining_ref} v_k(j)
        using integers, then cast to float in [0,1].
        remaining_ref is typically the set of remaining items at decision time
        (i.e., before removing the selected item). Pass that set explicitly.
    2. Let φ_t(i) be the advisor frequency for item i in {0..K}.
    3. Reward:
        r_t = sum_k \tilde v_k(i) + β * φ_t(i)

All objectives are treated as "maximize".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from .state import Advisors
from .utils.normalization import remaining_max, normalize_item_by_max, EMAMaxTracker

@dataclass(frozen=True)
class RewardDetails:
    """Rich diagnostics for logging/analysis."""
    norm_components: tuple      #Length K, each in [0,1]
    advisor_freq: int           # φ_t(i) in [0..K]
    max_vec_int: tuple          # reference maxima used for normalization
    mode: str                   # "max" or "ema"

class RewardManager:
    """
    Compute rewards with direction-agnostoc normalization (all - maximize) and advisor shaping.

    Paramters
    -----------
    K: int 
        Number of objectives.
    beta: float
        Advisor shaping coefficient (>= 0)
    normalization_mode: {"max", "ema"}
        - "max": use per-step max over remaining_ref.
        - "ema": track an EMA of the per-step max for smoother normalization.
    ema_decay: float
        Only used if normalization_mode == "ema"; closer to 1.0 = slower updates.
    """

    def __init__(
            self,
            K: int,
            beta: float,
            normalization_mode: Literal["max", "ema"] = "max",
            ema_decay: float = 0.9,
    ):
        if K<= 0:
            raise ValueError("K must be positive")
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        if normalization_mode not in ("max", "ema"):
            raise ValueError('normalization_mode must be "max" or "ema"')
        
        self.K = int(K)
        self.beta = float(beta)
        self.mode = normalization_mode
        self._ema_tracker = EMAMaxTracker(K=self.K, decay=ema_decay) if self.mode == "ema" else None

    # ---------- Core API -------------

    def compute_reward(
            self,
            selected_id: int,
            remaining_ref_ids: np.ndarray,
            objs_int: np.ndarray,
            advisors: Advisors,
    ) -> Tuple[float, RewardDetails]:
        """
        Compute reward for selecting 'selected_id' given a reference remaining set.

        Parameters
        -----------
        Selected_id: int 
            The chosen item id (should be present in remaining_ref_ids if you use "pre-removal" set).
        remaining_ref_ids: np.ndarray[int64]
            The reference remaining set used to compute per-objective maxima,
            Typically this is the set at decision time (pre-removal)
        objs_int: np.ndarray[int64] shape (N, K)
            Integerized objective matrix for the catalog.
        advisors: Advisors
            Advisors sets and frequency map for the current state

        Returns
        ----------
        (reward, details)
            reward: float in [0, K + beta*K]
            details: RewardDetails(norm_components, φ, reference_max, mode)

        Notes
        --------
        If you prefer "post-removal" normalization, pass the updated remaining set;
        the API is flexible.
        """

        if objs_int.ndim != 2 or objs_int.shape[1] != self.K:
            raise ValueError(f"objs_int must be (N, {self.K})")
        if remaining_ref_ids.size == 0:
            # No normalization context; degenerate reward
            phi = int(advisors.freq.get(int(selected_id), 0))
            return float(self.beta * phi), RewardDetails(
                norm_components=tuple(0.0 for _ in range(self.K)),
                advisor_freq=phi,
                max_vec_int=tuple(0 for _ in range(self.K)),
                mode=self.mode,
            )
        
        # 1. Reference maxima (either step-max or EMA of max)
        ref_max = remaining_max(objs_int, remaining_ref_ids)    #int64[K]
        if self.mode == "ema":
            assert self._ema_tracker is not None
            ref_max = self._ema_tracker.update(ref_max)         #float[K]
            # Normalize with EMA (float vector)
            v = objs_int[int(selected_id), :].astype(np.int64, copy=False)
            denom = np.maximum(ref_max.astype(np.float64, copy=False), 1e-12)
            norm = np.clip(v.astype(np.float64) / denom, 0.0, 1.0)
            max_vec_for_details = tuple(int(x) for x in np.round(ref_max).astype(np.int64).tolist())
        else:
            # "max" mode: normalize by integer max (fast and simple)
            v = objs_int[int(selected_id), :].astype(np.int64, copy=False)
            norm = normalize_item_by_max(v, ref_max, eps=1e-12)
            max_vec_for_details = tuple(int(x) for x in ref_max.tolist())

        # 2. Advisor frequency φ(i)
        phi = int(advisors.freq.get(int(selected_id), 0))
        # 3. Reward
        base = float(np.sum(norm, dtype=np.float64))
        shaped = base + self.beta * float(phi)

        # Bound the reward defensively (floating noise guard)
        K = float(self.K)
        upper = K + self.beta * K
        shaped = float(np.clip(shaped, 0.0, upper))

        details = RewardDetails(
            norm_components=tuple(float(x) for x in norm.tolist()),
            advisor_freq=phi,
            max_vec_int=max_vec_for_details,
            mode=self.mode,
        )
        return shaped, details
    
    # -------- Utilities ----------------

    def theoretical_bounds(self) -> Tuple[float, float]:
        """
        Return (min_reward, max_reward) for current K and beta.
        """
        lo = 0.0
        hi = float(self.K) + self.beta * float(self.K)
        return lo, hi
    
    def reset(self) -> None:
        """
        Reset internal trackers (for EMA mode) at episode start.
        """
        if self._ema_tracker is not None:
            self._ema_tracker = EMAMaxTracker(K=self.K, decay=self._ema_tracker.decay, init_eps=self._ema_tracker.init_eps)