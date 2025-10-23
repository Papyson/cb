# utils/normalization.py

"""
Normalization helpers for per-step rewards. 

We keep this intentionally simple and numerically safe, optimized for small N (<= ~100):
    "Max over remaining" normalization: \tilde v_k(i) = v_k(i) / max_{j in remaining} v_k(j)
    Optional EMA of the remaining maxima if you need smoother rewards.

All objectives are treated as "maximize"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

def remaining_max(objs_int: np.ndarray, remaining_ids: np.ndarray) -> np.ndarray:
    """
    Compute per-objective maxima over the remaining set.

    Parameters
    ------------
    objs_int: np.int64[N, K]
    remaining_ids: np.int64[NR]

    Returns
    ---------
    max_vec: np.int64[K]
    """

    if remaining_ids.size == 0:
        raise ValueError("remaining_ids is empty")
    sub = objs_int[remaining_ids, :]
    # For K small, np.max over axis=0 is efficient
    return sub.max(axis=0)

def normalize_item_by_max(
        item_obj_int: np.ndarray,
        max_vec_int: np.ndarray,
        eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize the selected item's objective vector by per-objective maxima.

     \tilde v_k(i) = v_k(i) / max_k   with guards for zeros.

    Parameters
    ----------
    item_obj_int : np.int64[K]
    max_vec_int  : np.int64[K]
    eps          : float guard to avoid division by zero when max=0

    Returns
    ---------
    norm: np.float64[K] in [0, 1]
    """
    if item_obj_int.shape != max_vec_int.shape:
        raise ValueError("item_obj_int and max_vec_int must have same shape")
    max_f = max_vec_int.astype(np.float64, copy=False)
    v_f = item_obj_int.astype(np.float64, copy=False)
    denom = np.maximum(max_f, eps)
    out = v_f / denom
    # Numeric noise clip
    return np.clip(out, 0.0, 1.0)

# ----------------------------
# Optional EMA smoothing
# ----------------------------

@dataclass
class EMAMaxTracker:
    """
    Exponential Moving Average tracker for per-objective maxima.

    Use when per-step remaining maxima are too jumpy (rare at small N).
    In practice, start with plain max; switch to EMA only if needed. 

    Fields
    --------
    K: number of objectives
    decay: EMA decay in (0,1); closer to 1.0 = slower updates
    init_eps: small floor to avoid zero denominators

    Methods
    ---------
    update(ref_max_int) -> np.ndarray[float64]:
        Update the EMA from an integer max vector and return the current EMA.
    normalize(item_obj_int) -> np.ndarray[float64 K]:
        Normalize by the current EMA vector (call update() first each step).
    """
    K: int
    decay: float = 0.9
    init_eps: float = 1e-12

    def __post_init__(self):
        if not (0.0 < self.decay < 1.0):
            raise ValueError("decay must be (0,1)")
        self._ema = np.full(self.K, self.init_eps, dtype=np.float64)

    def update(self, ref_max_init: np.ndarray) -> np.ndarray:
        if ref_max_init.shape != (self.K,):
            raise ValueError(f"ref_max_init must be shape ({self.K},)")
        ref = ref_max_init.astype(np.float64, copy=False)
        self._ema = self.decay * self._ema + (1.0 - self.decay) * ref
        # Keep a small floor
        self._ema = np.maximum(self._ema, self.init_eps)
        return self._ema.copy()
    
    def normalize(self, item_obj_int: np.ndarray) -> np.ndarray:
        if item_obj_int.shape != (self.K,):
            raise ValueError(f"item_obj_int must be shape ({self.K},)")
        v = item_obj_int.astype(np.float64, copy=False)
        denom = np.maximum(self._ema, self.init_eps)
        return np.clip(v / denom, 0.0, 1.0)