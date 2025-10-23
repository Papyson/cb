# recommend.py
"""
RecommendationManager: per-objective knapsack solves -> advisor sets and frequencies.

Design goals
---------------
Deterministic per state: cache key derived from (remaining set, budget bucket).
Stable tie-breaks via utils.ilp (lexicographic on item ids).
Lightweight LRU cache to avoid re-solving identical states.
Clean separation: this module is stateless w.r.t. catalog contents; you pass arrays in.

Inputs
-------------
-remainings_ids: np.int64[NR] sorted ascending
-budget_rem_int: int (integerized)
-costs_int: np.int64[N]
-objs_int: np.int64[N, K]
-config knobs: K, time_limit_ms, cache_size, budget_bucket
-seeds: base_seed (derive per-objective seeds deterministically)

Outputs
-------------
-Advisors: rsets[K] + freq map item_id -> {0..K}
-Stats: cache_hit(bool), solver_wall_ms(float), per_obj_status(list[str])
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from citybuilder_env.state import Advisors
from citybuilder_env.utils.ilp import solve_knapsack_single_objective
from citybuilder_env.utils.rng import hash_uint64

# ------------------------------
# Small LRU cache
# ------------------------------

class _LRUCache:
    """Tiny OrderedDict-based LRU cache: key -> value. Evicts oldest on overflow."""
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._od: OrderedDict = OrderedDict()

    def get(self, key):
        if self.capacity == 0:
            return None
        v = self._od.get(key)
        if v is None:
            return None
        # Move to end (most recently used)
        self._od.move_to_end(key)
        return v
    
    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self._od:
            self._od.move_to_end(key)
            self._od[key] = value
        else:
            self._od[key] = value
            if len(self._od) > self.capacity:
                self._od.popitem(last=False)    # evict LRU

# -------------------
# Public Manager
# -------------------

@dataclass
class RecommendationStats:
    cache_hit: bool
    total_wall_ms: float
    per_objective_status: List[str]

class RecommendationManager:
    """
    Compute advisor sets using per-objective knapsack solves with caching.

    Parameters
    -------------
    K: int
        Number of objectives (all are "maximize").
    time_limit_ms: int
        Wall time per solve.
    cache_size: int
        Max distinct (remaining, budget_bucket) states to cache.
    budget_bucket: int
        Bucket width for budget integer bucketing in the cache key.
    base_seed: int
        Seed root for deterministic solver seeds (per objective, per state).

    Note: This object is light; feel free to construct once per catalog.
    """

    def __init__(self, K: int, time_limit_ms: int, cache_size: int, budget_bucket: int, base_seed: int):
        if K <= 0:
            raise ValueError("K must be positive")
        if time_limit_ms <= 0: 
            raise ValueError("time_limit_ms must be positive")
        if budget_bucket <= 0:
            raise ValueError("budget_bucket must be positive")
        
        self.K = int(K)
        self.time_limit_ms = int(time_limit_ms)
        self.cache = _LRUCache(int(cache_size))
        self.budget_bucket = int(budget_bucket)
        self.base_seed = int(base_seed)

    # ---------- Keying ---------------
    def _make_cache_key(self, remaining_ids: np.ndarray, budget_rem_int: int, N_total: int) -> bytes:
        """
        Build a compact, deterministic cache key:
        key = packbits(availability_mask[N_total]) || bucket(budget)
        """

        # Fixed-length boolean mask for the whole catalog
        avail = np.zeros(N_total, dtype=np.uint8)
        # remaining_ids is sorted; set ones
        avail[remaining_ids] = 1
        bitmap = np.packbits(avail, bitorder="little").tobytes()
        bucket = budget_rem_int // self.budget_bucket
        # Compose bytes: 8 bytes of bucket (little-endian) + bitmap
        return int(bucket).to_bytes(8, "little", signed=False) + bitmap
    
    def _derive_solver_seed(self, obj_index: int, cache_key: bytes) -> int:
        """
        Deterministically derive a per-objective solver seed from the based_seed and the cache key.
        """
        u = hash_uint64(self.base_seed, "recommend", obj_index, cache_key)
        return int(u)
    
    # ---------- Main API ----------------
    def compute_advisors(
            self,
            remaining_ids: np.ndarray,
            budget_rem_int: int,
            costs_int: np.ndarray,
            objs_int: np.ndarray,
    ) -> Tuple[Advisors, RecommendationStats]:
        """
        Compute/return advisor sets for each objective and advisor-frequency map.

        Parameters
        ------------
        remaining_ids: np.ndarray[int64], sorted
        budget_rem_int: int
        costs_int: np.ndarray[int64] of shape (N,)
        objs_int: np.ndarray[int64] of shape (N, K)

        Returns
        ----------
        Advisors, RecommendationStats
        """

        if remaining_ids.size == 0 or budget_rem_int <= 0:
            # Trivial: no recommendations possible
            empty = Advisors.empty(self.K)
            return empty, RecommendationStats(cache_hit=False, total_wall_ms=0.0, per_objective_status=[])
        
        N_total = int(costs_int.shape[0])
        key = self._make_cache_key(remaining_ids, int(budget_rem_int), N_total)

        cached = self.cache.get(key)
        if cached is not None:
            advisors, stats = cached
            # Re-wrap stats with cache_hit=True and zero extra time (diagnostic)
            return advisors, RecommendationStats(cache_hit=True, total_wall_ms=stats.total_wall_ms, per_objective_status=stats.per_objective_status)
        
        # Not in cache -> solve K single-objective knapsacks on the remaining subset
        total_ms = 0.0
        per_status: List[str] = []
        rsets: List[List[int]] = [[] for _ in range(self.K)]

        # Slice arrays to remaining candidates once
        rem_ids = remaining_ids
        rem_costs = costs_int[rem_ids]          # shape(NR,)
        # For eeach objective k, take column and slice
        for k in range(self.K):
            values_k = objs_int[rem_ids, k]     # shape(NR,)
            seed_k = self._derive_solver_seed(k, key)
            res = solve_knapsack_single_objective(
                values=values_k,
                costs=rem_costs,
                budget=int(budget_rem_int),
                items_ids=rem_ids.tolist(),
                time_limit_ms=self.time_limit_ms,
                seed=seed_k,
                warm_start_ids=None,            # can add greedy hint latter if desired
            )
            rsets[k] = res.chosen_ids # already sorted by solver wrapper
            total_ms += float(res.wall_time_ms)
            per_status.append(res.solver_status)

        # Build advisor frequency φ(i)
        freq: Dict[int, int] = {}
        for k in range(self.K):
            for iid in rsets[k]:
                freq[iid] = freq.get(iid, 0) + 1
        
        advisors = Advisors(rsets=rsets, freq=freq)
        stats = RecommendationStats(cache_hit=False, total_wall_ms=total_ms, per_objective_status=per_status)
        self.cache.put(key, (advisors, stats))
        return advisors, stats