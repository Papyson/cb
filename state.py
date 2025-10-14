# state.py
"""
Typed data structures for the CityBuilder RL Enviornment.

This module centralizes the "shape" of everything the environment and managers
exchange: Items, Catalogs, Advisors, EpisodeConfig, EpisodeState.

Design Choices
--------------
Strong typing + dataclasses for clarity and IDE help.
Immutable (frozen) records for static entities (Item, Catalog metadata).
Mutable EpisodeState for step-wise evolution.
Integer arithmetic (costs/objectives) for ILP stability; float views for logs.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------
# Core static entities 
# --------------------------

@dataclass(frozen=True)
class Item:
    """
    One catalog item with integrized cost and objectives.

    Fields
    -------
    id: Stable integer ID (0...N-1 or any stable assignment)
    cost_int: Integerized cost (scaled by Catalog.int_scale).
    obj_int: np.ndarray of shape (K,), dtype=int64. All objectives are treated as "maximize" throughout the system.
    meta: Optional arbitrary metadata (e.g., name, category).
    """

    id: int
    cost_int: int
    obj_int: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Lightweight type/shape assertions for early failure.
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("Item.id must be a non-negative int")
        if not isinstance(self.cost_int, (int, np.integer)) or self.cost_int <= 0:
            raise ValueError("Item.cost_int must be a positive integer")
        if not isinstance(self.obj_int, np.ndarray):
            raise TypeError("Item.obj_int must be a numpy.ndarray")
        if self.obj_int.ndim != 1:
            raise ValueError("Item.obj_int must be 1D array")
        if self.obj_int.dtype.kind not in ("i", "u"):
            raise TypeError("Item.obj_int must have integer dtype")
        
    def float_view(self, int_scale: int) -> Tuple[float, np.ndarray]:
        """
        Return (cost_float, obj_float[K] using the provided scale.
        """
        cost_f = float(self.cost_int) / float(int_scale)
        obj_f = (self.obj_int.astype(np.float64, copy=False) / float(int_scale)).copy()
        return cost_f, obj_f
    
@dataclass(frozen=True)
class Catalog:
    """
    A complete snapshot of the generated item catalog.

    Fields
    ------
    items: List[Item], stable order.
    int_scale: The integerization factor used for costs/objectives.
    seed: The catalog-level seed used for generation (for replay).
    name: Human-readable label (e.g., "catalog_0001").
    K: Number of objectives per item.
    """
    items: List[Item]
    int_scale: int
    seed: int
    name: str
    K: int

    def __post_init__(self):
        if self.int_scale <= 0:
            raise ValueError("Catalog.int_scale must be positive")
        if self.K <= 0:
            raise ValueError("Catalog.K must be positive")
        # Validate items
        seen = set()
        for it in self.items:
            if it.id in seen:
                raise ValueError(f"Duplicate item id detected: {it.id}")
            seen.add(it.id)
            if it.obj_int.shape[0] != self.K:
                raise ValueError(f"Item {it.id} has obj_int length {it.obj_int.shape[0]} != Catalog.K {self.K}")

    @property
    def N(self) -> int:
        """Number of items in the catalog. O(1)."""
        return len(self.items)

    def costs_array(self) -> np.ndarray:
        """Return costs as int64 array of shape (N,). O(N)."""
        return np.fromiter((it.cost_int for it in self.items), dtype=np.int64, count=self.N)

    def objs_matrix(self) -> np.ndarray:
        """Return objectives as int64 matrix of shape (N, K). O(N*K)."""
        mat = np.empty((self.N, self.K), dtype=np.int64)
        for i, it in enumerate(self.items):
            mat[i, :] = it.obj_int
        return mat

    def ids_array(self) -> np.ndarray:
        """Return item ids as int64 array (N,). O(N)."""
        return np.fromiter((it.id for it in self.items), dtype=np.int64, count=self.N)
        
# ------------------------------
# Advisors (recommendation view)
# ------------------------------

@dataclass(frozen=True)
class Advisors:
    """
    Advisor recommendations derived from per-objective knapsack solves.

    Fields
    ---------
    rsets: list of K lists, where rsets[k] contains item_ids recommended
            by the single-objective solve on the objective k.
    freq: dict item_id -> integer in [0..k], counting how many per-obj
            solutions include the item.
    """
    rsets: List[List[int]]
    freq: Dict[int, int]

    def __post_init__(self):
        # Validate length consistency and freq bounds
        if len(self.rsets) == 0:
            raise ValueError("Advisors.rsets cannot be empty")
        K = len(self.rsets)
        # Flatten & check freq bounds if provided (non-strict in case of empty)
        if self.freq:
            for item_id, f in self.freq.items():
                if f < 0 or f > K:
                    raise ValueError(f"Advisor frequency out of bounds for item {item_id}: {f} not in [0..{K}]")
    
    @classmethod
    def empty (cls, K: int) -> "Advisors":
        """Utility for initialization before first solver call."""
        return Advisors(rsets=[[] for _ in range(K)], freq={})
    
    def for_index(self, k: int) -> List[int]:
        """Return advisor set for objective index k."""
        if not (0 <= k < len(self.rsets)):
            raise IndexError("Advisor index out range")
        return self.rsets[k]
    
# -----------------------
# Episode Configuration
# -----------------------

@dataclass(frozen=True)
class EpisodeConfig:
    """
    Frozen configuration for one class of episodes.

    Fields
    --------
    K: number of objectives.
    target_picks: target number of selections in an episode(soft cap)
    budget_multiplier: scales expected total cost to derive the episode budget.
    int_scale: intergerization factor (must match catalog).
    beta: advisor-frequency shaping coefficient in reward.
    max_steps: hard on steps (safety).
    solver_time_limits_ms: per-solve time limit for CP-SAT (advisors, frontier).
    cache_size: max LRU entries for advisor caching. 
    budget_bucket: integer bucketing for advisor cache key on budget.
    num_frontier_weights: number of weight vectors to scan for Pareto frontier.
    """
    K: int
    target_picks: int
    budget_multiplier: float
    int_scale: int
    beta: float
    max_steps: int
    solver_time_limit_ms: int
    cache_size: int
    budget_bucket: int
    num_frontier_weights: int

    def __post_int__(self):
        if self.K <= 0:
            raise ValueError ("EpisodeConfig.K must be postive")
        if self.target_picks <= 0:
            raise ValueError("EpisodeConfig.target_picks must be positive")
        if self.int_scale <= 0:
            raise ValueError("EpisodeConfig.int_scale must be positive")
        if self.max_steps <= 0:
            raise ValueError("EpisodeConfig.max_steps must be postive")
        if self.cache_size < 0:
            raise ValueError("EpisodeConfig.cache_size must be non-negative")
        if self.budget_bucket <= 0:
            raise ValueError("EpisodeConfig.budget_bucket must be positive")
        if self.num_frontier_weights <= 0:
            raise ValueError("EpisodeConfig.num_frontier_weights must be positive")
        
    def derive_budget_int(self, expected_cost_float: float) -> int:
        """
        Compute integerized episode budget from expected per-item cost.

        Formula:
            budget = round(int_scale * budget_multiplier * expected_cost * target_picks)
        """
        if expected_cost_float <= 0.0:
            raise ValueError("expected_cost_float must be positive")
        budget = round(self.int_scale * self.budget_multiplier * expected_cost_float * self.target_picks)
        # Ensure at least one cheapest item can be bought in typical catalogs;
        # caller should still validate feasibility at runtime.
        return max(int(budget), 1)
    
# -------------------------
# Episode state (mutable)
# -------------------------

@dataclass
class EpisodeState:
    """
    Mutable, per-episode state tracked by the environment.

    Fields
    ---------
    catalog: the immutable catalog reference (ids/costs/objs).
    budget_rem_int: remaining budget in integer units.
    scores_cum_int: np.int64[K] cumulative objective totals.
    remaining_ids: np.int64[N_rem] array of selectable item ids (sorted for determinism).
    selected_ids: List[int] of chosen items in order.
    step_idx: current step index (0-based).
    advisors: Advisors object (updated after each selection).
    """

    catalog: Catalog
    budget_rem_int: int
    scores_cum_int: np.ndarray
    remaining_ids: np.ndarray
    selected_ids: List[int] = field(default_factory=list)
    step_idx: int = 0
    advisors: Advisors = field(default_factory=lambda: Advisors.empty(1)) #replaced at reset

    def __post_init__(self):
        if self.scores_cum_int.shape != (self.catalog.K,):
            raise ValueError("scores_cum_int must have shape (K,)")
        if self.scores_cum_int.dtype.kind not in ("i","u"):
            raise ValueError("scores_cum_int must have integer dtype")
        if self.remaining_ids.dtype.kind not in ("i", "u"):
            raise TypeError("remianing_ids must have integer dtype")
        # Ensure determinism: keep remaining_ids sorted at all times.
        self.remaining_ids = np.sort(self.remaining_ids.astype(np.int64, copy=True))

    @property
    def done(self) -> bool:
        """Convenience flag; the Env enforces actual teermination conditions."""
        return (self.steps_idx >= 10**9) or (self.remaining_ids.size == 0) or (self.budget_rem_int <= 0)
    
    def is_feasible(self, item_id: int, costs: np.ndarray) -> bool:
        """
        Check feasibility of selecting 'item_id' under current budget & availbility.
        """

        # Binary search in sorted remaining_ids
        idx = np.searchsorted(self.remaining_ids, item_id)
        in_remaining = idx < self.remaining_ids.size and self.remaining_ids[idx] == item_id
        if not in_remaining:
            return False
        return int(costs[item_id]) <= self.budget_rem_int
    
    def mark_selected(self, item_id: int) -> None:
        """
        Remove item_id from remaining_ids and append to selected_ids.
        """
        pos = np.searchsorted(self.remaining_ids, item_id)
        if pos >= self.remaining_ids.size or self.remaining_ids[pos] != item_id:
            raise KeyError(f"item_id {item_id} not in remaining_ids")
        self.remaining_ids = np.delete(self.remaining_ids, pos)
        self.selected_ids.append(item_id)

    def snapshot(self) -> Dict[str, Any]:
        """Produce a lightweight dict for logging/telemetry at this step."""
        return {
            "step_idx": self.step_idx,
            "budget_rem_int": int(self.budget_rem_int),
            "scores_cum_int": self.scores_cum_int.copy(),
            "n_remaining": int(self.remaining_ids.size),
            "n_selected": len(self.selected_ids),
        }


