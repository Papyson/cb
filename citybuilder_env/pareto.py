# pareto.py
"""
Pareto frontier approximation (all objectives maximize) and distance metrics.

Approach (approximate candidate generation)
-----------
- Weight-scan the simplex: for a set of weight vectors λ ∈ Δ^{K-1},
  solve a weighted single-objective 0/1 knapsack:
      maximize sum_i (sum_k λ_k * v_k(i)) * x_i
      s.t. sum_i cost(i) * x_i ≤ budget, x_i ∈ {0,1}
- Collect solution, map each to a K-dim objective vector (sum of objectives).
- Filter to non-dominated points (all-maximize).
- Normalize agent and frontier points per objective (min-max over the union).
- Report Chebyshev (∞-norm) distance from agent to the closest frontier point.

Determinism
-------------
Weight vectors are generated with a DeterministicRNG child stream.
Each weight's knapsack uses a seed derived from (base_seed, "w", w_idx).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .utils.ilp import solve_knapsack_single_objective
from .utils.rng import DeterministicRNG, hash_uint64


@dataclass(frozen=True)
class ParetoResult:
    """
    Results for an episode-end Pareto evaluation.
    """
    frontier_obj_int: np.ndarray            # shape (F, K), integer sums per frontier point
    frontier_indices: List[int]             # indices in the scanned set that survived
    agent_scores_int: Tuple[int, ...]       # K-tuple
    distance_chebyshev: float               # on [0,1] scale after min-max normalization
    best_frontier_idx: int                  # index into frontier_obj_int
    norm_mins: Tuple[float, ...]            # per-objective minima used for normalization
    norm_maxs: Tuple[float, ...]            # per-objective maxima used for normalization

    # --- New optional payloads (populated only if requested) ---
    all_distances: Optional[np.ndarray] = None      # (F,) distances to every frontier point
    knn_indices: Optional[np.ndarray] = None        # (k,) indices into frontier_obj_int
    knn_distances: Optional[np.ndarray] = None      # (k,) distances for those indices


def _make_weight_vectors(K: int, num_weights: int, rng: DeterministicRNG) -> List[np.ndarray]:
    """
    Build a deterministic set of weight vectors on the simplex Δ^{K-1}.
    Includes corners and center; fills the rest via Dirichlet(1).
    """
    vecs: List[np.ndarray] = []

    # Corners
    for k in range(K):
        e = np.zeros(K, dtype=np.float64)
        e[k] = 1.0
        vecs.append(e)

    # Center
    vecs.append(np.full(K, 1.0 / float(K), dtype=np.float64))

    # Fill the rest via Dirichlet (α=1)
    need = max(0, num_weights - len(vecs))
    if need > 0:
        g = rng.child("weights").gen
        samples = g.dirichlet(alpha=np.ones(K, dtype=np.float64), size=need)
        for s in samples:
            v = s.astype(np.float64, copy=False)
            # numeric cleanup + renormalize
            v[v < 0.0] = 0.0
            ssum = float(v.sum())
            vecs.append(v / ssum if ssum > 0.0 else np.eye(1, K, 0, dtype=np.float64).ravel())

    # Ensure each sums to 1 exactly
    for v in vecs:
        s = float(v.sum())
        if s <= 0.0:
            v[:] = 0.0
            v[0] = 1.0
        else:
            v /= s
    return vecs


def _pareto_frontier_max(points: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated points under 'maximize-all' convention.

    A point a dominates b iff: for all k: a_k ≥ b_k  and  exists k: a_k > b_k
    """
    M, K = points.shape
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        if not keep[i]:
            continue
        p = points[i]
        # Any j that strictly dominates i -> discard i
        for j in range(M):
            if i == j or not keep[j]:
                continue
            q = points[j]
            if np.all(q >= p) and np.any(q > p):
                keep[i] = False
                break
    return np.flatnonzero(keep)


def _normalize(points: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Max-normalize to [0,1], safe for zero span."""
    span = np.maximum(maxs - mins, 1e-12)
    return (points - mins) / span


def normalized_distances(agent_norm: np.ndarray, frontier_norm: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev (L-infinity) distances from a normalized agent vector (K,)
    to each normalized frontier point (F,K). Returns shape (F,).
    """
    diffs = np.abs(frontier_norm - agent_norm[None, :])  # (F,K)
    return diffs.max(axis=1)  # (F,)


def compute_pareto_frontier_and_distance(
    objs_int: np.ndarray,
    costs_int: np.ndarray,
    budget_int: int,
    agent_scores_int: np.ndarray,
    *,
    K: int,
    num_weights: int,
    time_limit_ms: int,
    base_seed: int,
    # --- New optional knobs; defaults preserve old behavior ---
    return_all_distances: bool = False,
    return_knn: bool = False,
    k: int = 5,
) -> ParetoResult:
    """
    Compute a Pareto frontier approximation and proximity metrics for the agent's final point.

    Parameters
    ----------
    objs_int        : (N,K) integerized objective matrix
    costs_int       : (N,)  integerized costs
    budget_int      : int   capacity used for frontier solves (initial episode budget)
    agent_scores_int: (K,)  final cumulative objective totals of the agent
    K               : int   number of objectives (all maximize)
    num_weights     : int   number of weight vectors to scan (>= K+1)
    time_limit_ms   : int   per-solve time limit
    base_seed       : int   seed for deterministic weight generation and solver seeds

    Optional (additive; set flags to True to receive extra outputs)
    ----------------------------------------------------------------
    return_all_distances : include Chebyshev distances to each frontier point
    return_knn           : include k nearest frontier indices and distances
    k                    : number of nearest frontier points to return (>=1)

    Returns
    -------
    ParetoResult  (backward compatible: core fields always populated)
    """
    if objs_int.ndim != 2 or objs_int.shape[1] != K:
        raise ValueError(f"objs_int must be (N, {K})")
    if costs_int.ndim != 1 or costs_int.shape[0] != objs_int.shape[0]:
        raise ValueError("costs_int must be (N,), same N as objs_int")
    if num_weights < K + 1:
        num_weights = K + 1  # ensure corners + center at least

    N = objs_int.shape[0]

    # 1) Build weight set deterministically
    rng = DeterministicRNG(seed=int(base_seed), stream="pareto")
    W = _make_weight_vectors(K=K, num_weights=num_weights, rng=rng)

    # 2) Solve weighted knapsacks
    scanned_obj = np.zeros((len(W), K), dtype=np.int64)
    for w_idx, w in enumerate(W):
        # integer-valued objective coefficients per item: round(weighted sum)
        # (λ ≥ 0, obj_int ≥ 0) ⇒ values ≥ 0
        values_float = objs_int.astype(np.float64) @ w  # (N,)
        values_int = np.rint(values_float).astype(np.int64)
        seed = int(hash_uint64(base_seed, "w", w_idx))
        res = solve_knapsack_single_objective(
            values=values_int,
            costs=costs_int,
            budget=int(budget_int),
            items_ids=list(range(N)),
            time_limit_ms=int(time_limit_ms),
            seed=seed,
            warm_start_ids=None,
        )
        if len(res.chosen_ids) == 0:
            scanned_obj[w_idx, :] = 0
        else:
            sub = objs_int[res.chosen_ids, :]   # (chosen, K)
            scanned_obj[w_idx, :] = sub.sum(axis=0).astype(np.int64, copy=False)

    # 3) Filter to non-dominated points (all-maximize)
    keep_idx = _pareto_frontier_max(scanned_obj)
    frontier = scanned_obj[keep_idx, :]

    # Edge: no frontier (should be rare unless weights degenerate)
    if frontier.size == 0:
        mins = np.zeros(K, dtype=float)
        maxs = np.ones(K, dtype=float)
        return ParetoResult(
            frontier_obj_int=frontier,
            frontier_indices=[int(i) for i in keep_idx],  # empty
            agent_scores_int=tuple(int(x) for x in agent_scores_int.tolist()),
            distance_chebyshev=1.0,
            best_frontier_idx=0,
            norm_mins=tuple(float(x) for x in mins.tolist()),
            norm_maxs=tuple(float(x) for x in maxs.tolist()),
            all_distances=(np.array([]) if return_all_distances else None),
            knn_indices=(np.array([], dtype=int) if return_knn else None),
            knn_distances=(np.array([]) if return_knn else None),
        )

    # 4) Normalize agent & frontier (min-max over union to [0,1])
    #    (keeps exact backward-compat with your original scaling)
    agent = agent_scores_int.astype(np.float64, copy=False)
    union = np.vstack([frontier.astype(np.float64, copy=False), agent.reshape(1, K)])
    mins = union.min(axis=0)
    maxs = union.max(axis=0)
    denom = np.maximum(maxs - mins, 1e-12)
    frontier_norm = (frontier - mins) / denom
    agent_norm = (agent - mins) / denom

    # 5) Chebyshev distances + best
    dists = normalized_distances(agent_norm, frontier_norm)  # (F,)
    best_idx_local = int(np.argmin(dists))
    best_dist = float(dists[best_idx_local])

    # 6) Optional k-NN over frontier points
    knn_idx = None
    knn_d = None
    if return_knn:
        k_eff = max(1, min(int(k), dists.shape[0]))
        order = np.argpartition(dists, kth=k_eff - 1)[:k_eff]  # k smallest (unordered)
        # sort those k by distance for stability
        small_order = order[np.argsort(dists[order])]
        knn_idx = small_order.astype(int)
        knn_d = dists[knn_idx]

    return ParetoResult(
        frontier_obj_int=frontier.astype(np.int64, copy=False),
        frontier_indices=[int(i) for i in keep_idx],
        agent_scores_int=tuple(int(x) for x in agent_scores_int.tolist()),
        distance_chebyshev=best_dist,
        best_frontier_idx=best_idx_local,
        norm_mins=tuple(float(x) for x in mins.tolist()),
        norm_maxs=tuple(float(x) for x in maxs.tolist()),
        all_distances=(dists if return_all_distances else None),
        knn_indices=knn_idx,
        knn_distances=knn_d,
    )
