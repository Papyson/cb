# pareto.py
"""
Pareto frontier approximation (all objectives maximize) and distance metrics.

Approach (approximat candidate generation)
-----------
-Weight-scan the simplex: for a set of weight vectors λ ∈ Δ^{K-1},
    solve a weighted single-objective 0/1 knapsack:
    maximize sum_i (sum_k λ_k * v_k(i)) * x_i
    s.t. sum_i cost(i) * x_i ≤ budget, x_i ∈ {0,1}
-Collect solution, map each to a K-dim objective vector (sum of objectives).
-Filter to non-dominated points (all-maximize).
-Normalize agent and frontier points per objective (min-max union).
-Report Chebyshev (∞-norm) distance from agent to the closest frontier point.

Determinism
-------------
Weight vectors are generated with a DeterministicRNG child stream.
Each weight's knapsack uses a seed derived from (base_seed, "w", w_idx).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from utils.ilp import solve_knapsack_single_objective
from utils.rng import DeterministicRNG, hash_uint64


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
) -> ParetoResult:
    """
    Compute a pareto frontier approximation and distance for the agent's final point.

    Parameters
    ------------
    objs_int: np.int64[N,K]             integerized objective matrix
    costs_int: np.int64[N]              integerized costs
    budget_int: int                     capacity used for frontier solves (initial episode budget)
    agent_scores_int: np.int64[K]       final cumulative objective totals of the agent.
    K: int                              number of objectives (all maximize)
    num_weights: int                    number of weight vectors to scan (>= K+1)
    time_limt_ms: int                   per-solve time limit
    base_seed: int                      seed for deterministic weight generation and solver seeds

    Returns
    -----------
    ParetoResult
    """
    if objs_int.ndim != 2 or objs_int.shape[1] != K:
        raise ValueError(f"objs_int must be (N, {K})")
    if costs_int.ndim != 1 or costs_int.shape[0] != objs_int.shape[0]:
        raise ValueError("costs_int must be (N,), same N as objs_int")
    if num_weights < K + 1:
        num_weights = K + 1 # ensure corners + center at least

    N = objs_int.shape[0]
    # 1. Build weight set deterministically
    rng = DeterministicRNG(seed=int(base_seed), stream="pareto")
    W = _make_weight_vectors(K=K, num_weights=num_weights, rng=rng)

    # 2. Solve weighted knapsacks
    scanned_obj = np.zeros((len(W), K), dtype=np.int64)
    for w_idx, w in enumerate(W):
        # integer-valued objective coefficients per item: round(weighted sum)
        # (λ ≥ 0, obj_int ≥ 0) ⇒ values ≥ 0
        values_float = objs_int.astype(np.float64) @ w # (N,)
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
            scanned_obj[w_idx, :] = np.zeros(K, dtype=np.int64)
        else:
            sub = objs_int[res.chosen_ids, :]   # (chosen, K)
            scanned_obj[w_idx, :] = sub.sum(axis=0).astype(np.int64, copy=False)
        
    # 3. Filter to non-dominated points (all-maximize)
    keep_idx = _pareto_frontier_max(scanned_obj)
    frontier = scanned_obj[keep_idx, :]

    # 4. Normalize agent & frontier (min-max over union to [0,1])
    agent = agent_scores_int.astype(np.float64, copy=False)
    union = np.vstack([frontier.astype(np.float64, copy=False), agent.reshape(1,K)])
    mins = union.min(axis=0)
    maxs = union.max(axis=0)
    denom = np.maximum(maxs - mins, 1e-12)
    frontier_norm = (frontier - mins) / denom
    agent_norm = (agent - mins) / denom

    # 5. Chebyshev (∞-norm) distance to the closest frontier point
    diffs = np.abs(frontier_norm - agent_norm.reshape(1, K))    # (F, K)
    cheb = diffs.max(axis=1)
    best_idx_local = int(np.argmin(cheb))
    best_dist = float(cheb[best_idx_local])

    return ParetoResult(
        frontier_obj_int=frontier.astype(np.int64, copy=False),
        frontier_indices=[int(i) for i in keep_idx],
        agent_scores_int=tuple(int(x) for x in agent_scores_int.tolist()),
        distance_chebyshev=best_dist,
        best_frontier_idx=best_idx_local,
        norm_mins=tuple(float(x) for x in mins.tolist()),
        norm_maxs=tuple(float(x) for x in maxs.tolist()),
    )

# ----------------------------
# Helpers
# ----------------------------

def _make_weight_vectors(K: int, num_weights: int, rng: DeterministicRNG) -> List[np.ndarray]:
    """
    Build a deterministic set weight vectors on the simplex Δ^{K-1}.
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
            vecs.append(s.astype(np.float64, copy=False))

    # Small numeric cleanup + assert sum to 1
    for v in vecs:
        v[v < 0.0] = 0.0
        s = float(v.sum())
        if s<= 0.0:
            v[:] = 0.0
            v[0] = 1.0
        else:
            v /= s
    return vecs

def _pareto_frontier_max(points: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated points under 'maximize-all' convention.

    A point a dominates b iff:
        for all k: a_k ≥ b_k  and  exists k: a_k > b_k
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