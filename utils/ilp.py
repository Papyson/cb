# utils/ilp.py

"""
Deterministic CP-SAT wrapper for single-objective 0/1 knapsack

we solve: 
    maximize sum_i V[i] * x[i] (+ deterministic tie-break term)
    subject to sum_i C[i] * x[i] <= B
                x[i] in {0,1}

Design goals
---------------
Deterministic across runs (fixed seed, single thread, stable modeling).
Small integer coefficients (see items.integerize) for solver speed.
Optional warm start ("hint") to accelerate repeated solves.
Built-in lexicographic tie-break without harming optimality. 

Tie-break strategy
-------------------
Let M = (sum_i |v[i]|) + 1 (strictly larger than any possible tie-break sum)
Define secondary weights T[i] = (-item_id) to prefer **lower IDs**.
Objectives becomes: maximize M * sum_i V[i] + sum_i T[i] x[i]
This guarantees the primary sum dominates fully; among equal-value solutions,
the solver chooses the set with smallest sum of IDs (stable).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ortools.sat.python import cp_model

@dataclass(frozen=True)
class KnapsackResult:
    """ Solution container for a single-objective knapsack."""
    chosen_ids: List[int]
    objective_value: int            # primary objective sum (Σ V[i] x[i])
    total_cost: int                 # Σ C[i] x[i]
    solver_status: str
    wall_time_ms: float

def solve_knapsack_single_objective(
        values: np.ndarray,
        costs: np.ndarray,
        budget: int,
        *,
        items_ids: Optional[Sequence[int]] = None,
        time_limit_ms: int = 50,
        seed: int = 0,
        warm_start_ids: Optional[Iterable[int]] = None,
) -> KnapsackResult:
    """
    Deterministic 0/1 knapsack via CP-SAT with lexicopgraphic tie-break.

    Parameters
    ------------
    values: (N,) int64
        Non-negative integer objective coefficients to maximize.
    costs: (N,) int64
        Positive integer costs.
    budget: int
        Capacity (non-negative).
    item_ids: optional sequence of length N
        Stable ids corresponding to each row; default = [0..N-1].
    time_limit_ms: int
        Wall-clock limit for the solver (kept small for per-step recomputes).
    seed: int
        Random seed for CP-SAT; used only for internal decisions (we also make 
        modeling deterministic via tie-break).
    warm_start_ids: optional iterable of ints
        If provided, we hint x[i]=1 for these ids (feasible or not; CP-SAT will fix).

    Returns
    ---------
    KnapsackResult

    Notes
    ---------
    Single-threaded for determinism.
    If multiple solutions have identical ΣV, the tie-break selects the one with 
    the smallest sum of item IDs (lower IDs preferred).
    """
    # ----------- Validate Inputs ------------------------
    if values.ndim != 1 or costs.ndim != 1:
        raise ValueError("values and costs must be 1D arrays")
    if values.shape[0] != costs.shape[0]:
        raise ValueError("values and costs must have the same length")
    N = values.shape[0]
    if items_ids is None:
        items_ids = list(range(N))
    if len(items_ids) != N:
        raise ValueError("items_ids length must values/costs length")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    
    V = values.astype(np.int64, copy=False)
    C = costs.astype(np.int64, copy=False)
    ids = np.asarray(items_ids, dtype=np.int64)

    # Precompute constants for objective scaling (tie-break safety)
    M = int(np.abs(V).sum()) + 1

    # ---------- Build Model (deterministic) ------------
    mdl = cp_model.CpModel()

    # Vars: x_i ∈ {0,1}
    x = []
    for i in range(N):
        xi = mdl.NewBoolVar(f"x_{int(ids[i])}")
        x.append(xi)

    # Capacity constraint
    mdl.Add(sum(int(C[i]) * x[i] for i in range(N)) <= int(budget))

    # Objective with tie-break
    # Primary: Σ V[i] x[i] ; Secondary: prefer smaller IDs
    # Use *maximization* with two-scale trick: M*primary + Σ(-id[i])*x[i]
    primary = sum(int(V[i]) * x[i] for i in range(N))
    secondary = sum(int(-ids[i]) * x[i] for i in range(N))
    mdl.Maximize(M * primary + secondary)

    # Optional warm start (hints)
    if warm_start_ids is not None:
        warm = set(int(i) for i in warm_start_ids)
        for i in range(N):
            mdl.AddHint(x[i], 1 if int(ids[i]) in warm else 0)
    
    # ----------- Solver Params -----------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.001, time_limit_ms / 1000.0)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = 1            # determinism
    solver.parameters.log_search_progress = False

    # Solve
    status = solver.Solve(mdl)

    # Extract Solution
    chosen: List[int] = []
    obj_sum = 0
    cost_sum = 0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            if solver.BooleanValue(x[i]):
                chosen.append(int(ids[i]))
                obj_sum += int(V[i])
                cost_sum += int(C[i])

        # Deterministic order for output
        chosen.sort()

    status_str = _status_to_str(status)
    return KnapsackResult(
        chosen_ids=chosen,
        objective_value=obj_sum,
        total_cost=cost_sum,
        solver_status=status_str,
        wall_time_ms=solver.WallTime() * 1000.0,
    )

def _status_to_str(status: int) -> str:
    if status == cp_model.OPTIMAL:
        return "OPTIMAL"
    if status == cp_model.FEASIBILE:
        return "FEASIBILE"
    if status == cp_model.INFEASIBILE:
        return "INFEASIBILE"
    if status == cp_model.MODEL_INVALID:
        return "MODEL_INVALID"
    if status == cp_model.UNKNOWN:
        return "UNKNOWN"
    return f"STATUS_{status}"
