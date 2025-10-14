# score.py

"""
ScoreManager: integer-accurate accounting of budget and cumulative objectives,
plus a lightweight step ledger for telemetry and replay.

Design goals
-------------
Keep all math in integers (cost/objectives already integerized).
Enforce invariants (budget never negative, shapes, consistent).
Produce compact, step-wise records for analysis and summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

from state import EpisodeState

@dataclass(frozen=True)
class StepRecord:
    """ Immutable per-step record (ints only; floats can be reconstructed with int_scale)."""
    step_idx: int
    item_id: int
    cost_int: int
    delta_obj_int: tuple        #length K
    budget_after_int: int
    scores_after_int: tuple     #length K

class ScoreManager:
    """
    Integer-accurate scorer and ledger.

    Typical usage (inside the Env step loop)
    -------------------------------------------
    >>> info = score_mgr.apply_selection(state, costs_int, objs_int, item_id)
    >>> # state.scores_cum_int and state.budget_rem_int are updated in-place
    >>> # ledger entry is appended internally
    """

    def __init__(self, K: int):
        if K <= 0:
            raise ValueError("K must be positive")
        self.K = int(K)
        self._ledger: List[StepRecord] = []

    # --------- Episode Lifecycle -----------

    def reset(self) -> None:
        """Clear the step ledger at episode start."""
        self._ledger.clear()

    # -------- Core update -----------------
    def apply_selection(
            self,
            state: EpisodeState,
            costs_int: np.ndarray,
            objs_int: np.ndarray,
            item_id: int,
    ) -> Dict[str, int]:
        """
        Apply a valid selection to the episode state and write a ledger entry.

        Parameters
        -------------
        state: EpisodeState (mutated in-inplace)
        costs_int: np.int64[N]
        objs_int: np.int64[N, K]
        item_id: int, must be feasible under current state

        Returns
        ---------
        info: dict with lightweight fields (ints only) for the caller.
        keys: 'cost_int', 'budget_after_int'
        """
        # Basic shape checks (fast)
        if objs_int.ndim != 2 or objs_int.shape[1] != self.K:
            raise ValueError(f"objs_int must be (N, {self.K})")
        if costs_int.ndim != 1 or costs_int.shape[0] != objs_int.shape[0]:
            raise ValueError("costs_int must be (N,), same N as objs_int")
        
        # Grab integers for the selected item
        c = int(costs_int[item_id])
        v = objs_int[item_id, :].astype(np.int64, copy=False)

        # Update budget (integer; must not go negative)
        new_budget = int(state.budget_rem_int) - c
        if new_budget <  0:
            raise RuntimeError(
                f"Budget underflow after selecting item {item_id}:"
                f"{state.budget_rem_int} - {c} < 0"
            )
        state.budget_rem_int = new_budget

        # Update cummulative objectives (integer vector add)
        state.scores_cum_int = (state.scores_cum_int + v).astype(np.int64, copy=False)

        # Append step record (immutable)
        rec = StepRecord(
            step_idx=int(state.step_idx),
            item_id=int(item_id),
            cost_int=c,
            delta_obj_int=tuple(int(x) for x in v.tolist()),
            budget_after_int=new_budget,
            scores_after_int=tuple(int(x) for x in state.scores_cum_int.tolist()),
        )
        self._ledger.append(rec)

        return {
            "cost_int": c,
            "budget_after_int": new_budget,
        }
    
    # -------- Introspection -----------
    def ledger(self) -> List[StepRecord]:
        """Return the current ledger (list of StepRecord)."""
        return self._ledger
    
    def as_dict_list(self) -> List[Dict]:
        """Return ledger as a list of plain dicts (handy for CSV/JSON export)."""
        return [asdict(r) for r in self._ledger]