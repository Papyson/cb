# Selection.py

"""
Selection utilities:
-Build feasibility masks
-Validate actions
-Mutate the remaining set deterministically
-Termination checks

Thiw module does not touch scores or rewards. It only handles feasibility and
remaining-set mutation so env.step() order stays explicit and testable. 
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from citybuilder_env.state import EpisodeState

def build_action_mask(state: EpisodeState, costs_int: np.ndarray) -> np.ndarray:
    """
    Boolean mask over all items indicating which are currently selectable.

    An action (item_id) is feasible iff:
        it is present in state.remaining_ids, and
        its cost <= state.budget_rem_int

    Parameters
    ------------
    state: EpisodeState (remaining_ids must be sorted
    costs_int: np.int64[N]

    Returns
    ---------
    mask: np.ndarray[bool] of shape (N,)
    """

    N = int(costs_int.shape[0])
    mask = np.zeros(N, dtype=bool)
    if state.remaining_ids.size == 0:
        return mask
    mask[state.remaining_ids] = True
    mask &= (costs_int <= int(state.budget_rem_int))
    return mask

def is_feasible(action_id: int, state: EpisodeState, costs_int: np.ndarray) -> bool:
    """
    Chcek feasibilty using the state's current remaining set and budget.
    """
    return state.is_feasible(int(action_id), costs_int)

def apply_selection(state: EpisodeState, action_id: int) -> None:
    """
    Mutate the episode state after a validate selection:
        -Remove action_id from remaining_ids
        -append to selected_ids
        -increment step index
    """
    state.mark_selected(int(action_id))
    state.step_idx = int(state.step_idx) + 1


@dataclass(frozen=True)
class DoneCheck:
    done: bool 
    reason: str # "budget_exhausted" | "no_items" | "max_steps" | "continue"
    
def check_done(state: EpisodeState, max_steps: int) -> DoneCheck:
    """
    Termination checks in a deterministic order of precedence.

    Order
    ---------
    1. budget_exhausted: budget <= 0
    2. no_items: remaining set is empty 
    3. max_steps: step_idx >= max_steps
    else continue
    """

    if int(state.budget_rem_int) <= 0:
        return DoneCheck(True, "budget_exhausted")
    if state.remaining_ids.size == 0:
        return DoneCheck(True, "no_items")
    if int(state.step_idx) >= int(max_steps):
        return DoneCheck(True, "max_steps")
    return DoneCheck(False, "continue")
