"""Clone machine state for encoder simulation (no double-apply)."""

from __future__ import annotations

from greenfield.overflow import ColdStore
from greenfield.types import MachineState, Working


def clone_machine_state(state: MachineState) -> MachineState:
    cold_copy = None
    if state.cold_store is not None:
        cs: ColdStore = state.cold_store
        cold_copy = ColdStore(records=dict(cs.records), key_index=dict(cs.key_index))
    return MachineState(
        storage=state.storage.copy(),
        working=Working(),
        log=list(state.log),
        checkpoints={k: v.copy() for k, v in state.checkpoints.items()},
        gas_used=state.gas_used,
        cold_store=cold_copy,
        cold_index=dict(state.cold_index),
        cold_hits=state.cold_hits,
        overflow_evictions=state.overflow_evictions,
    )
