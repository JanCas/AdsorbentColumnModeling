"""IX (ion-exchange) column model — public API."""

from ix_model import (
    ColumnParams,
    PhaseConfig,
    State,
    adsorption_event,
    desorption_loading_drained,
    h_plus,
    initial_state,
    n_eq,
    run_cycle,
    run_phase,
    vector_field,
    K_W_SI,
    LITER_PER_M3,
)

__all__ = [
    "ColumnParams",
    "PhaseConfig",
    "State",
    "adsorption_event",
    "desorption_loading_drained",
    "h_plus",
    "initial_state",
    "n_eq",
    "run_cycle",
    "run_phase",
    "vector_field",
    "K_W_SI",
    "LITER_PER_M3",
]
