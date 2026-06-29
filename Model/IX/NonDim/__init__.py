"""Non-dimensional IX (ion-exchange) column model — public API.

The events and the returned QoI dict are deliberately ALIGNED with the
AlLDH non-dim model (Model/AlLDH/diffrax_non_dim_qmax.py) so Sobol indices
computed on one are directly comparable with the other.
"""

from ix_nondim_qmax import (
    C_THRESH_ADS_DEFAULT,
    C_THRESH_DES_DEFAULT,
    H_IN_ADS_DEFAULT,
    H_IN_DES_DEFAULT,
    NonDimParams,
    PhaseConfig,
    State,
    adsorption_event,
    default_params,
    desorption_event,
    h_star,
    initial_state,
    n_eq_star,
    qois,
    run_phase,
    vector_field,
)

__all__ = [
    "C_THRESH_ADS_DEFAULT",
    "C_THRESH_DES_DEFAULT",
    "H_IN_ADS_DEFAULT",
    "H_IN_DES_DEFAULT",
    "NonDimParams",
    "PhaseConfig",
    "State",
    "adsorption_event",
    "default_params",
    "desorption_event",
    "h_star",
    "initial_state",
    "n_eq_star",
    "qois",
    "run_phase",
    "vector_field",
]
