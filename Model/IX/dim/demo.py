"""Runnable demo for the IX column model.

Wires Model/IX/{ix_model, plotting, verification} into a load -> desorb cycle
and runs the sec.7 verification suite. Run with:

    python -m Model.IX.demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

# Allow `from utils.logging_setup import ...` whether run as a script or
# via `python -m Model.IX.demo`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ix_model import (
    ColumnParams,
    PhaseConfig,
    adsorption_event,
    desorption_loading_drained,
    initial_state,
    run_cycle,
    K_W_SI,
)
from plotting import apply_style, plot_breakthrough, plot_eluate, plot_profiles
from utils.logging_setup import configure_logging
from verification import run_all

_log = logging.getLogger(__name__)


def _demo_params(N: int = 40) -> ColumnParams:
    return ColumnParams(
        L=5e-3, eps=0.4, rho_p=1000.0, u_s=1e-3,
        Q_sites=3.98, Kstar=1.09e-9, Q2_sites=0.0, Kstar2=3.24e-4,
        k=5e-1, Kw=K_W_SI,
        A_in=20.0, T_in=-10.0,           # placeholder; phases override
        Ka=10.0 ** (-9.25) * 1000.0, C_B_in=0.0,   # unbuffered
        eta_min=0.3, n_residual=0.4,
        N=N,
    )


def _demo_phases() -> list[PhaseConfig]:
    # Adsorption: t_break ~ 250 s at N=40 (front limited by Da ~ 1).
    # Desorption: drains in ~10-20 s once the acidic eluent reaches the bed.
    # Per-phase save grids keep both curves well-resolved near their event.
    return [
        PhaseConfig(
            name="adsorption",
            A_in=20.0, T_in=-10.0, C_B_in=0.0,   # Li+ brine at pH 12
            t_max=3.0e3, dt0=1e-3,
            cond_fn=adsorption_event,
            max_steps=5_000_000,
            save_ts=jnp.linspace(0.0, 3.0e3, 500),
        ),
        PhaseConfig(
            name="desorption",
            A_in=0.0, T_in=100.0, C_B_in=0.0,    # acidic eluent at pH 1
            t_max=2.0e2, dt0=1e-3,
            cond_fn=desorption_loading_drained,
            max_steps=5_000_000,
            save_ts=jnp.linspace(0.0, 2.0e2, 400),
        ),
    ]


def main() -> None:
    configure_logging()
    _log.info("IX demo starting")
    apply_style()

    params = _demo_params()
    phases = _demo_phases()
    _log.info(
        "demo params: L=%.4f m, eps=%.2f, u_s=%.2e m/s, k=%.2f /s, N=%d; "
        "eta_min=%.2f, n_residual=%.2f",
        float(params.L), float(params.eps), float(params.u_s),
        float(params.k), int(params.N),
        float(params.eta_min), float(params.n_residual),
    )
    y0 = initial_state(int(params.N), A0=0.0, T0=float(phases[0].T_in), n0=0.0)

    sols, t_offsets = run_cycle(y0, phases, params)

    _log.info("rendering plots ...")
    plot_breakthrough(
        sols, params, t_offsets,
        phase_A_ins=[float(p.A_in) for p in phases],
        phase_names=[p.name for p in phases],
    )
    plot_profiles(sols[0], params)
    plot_eluate(sols[1], params, t_offset=t_offsets[1])

    print()
    run_all()
    _log.info("calling plt.show() — close all figures to exit")
    plt.show()


if __name__ == "__main__":
    main()
