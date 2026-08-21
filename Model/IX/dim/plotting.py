"""Matplotlib plots for the IX column model.

Kept separate from `ix_model.py` so the JAX core stays import-clean (no
matplotlib pulled in when the model is used inside an optimizer / vmap).
"""

import logging
from typing import Sequence

import diffrax
import matplotlib.pyplot as plt
import numpy as np

import JansPlottingStuff as JPS

from ix_model import ColumnParams, pH_from_state

_log = logging.getLogger(__name__)


def apply_style() -> None:
    """Apply Jan's matplotlib style. Call once near the top of a demo script."""
    _log.debug("applying JansPlottingStuff style")
    JPS.apply()


def _phase_grid(params: ColumnParams) -> np.ndarray:
    N = int(params.N)
    L = float(params.L)
    dx = L / N
    # Cell-centred nodes — spec sec.3 ("cell-centred", dx = L/N).
    return (np.arange(N) + 0.5) * dx


def _outlet_A(sol: diffrax.Solution) -> np.ndarray:
    return np.asarray(sol.ys.A[:, -1])


def plot_breakthrough(
    sols: Sequence[diffrax.Solution],
    params: ColumnParams,
    t_offsets: Sequence[float],
    phase_A_ins: Sequence[float],
    phase_names: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
):
    """Outlet [A+] and extraction efficiency eta vs absolute time.

    eta = 1 - A_out / A_in only makes sense for phases with A_in > 0
    (adsorption). The right panel is drawn only for those phases.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    else:
        fig = ax[0].figure if hasattr(ax, "__len__") else ax.figure

    names = phase_names if phase_names is not None else [f"phase {i}" for i in range(len(sols))]

    for sol, t0, name, A_in in zip(sols, t_offsets, names, phase_A_ins):
        ts = np.asarray(sol.ts) + float(t0)
        A_out = _outlet_A(sol)
        finite = np.isfinite(ts) & np.isfinite(A_out)
        ax[0].plot(ts[finite], A_out[finite], label=name)
        if A_in > 0:
            eta = 1.0 - A_out[finite] / A_in
            ax[1].plot(ts[finite], eta, label=name)

    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel(r"outlet $[A^+]$  [mol/m$^3$]")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel(r"extraction efficiency $\eta$")
    ax[1].axhline(float(params.eta_min), ls="--", lw=0.8, color="grey",
                  label=r"$\eta_{\min}$")
    for a in ax:
        a.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_profiles(
    sol: diffrax.Solution,
    params: ColumnParams,
    times: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
):
    """A(x), pH(x), n(x) at a handful of times within one phase.

    Filters out the Inf/NaN padding that diffrax writes past the event time.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    else:
        fig = ax[0].figure

    ts_all = np.asarray(sol.ts)
    finite = np.isfinite(ts_all) & np.isfinite(np.asarray(sol.ys.A[:, -1]))
    ts = ts_all[finite]

    if times is None:
        # 5 evenly spaced snapshots across the resolved time range
        idx_in_finite = np.linspace(0, len(ts) - 1, 5, dtype=int)
    else:
        idx_in_finite = np.array([int(np.argmin(np.abs(ts - t))) for t in times])
    # Map back to indices in the full sol.ys arrays
    finite_idx = np.where(finite)[0]
    idx = finite_idx[idx_in_finite]

    x = _phase_grid(params)
    A_all = np.asarray(sol.ys.A)
    T_all = np.asarray(sol.ys.T)
    n_all = np.asarray(sol.ys.n)

    for i in idx:
        label = f"t = {ts_all[i]:.1f} s"
        ax[0].plot(x, A_all[i], label=label)
        ax[1].plot(x, np.asarray(pH_from_state(T_all[i], CB_all[i], params)), label=label)
        ax[2].plot(x, n_all[i], label=label)

    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel(r"$[A^+]$  [mol/m$^3$]")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("pH")
    ax[2].set_xlabel("x [m]"); ax[2].set_ylabel(r"$n$  [mol/kg]")
    ax[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_eluate(
    desorb_sol: diffrax.Solution,
    params: ColumnParams,
    t_offset: float = 0.0,
    ax: plt.Axes | None = None,
):
    """Outlet [A+] during desorption + cumulative recovered moles per unit area.

    Recovery integrand is u_s * A_out (mol/m^2/s); time integral gives mol/m^2.
    Total recovered moles = u_s * A_cs * integral, but A_cs is not in params,
    so we report the column-cross-section-normalised value here.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    else:
        fig = ax[0].figure

    ts_full = np.asarray(desorb_sol.ts)
    A_full = _outlet_A(desorb_sol)
    finite = np.isfinite(ts_full) & np.isfinite(A_full)
    ts = ts_full[finite] + float(t_offset)
    A_out = A_full[finite]
    u_s = float(params.u_s)

    # Cumulative trapezoid integral of u_s * A_out vs t -> mol/m^2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (A_out[1:] + A_out[:-1]) *
                                            np.diff(ts) * u_s)])

    ax[0].plot(ts, A_out)
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"outlet $[A^+]$  [mol/m$^3$]")
    ax[1].plot(ts, cum)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"cumulative recovered  [mol/m$^2$]")
    fig.tight_layout()
    return fig
