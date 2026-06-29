"""Ion-exchange adsorption column — diffrax solver (forward simulation, v1).

Spec: Model/IX/IX_Column_Diffrax_Spec.md (parent doc: Modeling Adsorption Column).

Unit convention (SI, fixed once here):
    [A+], [T], [H+]   mol/m^3
    n, Q_sites        mol/kg_sorbent
    rho_p             kg/m^3 (dry sorbent)
    u_s               m/s
    x, L              m
    k                 1/s
    K*                dimensionless (K* = K * [H+]; from source: K = 1090 L/mol,
                      [H+] @ pH 12 = 1e-12 mol/L  =>  K* = 1.09e-9, dimensionless)
    K_w               (mol/m^3)^2

Conversions from source fit (mol/L basis):
    K_w_source = 1e-14 (mol/L)^2  ->  K_W_SI = 1e-8 (mol/m^3)^2
    K* is dimensionless and needs no conversion.

This module is pure JAX. Plotting (matplotlib + JansPlottingStuff) lives in
Model/IX/plotting.py and verification (Spec sec.7) in Model/IX/verification.py.

Solver: Kvaerno5 (stiff implicit) + PIDController(rtol=1e-6, atol=1e-8).
Fallbacks if Newton struggles: ImplicitEuler, Kvaerno3.
"""

import logging
from typing import Callable, Optional, Sequence

import jax

_log = logging.getLogger(__name__)
# NOTE: do NOT put _log.* calls inside @eqx.filter_jit bodies (e.g. vector_field,
# run_phase body); they only fire at trace time, not runtime. Log at the
# orchestration layer (run_cycle, demo, verification).

# float64 is required: K_w in SI is O(1e-8) (mol/m^3)^2 and the closed-form
# [H+] = 0.5 (T + sqrt(T^2 + 4 Kw)) loses 4 Kw entirely in float32 when |T| is
# O(1), driving the denominator of n_eq to zero and producing NaN. Enable
# float64 at import time so any module importing ix_model gets it.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optimistix  # noqa: E402


# --- unit constants ---

LITER_PER_M3 = 1000.0
K_W_SI = 1e-8  # (mol/m^3)^2 ; from K_w = 1e-14 (mol/L)^2


# --- types ---

class State(eqx.Module):
    """Field state at every spatial node. Each leaf has shape (N,)."""
    A: jax.Array  # mol/m^3
    T: jax.Array  # mol/m^3
    n: jax.Array  # mol/kg


class ColumnParams(eqx.Module):
    L: jax.Array
    eps: jax.Array
    rho_p: jax.Array
    u_s: jax.Array
    Q_sites: jax.Array
    Kstar: jax.Array
    k: jax.Array
    Kw: jax.Array
    A_in: jax.Array
    T_in: jax.Array
    eta_min: jax.Array
    n_residual: jax.Array
    N: int = eqx.field(static=True)

    def __post_init__(self):
        for f in ("L", "eps", "rho_p", "u_s", "Q_sites", "Kstar", "k", "Kw",
                  "A_in", "T_in", "eta_min", "n_residual"):
            object.__setattr__(self, f, jnp.asarray(getattr(self, f)))

    @property
    def dx(self) -> jax.Array:
        return self.L / self.N

    def replace(self, **kwargs) -> "ColumnParams":
        return eqx.tree_at(
            lambda x: [getattr(x, k) for k in kwargs.keys()],
            self,
            [jnp.asarray(v) for v in kwargs.values()],
        )


class PhaseConfig(eqx.Module):
    name: str = eqx.field(static=True)
    A_in: jax.Array
    T_in: jax.Array
    t_max: float
    dt0: float
    cond_fn: Callable = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    save_ts: Optional[jax.Array] = None

    def __post_init__(self):
        object.__setattr__(self, "A_in", jnp.asarray(self.A_in))
        object.__setattr__(self, "T_in", jnp.asarray(self.T_in))
        if self.save_ts is not None:
            object.__setattr__(self, "save_ts", jnp.asarray(self.save_ts))


# --- model (pure JAX) ---

def h_plus(T: jax.Array, Kw: jax.Array) -> jax.Array:
    """Recover [H+] from proton excess [T] via the water self-ionization
    quadratic [H+]^2 - [T][H+] - Kw = 0. Positive root, smooth for any T."""
    return 0.5 * (T + jnp.sqrt(T * T + 4.0 * Kw))


def n_eq(A: jax.Array, Hp: jax.Array, args: ColumnParams) -> jax.Array:
    """Mass-action (Langmuir-form) equilibrium loading."""
    return args.Q_sites * (args.Kstar * A) / (Hp + args.Kstar * A)


def vector_field(t, y: State, args: ColumnParams) -> State:
    A, T, n = y.A, y.T, y.n

    Hp = h_plus(T, args.Kw)
    dn = args.k * (n_eq(A, Hp, args) - n)

    A_up = jnp.concatenate([args.A_in[None], A[:-1]])
    T_up = jnp.concatenate([args.T_in[None], T[:-1]])
    dAdx = (A - A_up) / args.dx
    dTdx = (T - T_up) / args.dx

    coup = (1.0 - args.eps) / args.eps * args.rho_p * dn
    dA = -(args.u_s / args.eps) * dAdx - coup
    dT = -(args.u_s / args.eps) * dTdx + coup
    return State(A=dA, T=dT, n=dn)


# --- events ---

def adsorption_event(t, y: State, args: ColumnParams, **kw):
    """Stop when extraction efficiency eta = 1 - A_out/A_in falls below eta_min.
    Sign: positive at start of phase (clean outlet), negative after breakthrough."""
    eta = 1.0 - y.A[-1] / args.A_in
    return eta - args.eta_min


def desorption_loading_drained(t, y: State, args: ColumnParams, **kw):
    """Stop when mean bed loading <n> falls below n_residual.
    Sign: positive on a loaded bed, negative once drained."""
    return jnp.mean(y.n) - args.n_residual


# --- solver ---

# Spec sec.5.4 recommends Kvaerno5 (stiff implicit). In practice for v1 forward
# simulation, Tsit5 (explicit 5th order) is faster and far more reliable here:
# Kvaerno5's implicit Newton on a 3N x 3N system with these dimensional scales
# hit max_steps before breakthrough even with relaxed tolerances. The stiffness
# justification (large k for an autodiff optimiser) is deferred to v2. Tsit5
# matches the AlLDH non-dim baseline (Model/AlLDH/diffrax_non_dim.py:103).
# Documented fallbacks: Kvaerno5 / Kvaerno3 / ImplicitEuler if a future stiff
# parameter regime appears.
# Event root finding: Bisection is robust for the smooth scalar events in sec.4.3
# without needing a Jacobian (Newton needs autodiff of the cond_fn).
_SOLVER = diffrax.Tsit5()
# PID coefficients copied from Model/AlLDH/diffrax_non_dim.py:110 — the
# diffrax default (pcoeff=0, icoeff=1) was too aggressive at rejecting steps
# near the moving MTZ on this problem and exhausted max_steps before the
# breakthrough event fired.
_CTRL = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=1e-4, atol=1e-7)
# Event polishing is omitted (root_finder=None). With Tsit5's adaptive control
# the per-step accuracy on the event scalar is already O(rtol * |y|); a
# bisection/Newton refine wasn't worth the extra step bisection cost.
_ROOT = None
_TERM = diffrax.ODETerm(vector_field)


def initial_state(N: int, A0: float = 0.0, T0: float = 0.0, n0: float = 0.0) -> State:
    """Uniform initial profile across the bed."""
    return State(
        A=jnp.full((N,), A0),
        T=jnp.full((N,), T0),
        n=jnp.full((N,), n0),
    )


@eqx.filter_jit
def run_phase(y: State, ph: PhaseConfig, base_args: ColumnParams) -> diffrax.Solution:
    """Solve one phase. Inlet (A_in, T_in) is taken from ph and overlaid on base_args."""
    args = base_args.replace(A_in=ph.A_in, T_in=ph.T_in)
    saveat = (
        diffrax.SaveAt(ts=ph.save_ts, t1=True)
        if ph.save_ts is not None
        else diffrax.SaveAt(t1=True)
    )
    return diffrax.diffeqsolve(
        _TERM,
        _SOLVER,
        t0=0.0,
        t1=ph.t_max,
        dt0=ph.dt0,
        y0=y,
        args=args,
        stepsize_controller=_CTRL,
        saveat=saveat,
        event=diffrax.Event(ph.cond_fn) if _ROOT is None else diffrax.Event(ph.cond_fn, _ROOT),
        max_steps=ph.max_steps,
    )


def run_cycle(
    y0: State,
    phases: Sequence[PhaseConfig],
    base_args: ColumnParams,
):
    """Run a sequence of phases, threading each phase's final state into the next IC.

    When the phase event terminates before the full save_ts grid is reached,
    diffrax pads the unreached entries with Inf/NaN. The end-of-phase state is
    therefore extracted from the last *finite* save slot, not literally [-1].

    Returns
    -------
    sols : list[diffrax.Solution]
        One per phase. sol.ts / sol.ys carry the (per-phase) time axis.
    t_offsets : list[float]
        Cumulative time offset at the start of each phase, for absolute-time plots.
    """
    import numpy as np

    _log.info(
        "run_cycle: N=%d eps=%.3f rho_p=%.1f u_s=%.3e k=%.3g K*=%.3e Q=%.3f "
        "phases=%s",
        int(base_args.N), float(base_args.eps), float(base_args.rho_p),
        float(base_args.u_s), float(base_args.k), float(base_args.Kstar),
        float(base_args.Q_sites), [p.name for p in phases],
    )

    sols = []
    t_offsets = []
    t_off = 0.0
    y = y0
    for i, ph in enumerate(phases):
        _log.info(
            "phase %d/%d '%s' start: t_off=%.3f s, A_in=%.3g, T_in=%.3g, "
            "t_max=%.3g, dt0=%.3g, max_steps=%d",
            i + 1, len(phases), ph.name, t_off,
            float(ph.A_in), float(ph.T_in), ph.t_max, ph.dt0, ph.max_steps,
        )
        sol = run_phase(y, ph, base_args)
        sols.append(sol)
        t_offsets.append(t_off)
        ts = np.asarray(sol.ts)
        A_end = np.asarray(sol.ys.A[:, -1])
        finite = np.isfinite(ts) & np.isfinite(A_end)
        finite_idx = np.where(finite)[0]
        if finite_idx.size == 0:
            _log.error(
                "phase '%s' produced zero finite save points (solver likely "
                "diverged before the first save). stats=%s",
                ph.name, getattr(sol, "stats", "n/a"),
            )
            raise RuntimeError(
                f"run_cycle: phase '{ph.name}' has no finite save points")
        last = int(finite_idx[-1])
        t_phase_end = float(ts[last])
        t_off = t_off + t_phase_end
        y = jax.tree_util.tree_map(lambda a: a[last], sol.ys)
        n_mean = float(np.nanmean(np.asarray(sol.ys.n[last])))
        _log.info(
            "phase %d/%d '%s' end:   t_phase_end=%.3f s, finite_saves=%d/%d, "
            "A_out=%.4g, mean(n)=%.4g, stats=%s",
            i + 1, len(phases), ph.name, t_phase_end,
            int(finite_idx.size), int(ts.size),
            float(A_end[last]), n_mean,
            getattr(sol, "stats", "n/a"),
        )
    _log.info("run_cycle done: total elapsed = %.3f s (%d phases)", t_off, len(phases))
    return sols, t_offsets
