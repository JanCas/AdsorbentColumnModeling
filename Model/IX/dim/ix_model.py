"""Ion-exchange adsorption column — diffrax solver (forward simulation, v1).

Spec: Model/IX/IX_Column_Diffrax_Spec.md (parent doc: Modeling Adsorption Column).

Unit convention (SI, fixed once here):
    [A+], [T], [H+]   mol/m^3
    n, Q_sites        mol/kg_sorbent
    rho_p             kg/m^3 (dry sorbent)
    u_s               m/s
    x, L              m
    k                 1/s
    K*                dimensionless Li+/H+ mass-action constant
    K_w               (mol/m^3)^2

Conversions from source fit (mol/L basis):
    K_w_source = 1e-14 (mol/L)^2  ->  K_W_SI = 1e-8 (mol/m^3)^2
    K* is dimensionless and needs no unit conversion.

This module is pure JAX. Plotting (matplotlib + JansPlottingStuff) lives in
Model/IX/plotting.py and verification (Spec sec.7) in Model/IX/verification.py.

Solver: selected per phase via PhaseConfig.solver, default "tsit5" (explicit).
Use "kvaerno5" when the chemistry feedback stiffens the uptake ODE — it is both
faster and more robust there, and agrees with tsit5 to four decimals where both
run. Step control is PIDController(pcoeff=0.3, icoeff=0.4, rtol=1e-4, atol=1e-7).
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
    # Total proton, TOTH [mol/m^3], on the (H+, B-, H2O) component basis:
    #     TOTH = [H+] - [OH-] + [HB]
    # Conserved under advection apart from the exchange source. Li uptake
    # releases one proton per mole, so exchange always drives T upward.
    # With no buffer this is exactly the proton excess [H+] - [OH-].
    T: jax.Array
    n: jax.Array  # mol/kg
    # Total buffer, C_B = [HB] + [B-] [mol/m^3]. Conserved with NO source
    # term: protonating the buffer moves it between HB and B- but never
    # creates or destroys it. That is what makes it a buffer.
    C_B: jax.Array


class ColumnParams(eqx.Module):
    L: jax.Array
    eps: jax.Array
    rho_p: jax.Array
    u_s: jax.Array
    Q_sites: jax.Array
    Kstar: jax.Array
    # Second, weakly proton-competing site. Q2_sites = 0 recovers the
    # single-site form exactly, whatever Kstar2 is.
    Q2_sites: jax.Array
    Kstar2: jax.Array
    k: jax.Array
    Kw: jax.Array
    A_in: jax.Array
    T_in: jax.Array
    # Buffer acid dissociation constant [mol/m^3] and inlet buffer total
    # [mol/m^3]. C_B_in = 0 disables the buffer and recovers the pure-water
    # closure exactly, whatever Ka is set to.
    Ka: jax.Array
    C_B_in: jax.Array
    eta_min: jax.Array
    n_residual: jax.Array
    N: int = eqx.field(static=True)
    equilibrium_model: str = eqx.field(
        static=True, default="li_h_mass_action"
    )

    def __post_init__(self):
        if self.equilibrium_model != "li_h_mass_action":
            raise ValueError(
                "IX equilibrium_model must be 'li_h_mass_action', "
                f"got {self.equilibrium_model!r}"
            )
        for f in ("L", "eps", "rho_p", "u_s", "Q_sites", "Kstar",
                  "Q2_sites", "Kstar2", "k", "Kw",
                  "A_in", "T_in", "Ka", "C_B_in", "eta_min", "n_residual"):
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
    C_B_in: jax.Array   # inlet buffer total [mol/m^3]
    t_max: float
    dt0: float
    cond_fn: Callable = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    save_ts: Optional[jax.Array] = None
    # Solver selector. "tsit5" (explicit, the v1 default) is fastest when the
    # uptake ODE is mild. Large LDF rates under the water closure stiffen the
    # T -> [H+] -> n_eq feedback and exhaust max_steps on Tsit5; the implicit
    # Kvaerno paths handle those. Static, so changing it retraces.
    solver: str = eqx.field(static=True, default="tsit5")

    def __post_init__(self):
        object.__setattr__(self, "A_in", jnp.asarray(self.A_in))
        object.__setattr__(self, "T_in", jnp.asarray(self.T_in))
        object.__setattr__(self, "C_B_in", jnp.asarray(self.C_B_in))
        if self.save_ts is not None:
            object.__setattr__(self, "save_ts", jnp.asarray(self.save_ts))


# --- model (pure JAX) ---

def h_plus(T: jax.Array, Kw: jax.Array) -> jax.Array:
    """Recover [H+] from proton excess [T] via the water self-ionization
    quadratic [H+]^2 - [T][H+] - Kw = 0.

    The conjugate form on the basic (negative-T) branch avoids cancellation.
    """
    root = jnp.sqrt(T * T + 4.0 * Kw)
    acidic = 0.5 * (T + root)
    basic = 2.0 * Kw / (root - T)
    return jnp.where(T >= 0.0, acidic, basic)


# --- buffered proton closure ---------------------------------------------
#
# Components (H+, B-, H2O). The weak acid HB dissociates as HB <-> H+ + B-
# with Ka = [H+][B-]/[HB], so [HB] = C_B h/(h + Ka), and the proton total is
#
#     TOTH = h - Kw/h + C_B h/(h + Ka)                                   (*)
#
# Given the two transported totals (TOTH, C_B) this is solved for h. Clearing
# denominators gives a cubic, but its coefficients span ~20 decades here
# (Kw*Ka ~ 1e-15) and Cardano loses everything to cancellation in float64, so
# (*) is solved iteratively in log space instead.
#
# The residual is monotone increasing in h -- d(TOTH)/dh = 1 + Kw/h^2 +
# C_B Ka/(h+Ka)^2 > 0 -- so bisection cannot fail and Newton cannot leave the
# bracket. In log space the Newton slope is
#
#     d(TOTH)/d(ln h) = h + Kw/h + C_B Ka h/(h + Ka)^2
#
# which is exactly ln(10) times the Van Slyke buffer capacity. Setting C_B = 0
# reduces (*) to the water quadratic and this solver to h_plus, exactly.

_H_MIN = 1.0e-14   # mol/m^3, about pH 17
_H_MAX = 1.0e6     # mol/m^3, about pH -3
_N_BISECT = 32     # 36 decades / 2^32 -- far below float64 resolution
_N_NEWTON = 4      # polish; quadratic convergence from inside the bracket


def _toth_of_h(h: jax.Array, C_B: jax.Array, Ka: jax.Array,
               Kw: jax.Array) -> jax.Array:
    """Total proton implied by a free-proton concentration. Equation (*)."""
    return h - Kw / h + C_B * h / (h + Ka)


def proton_from_totals(T: jax.Array, C_B: jax.Array, Ka: jax.Array,
                       Kw: jax.Array) -> jax.Array:
    """Solve TOTH = h - Kw/h + C_B h/(h+Ka) for [H+] in mol/m^3.

    Bisection in log space to bracket, then Newton to polish. Both run a fixed
    number of iterations so the function is jit-friendly and has no data
    dependent control flow.
    """
    lo = jnp.full_like(jnp.asarray(T, dtype=float), jnp.log(_H_MIN))
    hi = jnp.full_like(lo, jnp.log(_H_MAX))

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        below = _toth_of_h(jnp.exp(mid), C_B, Ka, Kw) < T
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid))

    lo, hi = jax.lax.fori_loop(0, _N_BISECT, bisect, (lo, hi))

    def newton(_, x):
        h = jnp.exp(x)
        hk = h + Ka
        f = h - Kw / h + C_B * h / hk - T
        df = h + Kw / h + C_B * Ka * h / (hk * hk)
        return jnp.clip(x - f / df, jnp.log(_H_MIN), jnp.log(_H_MAX))

    return jnp.exp(jax.lax.fori_loop(0, _N_NEWTON, newton, 0.5 * (lo + hi)))


def buffer_capacity(h: jax.Array, C_B: jax.Array, Ka: jax.Array,
                    Kw: jax.Array) -> jax.Array:
    """Van Slyke buffer capacity, mol/m^3 per pH unit. Diagnostic only."""
    return 2.302585092994046 * (h + Kw / h + C_B * Ka * h / (h + Ka) ** 2)


def pH_from_state(T: jax.Array, C_B: jax.Array,
                  args: ColumnParams) -> jax.Array:
    """Operational pH from the transported totals."""
    return -jnp.log10(
        proton_from_totals(T, C_B, args.Ka, args.Kw) / LITER_PER_M3
    )


def proton_concentration(T: jax.Array, C_B: jax.Array,
                         args: ColumnParams) -> jax.Array:
    """Liquid [H+] in mol/m^3 from the transported totals."""
    return proton_from_totals(T, C_B, args.Ka, args.Kw)


def n_eq(A: jax.Array, Hp: jax.Array, args: ColumnParams) -> jax.Array:
    """Two-site Li+/H+ mass-action equilibrium loading, mol/kg.

        n_eq = Q1 K1 A/(H + K1 A)  +  Q2 K2 A/(H + K2 A)

    Site 1 is the ordinary ion-exchange site. Site 2 competes much more weakly
    with protons (K2 >> K1), so it stays loaded far down the pH range and
    produces the low-pH capacity plateau that a single site cannot: Zhang's
    Figure 6 retains 9-15% of capacity down to pH 2.05 where single-site mass
    action gives zero.

    Unlike a constant loading floor, this still goes to zero as A -> 0, so it
    creates no spurious sink and needs no cancelling initial condition. That is
    why the earlier `equilibrium_offset` had to be initialised at its own value
    and therefore contributed nothing.

    ``Q2_sites = 0`` recovers the single-site form exactly.
    """
    site1 = args.Q_sites * (args.Kstar * A) / (Hp + args.Kstar * A)
    site2 = args.Q2_sites * (args.Kstar2 * A) / (Hp + args.Kstar2 * A)
    return site1 + site2


def vector_field(t, y: State, args: ColumnParams) -> State:
    A, T, n, C_B = y.A, y.T, y.n, y.C_B

    Hp = proton_concentration(T, C_B, args)
    dn = args.k * (n_eq(A, Hp, args) - n)

    A_up = jnp.concatenate([args.A_in[None], A[:-1]])
    T_up = jnp.concatenate([args.T_in[None], T[:-1]])
    B_up = jnp.concatenate([args.C_B_in[None], C_B[:-1]])
    dAdx = (A - A_up) / args.dx
    dTdx = (T - T_up) / args.dx
    dBdx = (C_B - B_up) / args.dx

    coup = (1.0 - args.eps) / args.eps * args.rho_p * dn
    dA = -(args.u_s / args.eps) * dAdx - coup
    dT = -(args.u_s / args.eps) * dTdx + coup
    # The buffer is advected only. Exchange protonates it (B- -> HB) but
    # cannot create or destroy it, so C_B carries no source term.
    dB = -(args.u_s / args.eps) * dBdx
    return State(A=dA, T=dT, n=dn, C_B=dB)


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

def _solver_for(name: str):
    """Build the requested diffrax solver.

    The implicit paths use an optimistix Newton root finder, which is why
    optimistix is a dependency of this module.
    """
    if name == "tsit5":
        return diffrax.Tsit5()
    root_finder = optimistix.Newton(rtol=1e-8, atol=1e-10)
    if name == "kvaerno5":
        return diffrax.Kvaerno5(root_finder=root_finder)
    if name == "kvaerno3":
        return diffrax.Kvaerno3(root_finder=root_finder)
    if name == "implicit_euler":
        return diffrax.ImplicitEuler(root_finder=root_finder)
    raise ValueError(
        "solver must be one of 'tsit5', 'kvaerno5', 'kvaerno3', "
        f"'implicit_euler'; got {name!r}"
    )
# PID coefficients copied from Model/AlLDH/diffrax_non_dim.py:110 — the
# diffrax default (pcoeff=0, icoeff=1) was too aggressive at rejecting steps
# near the moving MTZ on this problem and exhausted max_steps before the
# breakthrough event fired.
_CTRL = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=1e-4, atol=1e-7)
# Event polishing is omitted: with adaptive control the per-step accuracy on the
# event scalar is already O(rtol * |y|), and a bisection refine costs more than
# it buys.
_TERM = diffrax.ODETerm(vector_field)


def initial_state(N: int, A0: float = 0.0, T0: float = 0.0, n0: float = 0.0,
                  C_B0: float = 0.0) -> State:
    """Uniform initial profile across the bed."""
    return State(
        A=jnp.full((N,), A0),
        T=jnp.full((N,), T0),
        n=jnp.full((N,), n0),
        C_B=jnp.full((N,), C_B0),
    )


@eqx.filter_jit
def run_phase(y: State, ph: PhaseConfig, base_args: ColumnParams) -> diffrax.Solution:
    """Solve one phase. Inlet (A_in, T_in) is taken from ph and overlaid on base_args."""
    args = base_args.replace(A_in=ph.A_in, T_in=ph.T_in, C_B_in=ph.C_B_in)
    saveat = (
        diffrax.SaveAt(ts=ph.save_ts, t1=True)
        if ph.save_ts is not None
        else diffrax.SaveAt(t1=True)
    )
    return diffrax.diffeqsolve(
        _TERM,
        _solver_for(ph.solver),
        t0=0.0,
        t1=ph.t_max,
        dt0=ph.dt0,
        y0=y,
        args=args,
        stepsize_controller=_CTRL,
        saveat=saveat,
        event=diffrax.Event(ph.cond_fn),
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
