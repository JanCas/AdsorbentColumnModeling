"""Non-dimensional ion-exchange column — q_max-normalised variant.

Sibling of Model/IX/NonDim/ix_nondim.py. The ONLY structural change is the
loading normalisation:

    n* = q / q_max          (this file)
    n* = q / q_eq(A_in)     (the sibling)

Under n* = q/q_max the Langmuir isotherm
    q_eq = q_max · K A / (H + K A)
becomes, with A* = A/A_in, H* = [H+]/A_in, and K* the bare dimensionless
mass-action constant (A_in-consistent, no hidden inlet-proton scale):

    n*_eq(A*, H*) = K* A* / (H* + K* A*),   n*_eq ∈ [0, 1]

At feed (A*=1, H*=H_in_ads) the loading equilibrium is
n*_eq = K*/(H_in_ads + K*) = f < 1 — the loading axis measures fraction of
total Langmuir capacity occupied. Λ uses the q_max loading scale:

    Λ = (1-ε)/ε · ρ_p · q_max / A_ref      (was q_eq(A_in)/A_ref)
      = Λ_feed · (H_in_ads + K*)/K*

Governing equations on x* ∈ [0,1], t* ≥ 0 are unchanged in form:

    ∂A*/∂t* = -∂A*/∂x* - Λ · dn*/dt*
    ∂T*/∂t* = -∂T*/∂x* + Λ · dn*/dt*
    dn*/dt* = Da · (n*_eq - n*)
    H*      = ( T* + sqrt(T*² + 4ω) ) / 2

Phase inlets (both convert an inlet proton level H* to T* via the exact
positive-root inverse T* = H* - ω/H*):
    Load:   A*_in = 1,  T*_in = H_in_ads - ω/H_in_ads  (feed proton level)
    Desorb: A*_in = 0,  T*_in = H_in_des - ω/H_in_des  (strip proton level)

Float64 is required: ω ∈ [10^-13, 10^-6] would lose 4ω entirely in the
h_star quadratic in float32 (same bug pattern as the dim IX hit).

This module is pure JAX. Logging is at module-load / module-import only;
NEVER inside @eqx.filter_jit bodies (would fire at trace time, not runtime).
The Sobol harness in Sensitivity/sobol_driver.py handles per-sample logging.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import jax

# Float64 must be enabled BEFORE jax.numpy import for the policy to take.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

_log = logging.getLogger(__name__)
# Reminder: do NOT put _log calls inside @eqx.filter_jit bodies (vector_field,
# run_phase, qois). Use the orchestration layer if you need runtime info.


# --- module constants -------------------------------------------------------

# Adsorption-phase inlet proton level H_in_ads = [H+]_in/A_in (the feed pH),
# spec §2.1: a constant of the non-dim choice. With A_ref ~ 1 mol/L and feed
# pH 12, [H+]_in ≈ 1e-12 mol/L -> H_in_ads ≈ 1e-9. Callers may override per
# NonDimParams.H_in_ads for a different convention.
H_IN_ADS_DEFAULT = 1.0e-9

# Desorption-phase inlet proton level H_in_des = [H+]_in/A_in (the strip pH).
# The eluent is strongly acidic, so this is many orders above H_in_ads.
H_IN_DES_DEFAULT = 1.0e2

# Load-phase outlet-concentration threshold (the AlLDH-non-dim semantic:
# stop when outlet A* >= C_thresh_ads). 0.7 corresponds to 30 %
# extraction-efficiency loss, well above the H+/A+ stoichiometric plateau.
C_THRESH_ADS_DEFAULT = 0.7

# Desorb-phase outlet-A* threshold (AlLDH semantic: stop when outlet A*
# drops below C_thresh_des). 0.1 is the same range as the feed-eq variant.
C_THRESH_DES_DEFAULT = 0.1


# --- types ------------------------------------------------------------------

class State(eqx.Module):
    """Field state at every cell-centred node + a cumulative outlet integrator.

    A, T, n are shape (N,) — the dimensionless aqueous A+, proton excess T*,
    and sorbent loading n* (= q/q_max in this variant). R_outlet is a scalar
    accumulator: dR/dt* = A*[-1] (matches AlLDH non-dim's `R_outlet = ∫
    C*(ζ=1) dτ` from Model/AlLDH/diffrax_non_dim_qmax.py). Reset to 0 at the
    start of each phase so the recovered amount is per-phase.
    """
    A: jax.Array
    T: jax.Array
    n: jax.Array
    R_outlet: jax.Array


class NonDimParams(eqx.Module):
    """All inputs to the forward model.

    Sampled groups (Sobol factors): Lambda, Da, K_star, omega, H_in_des.
    Fixed controls:                 C_thresh_ads, C_thresh_des, H_in_ads.
    Per-phase inlets (swapped via .replace()): A_in, T_in.
    Shape-determining static field: N.

    Note on Λ: in this q_max-normalised variant,
        Λ = (1-ε)/ε · ρ_p · q_max / A_ref
    (= Λ_feed-eq · (H_in_ads + K*)/K* if converting from the feed-eq sibling).

    Note on K_star: the bare dimensionless mass-action constant in the
    isotherm n*_eq = K* A*/(H* + K* A*). It was formerly named `theta` (the
    Langmuir feed-favorability K*/H*_in); the isotherm revert to the bare
    form moved the sampled axis to K* itself.

    `H_in_ads` and `H_in_des` are the SAME physical quantity — the
    dimensionless inlet proton level H* = [H+]_in/A_in — in the two phases
    (feed pH and strip pH respectively).
    """
    # Sampled
    Lambda:   jax.Array
    Da:       jax.Array
    K_star:   jax.Array
    omega:    jax.Array
    H_in_des: jax.Array
    # Per-phase inlets (overridden via .replace per phase)
    A_in: jax.Array
    T_in: jax.Array
    # Fixed controls (same role + bound semantics as AlLDH non-dim's
    # `C_thresh_ads`/`C_thresh_des` — outlet thresholds for load and desorb).
    C_thresh_ads: jax.Array
    C_thresh_des: jax.Array
    H_in_ads:     jax.Array
    # Grid
    N: int = eqx.field(static=True)

    def __post_init__(self):
        for f in ("Lambda", "Da", "K_star", "omega", "H_in_des",
                  "A_in", "T_in",
                  "C_thresh_ads", "C_thresh_des", "H_in_ads"):
            object.__setattr__(self, f, jnp.asarray(getattr(self, f)))

    @property
    def dx(self) -> jax.Array:
        # In non-dim, x* ∈ [0,1] -> dx* = 1/N (cell-centred).
        return jnp.asarray(1.0) / self.N

    def replace(self, **kwargs) -> "NonDimParams":
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


# --- model (pure JAX) -------------------------------------------------------

def h_star(T: jax.Array, omega: jax.Array) -> jax.Array:
    """Closed-form positive root of H*² - T*·H* - ω = 0.
    Smooth in T* over the basic/acidic divide; do NOT branch or clamp.
    """
    return 0.5 * (T + jnp.sqrt(T * T + 4.0 * omega))


def n_eq_star(A: jax.Array, H: jax.Array, args: NonDimParams) -> jax.Array:
    """Langmuir isotherm under q_max normalisation: n*_eq = q_eq/q_max.

    Bare mass-action form, A_in-consistent (no hidden inlet-proton scale):
        n*_eq(A*, H*) = K* A* / (H* + K* A*)
    At feed (A*=1, H*=H_in_ads): n*_eq = K*/(H_in_ads + K*) = f < 1. Bounded
    in [0, 1] with the saturation limit n*_eq → 1 as A* → ∞ (or as H* → 0).
    """
    return args.K_star * A / (H + args.K_star * A)


def vector_field(t, y: State, args: NonDimParams) -> State:
    A, T, n = y.A, y.T, y.n

    # Local proton level H* = h_star(T*, ω)
    H = h_star(T, args.omega)

    # Isotherm and LDF
    ne = n_eq_star(A, H, args)
    dn = args.Da * (ne - n)

    # First-order upwind advection with Dirichlet inlet ghost.
    A_up = jnp.concatenate([args.A_in[None], A[:-1]])
    T_up = jnp.concatenate([args.T_in[None], T[:-1]])
    dAdx = (A - A_up) / args.dx
    dTdx = (T - T_up) / args.dx

    # Coupling: A+T is purely advected (signs anti-symmetric), so verification
    # of the conservation is a clean mass-balance check.
    coup = args.Lambda * dn
    dA = -dAdx - coup
    dT = -dTdx + coup
    # R_outlet accumulator: dR/dt* = A*[-1]  (outlet concentration)
    # — mirrors AlLDH non-dim's column_ode.
    return State(A=dA, T=dT, n=dn, R_outlet=A[-1])


# --- events -----------------------------------------------------------------
#
# Both events are written as smooth scalar residuals (sign changes at the
# crossing) so they're root-finder-friendly later. Mathematically identical
# to the boolean form used in AlLDH non-dim
# (Model/AlLDH/diffrax_non_dim_qmax.py).

def adsorption_event(t, y: State, args: NonDimParams, **kw):
    """Load phase ends when outlet A* >= C_thresh_ads.
    Sign: positive while outlet is clean, negative after the crossing.
    """
    return args.C_thresh_ads - y.A[-1]


def desorption_event(t, y: State, args: NonDimParams, **kw):
    """Desorption ends when outlet concentration A*[-1] <= C_thresh_des.

    Bool form: fires at the first step where the condition is True,
    including t=0 if the outlet is already at/below threshold. See
    Results/Sensitivity/IX/1.5_alldhlo/event_reformulation_report.html
    (§3, §4 Option C) for the diagnosis.

    Requires C_thresh_des < C_thresh_ads (enforced by the Sobol adapter
    via a clamp; see Sensitivity/adapters/ix_nondim.py).
    """
    return y.A[-1] <= args.C_thresh_des


# --- solver -----------------------------------------------------------------

_SOLVER = diffrax.Tsit5()
# Kvaerno5/Kvaerno3 (implicit ESDIRK) were tried as a fix for NaN failures
# at log10(Λ·Da·θ) ≳ 4 in joint/1.6, but both were unacceptably slow
# (5-10× Tsit5 wall-clock). Pragmatic fix: trim the Sobol box via
# --log10-da-range to keep the stiff corner out.
_CTRL = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=1e-4, atol=1e-7)
_TERM = diffrax.ODETerm(vector_field)


def initial_state(N: int, A0: float = 0.0, T0: float = 0.0, n0: float = 0.0) -> State:
    """Uniform initial profile across the bed; R_outlet starts at 0."""
    return State(
        A=jnp.full((N,), A0, dtype=jnp.float64),
        T=jnp.full((N,), T0, dtype=jnp.float64),
        n=jnp.full((N,), n0, dtype=jnp.float64),
        R_outlet=jnp.asarray(0.0, dtype=jnp.float64),
    )


@eqx.filter_jit
def run_phase(y: State, ph: PhaseConfig, base_args: NonDimParams) -> diffrax.Solution:
    """Solve one phase. Inlet (A_in, T_in) overlaid on base_args via .replace."""
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
        event=diffrax.Event(ph.cond_fn),
        max_steps=ph.max_steps,
        # Return a NaN/Inf-padded solution instead of raising when max_steps
        # is exhausted, so a slow/stiff sample is censored (skipped by SALib)
        # rather than crashing the whole sweep. _last_finite_idx handles it.
        throw=False,
    )


# --- QoI wrapper ------------------------------------------------------------

_T_MAX_LOAD_DEFAULT = 500.0
_T_MAX_DES_DEFAULT  = 500.0
_MAX_STEPS_DEFAULT  = 4_000_000


def _default_load_phase(t_max: float = _T_MAX_LOAD_DEFAULT,
                        max_steps: int = _MAX_STEPS_DEFAULT) -> PhaseConfig:
    return PhaseConfig(
        name="load",
        A_in=jnp.asarray(1.0),
        T_in=jnp.asarray(0.0),     # placeholder; qois sets per omega
        t_max=t_max, dt0=1e-3,
        cond_fn=adsorption_event,
        max_steps=max_steps,
        save_ts=None,
    )


def _default_des_phase(t_max: float = _T_MAX_DES_DEFAULT,
                       max_steps: int = _MAX_STEPS_DEFAULT) -> PhaseConfig:
    return PhaseConfig(
        name="des",
        A_in=jnp.asarray(0.0),
        T_in=jnp.asarray(0.0),     # placeholder; qois sets per H_in_des
        t_max=t_max, dt0=1e-3,
        cond_fn=desorption_event,
        max_steps=max_steps,
        save_ts=None,
    )


def _last_finite_idx(sol: diffrax.Solution) -> jax.Array:
    """Return the index of the last finite save slot. JIT-friendly: uses
    where + argmax on a static-shape array; no Python int casting."""
    finite = jnp.isfinite(sol.ts) & jnp.isfinite(sol.ys.A[:, -1])
    indices = jnp.where(finite, jnp.arange(sol.ts.shape[0]), -1)
    return jnp.argmax(indices)


def _finite_mean(sol: diffrax.Solution, scalar_per_save: jax.Array) -> jax.Array:
    finite = jnp.isfinite(sol.ts) & jnp.isfinite(scalar_per_save)
    total = jnp.sum(jnp.where(finite, scalar_per_save, 0.0))
    n_fin = jnp.sum(finite)
    return total / jnp.maximum(n_fin, 1)


def _finite_max(scalar_per_save: jax.Array) -> jax.Array:
    return jnp.max(jnp.where(jnp.isfinite(scalar_per_save),
                             scalar_per_save, -jnp.inf))


def _bed_mean_at(sol: diffrax.Solution, idx: jax.Array) -> jax.Array:
    """Spatial mean over the bed at time index `idx` (cell-centred, dx*=1/N)."""
    n_field = sol.ys.n[idx]
    return jnp.mean(n_field)


def _phase_end_time(sol: diffrax.Solution) -> jax.Array:
    finite = jnp.isfinite(sol.ts)
    return jnp.max(jnp.where(finite, sol.ts, -jnp.inf))


@eqx.filter_jit
def qois(
    nd: NonDimParams,
    load_phase: PhaseConfig | None = None,
    des_phase:  PhaseConfig | None = None,
) -> dict[str, jax.Array]:
    """Run one load → desorb cycle and return the AlLDH non-dim metric set.

        tau_ads       — dimensionless adsorption time at outlet cutoff
        tau_des       — dimensionless desorption time at outlet cutoff
        U_b           — bed utilisation at end of adsorption: ∫ n*(ζ) dζ
                         (fraction of q_max occupied; max θ/(1+θ) at feed eq)
        R_outlet_des  — A recovered at outlet during desorption: ∫ A*(ζ=1) dτ
        productivity  — R_outlet_des / (tau_ads + tau_des)
        R_release     — Λ · max(U_b - ⟨n⟩_des_end, 0)
        R_wash        — max(⟨A⟩_load_end - ⟨A⟩_des_end, 0)

    Same keys as the feed-eq sibling for direct cross-model comparison.
    Note the absolute scale of U_b and R_release shifts because Λ and the
    loading axis use q_max as their basis here.
    """
    if load_phase is None:
        load_phase = _default_load_phase()
    if des_phase is None:
        des_phase = _default_des_phase()

    N = int(nd.N)

    # Inlet boundary T* per phase via the exact positive-root inverse
    # T* = H* - ω/H*  (so H*(T_load)==H_in_ads, H*(T_des)==H_in_des).
    T_load = nd.H_in_ads - nd.omega / nd.H_in_ads
    T_des  = nd.H_in_des - nd.omega / nd.H_in_des
    load_phase = eqx.tree_at(lambda p: p.T_in, load_phase, jnp.asarray(T_load))
    des_phase = eqx.tree_at(lambda p: p.T_in, des_phase, jnp.asarray(T_des))

    y0_load = State(
        A=jnp.zeros((N,)),
        T=jnp.full((N,), T_load),
        n=jnp.zeros((N,)),
        R_outlet=jnp.asarray(0.0),
    )

    # --- Phase 1: load ----------------------------------------------------
    sol_load = run_phase(y0_load, load_phase, nd)
    idx_load_end = _last_finite_idx(sol_load)
    tau_ads = sol_load.ts[idx_load_end]
    U_b = _bed_mean_at(sol_load, idx_load_end)
    y_at_load_end = jax.tree_util.tree_map(lambda a: a[idx_load_end], sol_load.ys)
    y0_des = eqx.tree_at(lambda y: y.R_outlet, y_at_load_end,
                         jnp.asarray(0.0))

    # --- Phase 2: desorb --------------------------------------------------
    sol_des = run_phase(y0_des, des_phase, nd)
    idx_des_end = _last_finite_idx(sol_des)
    tau_des = sol_des.ts[idx_des_end]
    R_outlet_des = sol_des.ys.R_outlet[idx_des_end]
    R_outlet_des = jnp.maximum(R_outlet_des, 1e-12)
    productivity = R_outlet_des / (tau_ads + tau_des)

    A_load_end_mean = jnp.mean(sol_load.ys.A[idx_load_end])
    A_des_end_mean  = jnp.mean(sol_des.ys.A[idx_des_end])
    n_des_end_mean  = _bed_mean_at(sol_des, idx_des_end)
    R_release = nd.Lambda * jnp.maximum(U_b - n_des_end_mean, 0.0)
    R_wash    = jnp.maximum(A_load_end_mean - A_des_end_mean, 0.0)

    # Outlet proton at end of desorption — diagnostic for acid breakthrough.
    # H*_out = h_star(T*[-1], ω) at idx_des_end; H*_out / H_in_ads expresses
    # it relative to the feed (adsorption) inlet proton level.
    T_out_des_end = sol_des.ys.T[idx_des_end, -1]
    H_out = h_star(T_out_des_end, nd.omega)
    H_out_over_H_in = H_out / nd.H_in_ads

    # Censor samples whose solver ran out of steps (throw=False returns a
    # partial, finite-but-WRONG state at the truncation time). Emit NaN so
    # SALib skips them instead of treating the truncated state as the answer.
    # Event/t1 termination have distinct result codes and stay valid.
    _MS = diffrax.RESULTS.max_steps_reached
    valid = (sol_load.result != _MS) & (sol_des.result != _MS)
    keep = jnp.where(valid, 1.0, jnp.nan)

    return {
        "tau_ads":      tau_ads * keep,
        "tau_des":      tau_des * keep,
        "U_b":          U_b * keep,
        "R_outlet_des": R_outlet_des * keep,
        "R_release":    R_release * keep,
        "R_wash":       R_wash * keep,
        "productivity": productivity * keep,
        "H_out":           H_out * keep,
        "H_out_over_H_in": H_out_over_H_in * keep,
    }


def default_params(
    *,
    Lambda: float = 1.0e3,
    Da: float = 1.0,
    K_star: float = H_IN_ADS_DEFAULT,   # K* ≈ H_in_ads -> feed loading f ≈ 0.5
    omega: float = 1.0e-10,
    H_in_des: float = H_IN_DES_DEFAULT,
    C_thresh_ads: float = C_THRESH_ADS_DEFAULT,
    C_thresh_des: float = C_THRESH_DES_DEFAULT,
    H_in_ads: float = H_IN_ADS_DEFAULT,
    N: int = 50,
) -> NonDimParams:
    """Convenience constructor for a sensible mid-range NonDimParams (q_max basis).

    `K_star` is the bare dimensionless mass-action constant; `H_in_ads` /
    `H_in_des` are the adsorption- and desorption-phase inlet proton levels.
    Converting Λ between bases is Λ_qmax = Λ_feed · (H_in_ads + K*)/K*.
    """
    return NonDimParams(
        Lambda=Lambda, Da=Da, K_star=K_star, omega=omega, H_in_des=H_in_des,
        A_in=1.0, T_in=0.0,
        C_thresh_ads=C_thresh_ads, C_thresh_des=C_thresh_des,
        H_in_ads=H_in_ads, N=N,
    )
