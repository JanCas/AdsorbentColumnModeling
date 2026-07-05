"""Dimensional 1-D packed-bed adsorption model for the AlLDH sorbent (SI units).

This is the *engineering* forward model that the NSGA-II optimizer
(``multi_optim.py``) calls to evaluate a candidate column design. It solves a
plug-flow advection PDE for the mobile (liquid) concentration ``C`` coupled to a
pseudo-second-order (PSO) linear-driving-force law for the solid loading ``n``:

    dC/dt = -u_inter * dC/dx - (1-eps)/eps * rho_p * dn/dt
    dn/dt =  k_s * (n_eq(C) - n) * |n_eq(C) - n|      (PSO: rate ~ distance^2)

with ``n_eq(C)`` an equilibrium isotherm supplied as a callable (Langmuir /
Temkin / Sips fitted to literature data). Axial dispersion is neglected, so
advection uses a first-order upwind (backward) finite difference on a uniform
grid; the inlet Dirichlet BC enters through a ghost cell and the outlet is a
free convective outflow (no BC imposed).

A full cycle is two phases chained back-to-back (see ``_run_model_jit``):
  1. Adsorption  - empty-ish bed, inlet ``C = C_in``; stops once the cumulative
                   outlet loss exceeds ``loss_fraction`` (breakthrough).
  2. Desorption  - loaded profile handed over from phase 1, inlet ``C = 0``;
                   stops once mean loading falls below ``desorption_threshold``.

Inputs: a ``ColumnParameters`` bundle (u_inter, k_s, epsilon, C_in, L, rho_p,
isotherm) plus the two scalar thresholds. Outputs: end-of-phase (t, C, n)
states, cumulative desorption outlet, and a solver-ok flag.

The heavy numerics live in ``_run_model_jit`` (``@eqx.filter_jit``); ``run_model``
is the thin, loggable Python wrapper around it.
"""
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.Dataclasses import ColumnParameters, Study, BreakthroughCurve

_log = logging.getLogger(__name__)
# `_run_model_jit` below is @eqx.filter_jit; never log inside its body or
# inside the helpers it calls under trace (column_ode, make_*_event,
# set_initial_adsorption_state, get_finish_state). Log in `run_model` only.

# Represents values on a uniform spatial grid with fixed spacing δx.
# The arithmetic operators (__add__, __mul__, ...) let diffrax treat this as a
# PyTree "vector", so ODE tangents can be scaled/added elementwise on .vals.
class SpatialDiscretisation(eqx.Module):
    δx: Float[Array, ""]
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        """Sample ``fn`` on ``n`` uniform nodes over [x0, x_final] into a grid field."""
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        δx = jnp.asarray((x_final - x0) / (n - 1), dtype=vals.dtype)
        return cls(δx, vals)

    def binop(self, other, fn):
        # Apply fn elementwise to .vals, keeping grid spacing; `other` may be a
        # bare array/scalar or another SpatialDiscretisation.
        other_vals = other.vals if isinstance(other, SpatialDiscretisation) else other
        return SpatialDiscretisation(self.δx, fn(self.vals, other_vals))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

# Full solver state: liquid concentration field C, solid loading field n, and a
# scalar accumulator for material that has left the outlet (used by the
# breakthrough / loss event).
class ColumnState(eqx.Module):
    C: SpatialDiscretisation
    n: SpatialDiscretisation
    cumulative_out: Float[Array, ""] = 0.0  # Integrated outlet flux over time


def _spatial_tangent(sd: SpatialDiscretisation, dvals_dt: Float[Array, "n"]) -> SpatialDiscretisation:
    """Return ODE tangent with fixed grid spacing (dδx/dt = 0)."""
    return SpatialDiscretisation(jnp.zeros_like(sd.δx), dvals_dt)

def column_ode(t, state: ColumnState, args: ColumnParameters):
    """Right-hand side of the coupled advection + PSO-kinetics PDE (per node).

    Returns dC/dt and dn/dt on the grid plus the outlet-flux accumulator rate.
    ``args`` carries the (constant) inlet BC ``C_in`` so switching to 0 gives the
    desorption phase without touching this function.
    """
    # jax.debug.print("args={a.C_in}", a=args)
    C = state.C
    n = state.n

    # Equilibrium loading at each node's local C, then PSO driving force:
    # dn/dt = k_s (n_eq - n)|n_eq - n|. The |.| keeps the correct sign for both
    # uptake (n_eq > n) and release (n_eq < n) while making the rate quadratic.
    n_star = jax.vmap(args.isotherm)(C.vals)
    dn_dt = args.k_s * (n_star - n.vals) * jnp.abs(n_star - n.vals)

    # Upwind (backward) difference dC/dx: shift the field one node downstream and
    # overwrite node 0 with the ghost-cell inlet value C_in (Dirichlet BC). Flow
    # is left-to-right so the upstream neighbour is the correct upwind stencil.
    C_prev = jnp.roll(C.vals, shift=1)
    C_prev = C_prev.at[0].set(args.C_in)

    advection_dc_dx = (C.vals - C_prev) / C.δx
    # Sink term: solute taken up by the solid, scaled by solid/liquid volume ratio.
    sorption = (1 - args.epsilon) / args.epsilon * args.rho_p * dn_dt

    dC_dt = -args.u_inter * advection_dc_dx - sorption

    # Rate of material leaving the column (outlet flux)
    # Flux = C_out * u_inter * epsilon (per unit cross-sectional area)
    C_out = C.vals[-1]
    d_cumulative_out_dt = C_out * args.u_inter * args.epsilon

    return ColumnState(
        C=_spatial_tangent(C, dC_dt),
        n=_spatial_tangent(n, dn_dt),
        cumulative_out=d_cumulative_out_dt
    )

def make_adsorption_event(bed_utilization):
    """Event: stop adsorption once mean bed loading reaches ``bed_utilization``
    of the feed-equilibrium capacity. (Unused by the optimizer path, which uses
    the outlet-loss event below; kept as an alternative stop criterion.)"""
    def adsorption_finish_event(t, y: ColumnState, params: ColumnParameters, **kwargs):
        total_adsorbed = jnp.sum(y.n.vals)
        potential_adsorbed = params.isotherm(params.C_in) * len(y.n.vals)
        ratio = total_adsorbed / potential_adsorbed

        return ratio > bed_utilization
    return adsorption_finish_event

def make_desorption_event(potential_adsorbed, threshold=0.02):
    """Create desorption event that stops when loading drops below threshold of max potential."""
    def desorption_finish_event(t, y: ColumnState, params: ColumnParameters, **kwargs):
        # Mean loading has fallen to ``threshold`` of the max potential loading:
        # the bed is considered stripped and elution is done.
        total_adsorbed = jnp.sum(y.n.vals)
        ratio = total_adsorbed / potential_adsorbed

        return ratio < threshold
    return desorption_finish_event

def make_adsorption_loss_event(loss_fraction):
    """
    Create adsorption event that stops when a fraction of incoming material is lost from outlet.

    Args:
        loss_fraction: Fraction of total incoming material lost (e.g., 0.05 for 5%)

    Compares cumulative outlet to cumulative inlet:
        cumulative_out / (C_in * u_inter * epsilon * t) > loss_fraction
    """
    def adsorption_loss_event(t, y, params: ColumnParameters, **kwargs):
        # Cumulative material fed in so far = C_in * (superficial flux) * t.
        # Trip once the fraction that has escaped the outlet exceeds loss_fraction
        # (breakthrough). This is the stop criterion the optimizer actually uses.
        cumulative_in = params.C_in * params.u_inter * params.epsilon * t
        fraction_lost = jnp.where(cumulative_in > 0, y.cumulative_out / cumulative_in, 0.0)
        # jax.debug.print("{t}, {cumulative_in}, {fraction_lost}", t=t, cumulative_in=cumulative_in, fraction_lost=fraction_lost)
        return fraction_lost > loss_fraction

    return adsorption_loss_event


def get_finish_state(solution):
    """Extract the last valid saved sample from a diffrax solution.

    An event-terminated solve fills unused SaveAt slots with inf timestamps, so
    the number of finite ``ts`` entries locates the true final index.
    """
    valid = jnp.isfinite(solution.ts)
    idx = jnp.sum(valid)-1

    return (
        solution.ts[idx],
        solution.ys.C.vals[idx],
        solution.ys.n.vals[idx],
        solution.ys.cumulative_out[idx]
    )

def set_initial_adsorption_state(x0, x_final, n, desorption_threshold, column_params: ColumnParameters) -> ColumnState:
    """Create initial adsorption state with residual loading from previous desorption cycle.

    n(x, t=0) = r_des * n_eq(C_in) uniformly across the column.
    """
    n_eq = column_params.isotherm(column_params.C_in)
    initial_loading = desorption_threshold * n_eq
    return ColumnState(
        C=SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0.0),
        n=SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: initial_loading),
        cumulative_out=jnp.array(0.0)
    )

@eqx.filter_jit
def _run_model_jit(
    column_params: ColumnParameters,
    loss_fraction: Float[Array, ""],
    desorption_threshold: Float[Array, ""],
) -> float:
    """JIT-friendly model implementation with array-valued runtime inputs."""

    #Spatial discretization
    x0 = 0
    x_final = column_params.L
    n = 100
    y0_ads = set_initial_adsorption_state(x0, x_final, n, desorption_threshold, column_params)

    #Temporal discretization
    # t_final = inf: the phase runs until its terminating Event fires, not a
    # fixed end time. dt is only the initial step; the PID controller adapts it.
    t0 = 0
    t_final = jnp.inf
    dt = 1
    saveat_ads = diffrax.SaveAt(t1=True)  # Only final state
    saveat_des = diffrax.SaveAt(t0=True, t1=True)  # Initial + final for li_recovered

    #Tolerances
    rtol=1e-4
    atol=1e-6
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    # stepsize_controller = diffrax.ConstantStepSize()

    # ---- Phase 1: Adsorption (inlet C_in, stop at breakthrough loss) --------
    ads_event = diffrax.Event(make_adsorption_loss_event(loss_fraction))

    solution_ads = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(column_ode),
        solver=diffrax.Tsit5(),
        t0=t0,
        t1=t_final,
        dt0=dt,
        y0=y0_ads,
        args=column_params,
        saveat=saveat_ads,
        stepsize_controller=stepsize_controller,
        event=ads_event,
        max_steps=2**15,
        throw=False  # don't raise on failure; report via solver_ok instead
    )
    t_ads_final, C_ads_final, n_ads_final, cumulative_out_ads_final = get_finish_state(solution_ads)

    ads_ok = diffrax.is_okay(solution_ads.result)

    # Hand the end-of-adsorption profiles over as the desorption initial state.
    y0_des = ColumnState(
        C=SpatialDiscretisation(y0_ads.C.δx, C_ads_final),
        n=SpatialDiscretisation(y0_ads.n.δx, n_ads_final),
        cumulative_out=jnp.array(0.0)  # Reset for desorption phase
    )

    # Compute adsorption finishing condition before modifying column_params
    cumulative_in = column_params.C_in * column_params.u_inter * column_params.epsilon * t_ads_final
    fraction_lost = jnp.where(cumulative_in > 0, cumulative_out_ads_final / cumulative_in,0)

    # Calculate potential loading before modifying column_params
    potential_adsorbed = column_params.isotherm(column_params.C_in) * n
    des_event = diffrax.Event(make_desorption_event(potential_adsorbed, desorption_threshold))

    # ---- Phase 2: Desorption ------------------------------------------------
    # Switch the inlet BC to zero (strip fluid) and re-solve the SAME ode from
    # the loaded state; the bed drains until mean loading drops below threshold.
    column_params = column_params.replace(C_in=0)

    solution_des = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(column_ode),
        solver=diffrax.Tsit5(),
        t0 = t_ads_final,
        t1=t_final,
        dt0=dt,
        y0=y0_des,
        args=column_params,
        saveat=saveat_des,
        stepsize_controller=stepsize_controller,
        event=des_event,
        max_steps = 2**15,
        throw=False
    )

    t_des_final, C_des_final, n_des_final, cumulative_out_des_final = get_finish_state(solution=solution_des)

    des_ok = diffrax.is_okay(solution_des.result)
    solver_ok = ads_ok & des_ok  # both phases must have integrated cleanly

    return (t_ads_final, C_ads_final, n_ads_final), (t_des_final, C_des_final, n_des_final, cumulative_out_des_final), fraction_lost, solver_ok


def run_model(column_params: ColumnParameters, loss_fraction: float = 0.05, desorption_threshold: float = .02) -> float:
    """Run the column model while keeping frequently changed scalars as dynamic JAX inputs."""
    _log.debug(
        "run_model: u_inter=%.3e k_s=%.3e eps=%.3f C_in=%.3g L=%.4f rho_p=%.1f "
        "loss_fraction=%.3f desorption_threshold=%.3f",
        float(column_params.u_inter), float(column_params.k_s),
        float(column_params.epsilon), float(column_params.C_in),
        float(column_params.L), float(column_params.rho_p),
        float(loss_fraction), float(desorption_threshold),
    )
    result = _run_model_jit(
        column_params,
        jnp.asarray(loss_fraction),
        jnp.asarray(desorption_threshold),
    )
    (t_ads, _, _), (t_des, _, _, _), fraction_lost, solver_ok = result
    if not bool(solver_ok):
        _log.warning(
            "run_model: solver_ok=False (t_ads=%.3g, t_des=%.3g, fraction_lost=%.3g)",
            float(t_ads), float(t_des), float(fraction_lost),
        )
    else:
        _log.debug(
            "run_model done: t_ads=%.3g, t_des=%.3g, fraction_lost=%.3g",
            float(t_ads), float(t_des), float(fraction_lost),
        )
    return result

'''
def plot_breakthrough(t, C, column_params: ColumnParameters, curve: BreakthroughCurve = None):
    """Plot the breakthrough curve (C_out/C_in vs BV)."""
    import matplotlib.pyplot as plt

    # Convert time to Bed Volumes: BV = u_superficial * t / L = u_inter * epsilon * t / L
    BV = column_params.u_inter * column_params.epsilon * t / column_params.L

    # Get outlet concentration (last spatial node) at each time
    C_out = C[:, -1]
    C_out_over_C_in = C_out / column_params.C_in

    plt.figure(figsize=(10, 6))
    plt.plot(BV, C_out_over_C_in, label='Model')

    # Plot experimental data if provided
    if curve is not None:
        plt.scatter(curve.BV, curve.C_out_over_C_in, label='Experimental', marker='o')

    plt.xlabel('Bed Volumes (BV)')
    plt.ylabel('C_out / C_in')
    plt.title('Breakthrough Curve')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.show()

def plot_desorption(t, C, n):
    """Plot the desorption curve (C_out and sorbent loading vs time)."""
    import matplotlib.pyplot as plt

    # Get outlet concentration (last spatial node) at each time
    C_out = C[:, -1]
    # Average sorbent loading across the column
    n_avg = n.sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('C_out (mol/m³)', color='tab:blue')
    ax1.plot(t, C_out, label='C_out', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sorbent loading (mol/kg)', color='tab:orange')
    ax2.plot(t, n_avg, label='n_avg', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Desorption Curve')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True)
    plt.show()


if __name__ == "__main__":
    study = Study.from_json("LiteratureReview/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")

    column_length = study.column_experiments.column_properties.Length_si
    column_diameter = study.column_experiments.column_properties.Diameter_si

    # Get the experimental breakthrough curve
    exp_curve = study.column_experiments.breakthrough_curves.filter(flowrate=8)[0]

    flowrate = study.column_experiments.superficial_flowrate_si()[2]
    u_super = float(study.column_experiments.superficial_velocity_si()[2])
    print(u_super)
    params = study.to_column_parameter(column_length, 50, u_super)

    (t_ads, C_ads, n_ads), (t_des_final, C_des_final, n_des_final) = run_model(params)

    params = params.replace(C_in=50)

    plot_breakthrough(t_ads, C_ads, params, exp_curve)
    plot_desorption(t_des_final, C_des_final, n_des_final)

'''
