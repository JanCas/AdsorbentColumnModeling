"""Diffrax plug-flow column for the multi-ion ALLDH isotherm.

Fluid concentrations are [mol/m3 solution] and solid loading is [mol/kg
sorbent]. Only LiCl sorbs; NaCl and MgCl2 are transported backgrounds that
affect equilibrium through their Pitzer activities. The activity calculation
converts concentration to molality at that boundary only. Uptake uses a
sign-preserving pseudo-second-order rate.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from utils.StreamData import Composition

from .isotherm import ALLDHIsotherm


class AqueousConcentration(eqx.Module):
    """LiCl/NaCl/MgCl2 concentrations [mol/m3 solution]."""

    c_LiCl: jax.Array
    c_NaCl: jax.Array
    c_MgCl2: jax.Array

    def __post_init__(self):
        for name in ("c_LiCl", "c_NaCl", "c_MgCl2"):
            object.__setattr__(
                self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64)
            )

    def molality(self, water_mass_concentration):
        """Convert to Pitzer molalities using kg water/m3 solution."""
        return Composition(
            self.c_LiCl / water_mass_concentration,
            self.c_NaCl / water_mass_concentration,
            self.c_MgCl2 / water_mass_concentration,
        )


class ColumnState(eqx.Module):
    """Cell-average fields on a uniform finite-volume grid."""

    c_LiCl: jax.Array
    c_NaCl: jax.Array
    c_MgCl2: jax.Array
    q: jax.Array
    cumulative_Li_out: jax.Array = eqx.field(
        default=0.0,
        converter=lambda x: jnp.asarray(x, dtype=jnp.float64),
    )


class CycleResult(eqx.Module):
    """Full phase trajectories and dimensional cycle metrics."""

    adsorption_ts: jax.Array
    adsorption: ColumnState
    desorption_ts: jax.Array
    desorption: ColumnState
    adsorption_duration: jax.Array
    desorption_duration: jax.Array
    bed_utilization: jax.Array
    li_recovered: jax.Array
    productivity: jax.Array


class ColumnParams(eqx.Module):
    """Physical parameters for one constant-inlet flow phase.

    ``k2`` is the PSO rate constant [kg sorbent/(mol s)].
    """

    L: jax.Array
    eps: jax.Array
    rho_p: jax.Array
    water_mass_concentration: jax.Array
    u_s: jax.Array
    k2: jax.Array
    inlet: AqueousConcentration
    isotherm: ALLDHIsotherm
    N: int = eqx.field(static=True)

    def __post_init__(self):
        for name in (
            "L",
            "eps",
            "rho_p",
            "water_mass_concentration",
            "u_s",
            "k2",
        ):
            object.__setattr__(
                self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64)
            )

    @property
    def dx(self):
        return self.L / self.N


def initial_state(
    params: ColumnParams,
    concentration: AqueousConcentration | None = None,
    q0=0.0,
) -> ColumnState:
    """Return a spatially uniform initial column state."""
    if concentration is None:
        concentration = AqueousConcentration(0.0, 0.0, 0.0)
    return ColumnState(
        c_LiCl=jnp.full(params.N, concentration.c_LiCl),
        c_NaCl=jnp.full(params.N, concentration.c_NaCl),
        c_MgCl2=jnp.full(params.N, concentration.c_MgCl2),
        q=jnp.full(params.N, q0, dtype=jnp.float64),
        cumulative_Li_out=0.0,
    )


def column_rhs(t, state: ColumnState, params: ColumnParams) -> ColumnState:
    """Method-of-lines balance with upwind advection and PSO uptake."""
    del t
    concentration = AqueousConcentration(
        jnp.maximum(state.c_LiCl, 0.0),
        jnp.maximum(state.c_NaCl, 0.0),
        jnp.maximum(state.c_MgCl2, 0.0),
    )
    composition = concentration.molality(params.water_mass_concentration)
    q_eq = jax.vmap(params.isotherm)(composition)
    driving_force = q_eq - state.q
    dq = params.k2 * driving_force * jnp.abs(driving_force)

    fluid = jnp.stack((state.c_LiCl, state.c_NaCl, state.c_MgCl2))
    inlet = jnp.stack(
        (params.inlet.c_LiCl, params.inlet.c_NaCl, params.inlet.c_MgCl2)
    )
    upstream = jnp.concatenate((inlet[:, None], fluid[:, :-1]), axis=1)
    dc = -(params.u_s / params.eps) * (fluid - upstream) / params.dx

    coupling = (1.0 - params.eps) * params.rho_p / params.eps
    dc = dc.at[0].add(-coupling * dq)
    return ColumnState(
        dc[0],
        dc[1],
        dc[2],
        dq,
        params.u_s * state.c_LiCl[-1],
    )


def make_adsorption_event(outlet_fraction: float):
    """Stop when outlet LiCl reaches a fraction of feed LiCl."""
    if not 0.0 < outlet_fraction < 1.0:
        raise ValueError("outlet_fraction must lie between 0 and 1")

    def event(t, y: ColumnState, args: ColumnParams, **kwargs):
        del t, kwargs
        return y.c_LiCl[-1] - outlet_fraction * args.inlet.c_LiCl

    return event


def make_desorption_event(loading_fraction: float):
    """Stop when mean loading falls below a fraction of ``q_max``."""
    if not 0.0 < loading_fraction < 1.0:
        raise ValueError("loading_fraction must lie between 0 and 1")

    def event(t, y: ColumnState, args: ColumnParams, **kwargs):
        del t, kwargs
        return jnp.mean(y.q) - loading_fraction * args.isotherm.q_max

    return event


def _validate(params, t_final, y0, save_ts):
    if params.N < 2:
        raise ValueError("N must be at least 2")
    positive = ("L", "eps", "rho_p", "water_mass_concentration", "u_s")
    if any(
        not np.isfinite(float(getattr(params, x)))
        or float(getattr(params, x)) <= 0
        for x in positive
    ):
        raise ValueError(f"{', '.join(positive)} must be finite and positive")
    if not 0.0 < float(params.eps) < 1.0:
        raise ValueError("eps must lie between 0 and 1")
    if not np.isfinite(float(params.k2)) or float(params.k2) < 0.0:
        raise ValueError("k2 must be finite and nonnegative")
    isotherm_values = np.asarray(
        [params.isotherm.q_max, params.isotherm.log_K_i, params.isotherm.n_h2o],
        dtype=float,
    )
    if np.any(~np.isfinite(isotherm_values)) or isotherm_values[0] <= 0.0:
        raise ValueError("isotherm parameters must be finite with q_max positive")
    inlet = np.asarray(
        [params.inlet.c_LiCl, params.inlet.c_NaCl, params.inlet.c_MgCl2],
        dtype=float,
    )
    if np.any(~np.isfinite(inlet)) or np.any(inlet < 0.0):
        raise ValueError("inlet concentrations must be finite and nonnegative")
    if not np.isfinite(t_final) or t_final <= 0.0:
        raise ValueError("t_final must be finite and positive")
    profiles = (y0.c_LiCl, y0.c_NaCl, y0.c_MgCl2, y0.q)
    for profile in profiles:
        values = np.asarray(profile)
        if values.shape != (params.N,) or np.any(~np.isfinite(values)):
            raise ValueError(
                f"every initial-state field must be finite with shape ({params.N},)"
            )
    cumulative_out = np.asarray(y0.cumulative_Li_out)
    if (
        cumulative_out.shape != ()
        or not np.isfinite(cumulative_out)
        or cumulative_out < 0.0
    ):
        raise ValueError(
            "initial cumulative_Li_out must be a nonnegative finite scalar"
        )
    if any(np.any(np.asarray(x) < 0.0) for x in profiles):
        raise ValueError("initial concentrations and loading must be nonnegative")
    if np.any(np.asarray(y0.q) > isotherm_values[0]):
        raise ValueError("initial loading cannot exceed q_max")
    ts = np.asarray(save_ts)
    if ts.ndim != 1 or len(ts) == 0 or np.any(~np.isfinite(ts)):
        raise ValueError("save_ts must be a nonempty finite one-dimensional array")
    if ts[0] < 0.0 or ts[-1] > t_final or np.any(np.diff(ts) <= 0.0):
        raise ValueError("save_ts must increase strictly within [0, t_final]")


_SOLVER = diffrax.Tsit5()
_CONTROLLER = diffrax.PIDController(rtol=1e-5, atol=1e-8)
_ROOT_FINDER = optx.Newton(1e-6, 1e-8, optx.rms_norm)
_TERM = diffrax.ODETerm(column_rhs)


def _solve(params, t_final, y0, saveat, event=None, event_direction=None):
    return diffrax.diffeqsolve(
        _TERM,
        _SOLVER,
        t0=0.0,
        t1=t_final,
        dt0=None,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=_CONTROLLER,
        event=(
            diffrax.Event(event, _ROOT_FINDER, direction=event_direction)
            if event is not None
            else None
        ),
        max_steps=100_000,
    )


def _finite_trajectory(solution):
    n = int(np.count_nonzero(np.isfinite(np.asarray(solution.ts))))
    return (
        solution.ts[:n],
        jax.tree_util.tree_map(lambda x: x[:n], solution.ys),
    )


def simulate_column(
    params: ColumnParams,
    t_final: float,
    *,
    y0: ColumnState | None = None,
    save_ts=None,
    n_save: int = 101,
) -> diffrax.Solution:
    """Integrate one constant-inlet phase and return the Diffrax solution."""
    if save_ts is None:
        if n_save < 2:
            raise ValueError("n_save must be at least 2")
        save_ts = jnp.linspace(0.0, t_final, n_save)
    else:
        save_ts = jnp.asarray(save_ts, dtype=jnp.float64)
    y0 = initial_state(params) if y0 is None else y0
    _validate(params, t_final, y0, save_ts)

    return _solve(
        params,
        t_final,
        y0,
        diffrax.SaveAt(ts=save_ts),
    )


def simulate_cycle(
    adsorption_params: ColumnParams,
    desorption_inlet: AqueousConcentration,
    *,
    adsorption_t_max: float,
    desorption_t_max: float,
    adsorption_outlet_fraction: float = 0.05,
    desorption_loading_fraction: float = 0.02,
    y0: ColumnState | None = None,
    n_save: int = 501,
) -> CycleResult:
    """Simulate adsorption to breakthrough, then desorption to residual loading.

    ``li_recovered`` is the desorption outlet integral [mol/m2], and
    ``productivity`` is recovery per cycle time [mol/m2/s].
    """
    if n_save < 2:
        raise ValueError("n_save must be at least 2")
    if float(adsorption_params.inlet.c_LiCl) <= 0.0:
        raise ValueError("adsorption inlet LiCl must be positive")
    ads_event = make_adsorption_event(adsorption_outlet_fraction)
    des_event = make_desorption_event(desorption_loading_fraction)
    if y0 is None:
        y0 = initial_state(
            adsorption_params,
            AqueousConcentration(
                0.0,
                adsorption_params.inlet.c_NaCl,
                adsorption_params.inlet.c_MgCl2,
            ),
            q0=desorption_loading_fraction * adsorption_params.isotherm.q_max,
        )
    else:
        y0 = eqx.tree_at(
            lambda y: y.cumulative_Li_out, y0, jnp.asarray(0.0)
        )
    ads_save_ts = jnp.linspace(0.0, adsorption_t_max, n_save)
    _validate(adsorption_params, adsorption_t_max, y0, ads_save_ts)
    breakthrough = (
        adsorption_outlet_fraction * float(adsorption_params.inlet.c_LiCl)
    )
    if float(y0.c_LiCl[-1]) >= breakthrough:
        raise ValueError("initial outlet already satisfies the adsorption event")
    ads_sol = _solve(
        adsorption_params,
        adsorption_t_max,
        y0,
        diffrax.SaveAt(ts=ads_save_ts, t1=True),
        ads_event,
        True,
    )
    if not bool(diffrax.is_okay(ads_sol.result)):
        raise RuntimeError(f"adsorption solve failed: {ads_sol.result}")
    adsorption_ts, adsorption = _finite_trajectory(ads_sol)
    if float(adsorption_ts[-1]) >= adsorption_t_max:
        raise RuntimeError("adsorption did not reach its outlet event before t_max")

    ads_final = jax.tree_util.tree_map(lambda x: x[-1], adsorption)
    desorption_params = eqx.tree_at(
        lambda p: p.inlet, adsorption_params, desorption_inlet
    )
    des_y0 = eqx.tree_at(
        lambda y: y.cumulative_Li_out, ads_final, jnp.asarray(0.0)
    )
    des_save_ts = jnp.linspace(0.0, desorption_t_max, n_save)
    _validate(desorption_params, desorption_t_max, des_y0, des_save_ts)
    if float(jnp.mean(ads_final.q)) <= (
        desorption_loading_fraction * float(adsorption_params.isotherm.q_max)
    ):
        raise RuntimeError("adsorption ended below the desorption loading threshold")
    des_sol = _solve(
        desorption_params,
        desorption_t_max,
        des_y0,
        diffrax.SaveAt(ts=des_save_ts, t1=True),
        des_event,
        False,
    )
    if not bool(diffrax.is_okay(des_sol.result)):
        raise RuntimeError(f"desorption solve failed: {des_sol.result}")
    desorption_ts, desorption = _finite_trajectory(des_sol)
    if float(desorption_ts[-1]) >= desorption_t_max:
        raise RuntimeError("desorption did not reach its loading event before t_max")
    adsorption_duration = adsorption_ts[-1]
    desorption_duration = desorption_ts[-1]
    bed_utilization = jnp.mean(adsorption.q[-1]) / adsorption_params.isotherm.q_max
    li_recovered = desorption.cumulative_Li_out[-1]
    productivity = li_recovered / (adsorption_duration + desorption_duration)
    return CycleResult(
        adsorption_ts,
        adsorption,
        desorption_ts,
        desorption,
        adsorption_duration,
        desorption_duration,
        bed_utilization,
        li_recovered,
        productivity,
    )
