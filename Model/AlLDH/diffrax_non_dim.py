import logging

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

_log = logging.getLogger(__name__)
# `run_cycle` below is @eqx.filter_jit; do NOT put _log calls inside its body.
# Callers (multi_optim.py, Sensitivity/sobol_driver.py) handle runtime logging.

N_SPATIAL = 20  # number of spatial grid points


# ---------------------------------------------------------------------------
# Isotherm
# ---------------------------------------------------------------------------
def langmuir_isotherm_non_dim(C_star, theta):
    return (1 + theta) * C_star / (1 + theta * C_star)


# ---------------------------------------------------------------------------
# Column state
# ---------------------------------------------------------------------------
class ColumnState(eqx.Module):
    C_star: Float[Array, "n"]
    n_star: Float[Array, "n"]
    R_outlet: Float[Array, ""]        # ∫ C*(ζ=1) dτ


# ---------------------------------------------------------------------------
# Dimensionless numbers (physical parameters only)
# ---------------------------------------------------------------------------
class NonDimNumbers(eqx.Module):
    Da: Float[Array, ""]       # Damkoehler number  kn L / u
    Lambda: Float[Array, ""]   # sorbent-to-fluid capacity ratio
    theta: Float[Array, ""]    # isotherm steepness
    epsilon: Float[Array, ""]  # bed porosity

    def __post_init__(self):
        object.__setattr__(self, "Da", jnp.asarray(self.Da))
        object.__setattr__(self, "Lambda", jnp.asarray(self.Lambda))
        object.__setattr__(self, "theta", jnp.asarray(self.theta))
        object.__setattr__(self, "epsilon", jnp.asarray(self.epsilon))


# ---------------------------------------------------------------------------
# ODE args passed to diffrax (includes inlet BC and grid spacing)
# ---------------------------------------------------------------------------
class ColumnArgs(eqx.Module):
    Da: Float[Array, ""]
    Lambda: Float[Array, ""]
    theta: Float[Array, ""]
    epsilon: Float[Array, ""]
    c_inlet: Float[Array, ""]   # 1.0 for adsorption, 0.0 for desorption
    dzeta: Float[Array, ""]     # spatial grid spacing

    @classmethod
    def from_non_dim(cls, nd: NonDimNumbers, c_inlet: float):
        return cls(
            Da=nd.Da, Lambda=nd.Lambda, theta=nd.theta,
            epsilon=nd.epsilon, c_inlet=jnp.asarray(c_inlet),
            dzeta=jnp.asarray(1.0 / (N_SPATIAL - 1)),
        )


# ---------------------------------------------------------------------------
# ODE right-hand side  (standard LDF kinetics)
# ---------------------------------------------------------------------------
def column_ode(t, state: ColumnState, args: ColumnArgs):
    C = state.C_star
    n = state.n_star

    n_eq = langmuir_isotherm_non_dim(C, args.theta)

    dn_dt = args.Da * (n_eq - n)

    C_prev = jnp.roll(C, 1).at[0].set(args.c_inlet)
    advection = (1.0 / args.epsilon) * (C - C_prev) / args.dzeta
    dC_dt = -advection - args.Lambda * dn_dt

    return ColumnState(
        C_star=dC_dt,
        n_star=dn_dt,
        R_outlet=jnp.asarray(C[-1]),           # dR_outlet/dτ = C*(ζ=1, τ)
    )


# ---------------------------------------------------------------------------
# Event functions (closure pattern — threshold captured at creation time)
# ---------------------------------------------------------------------------
def make_ads_event(c_thresh):
    """Stop adsorption when outlet concentration reaches *c_thresh*."""
    def _event(t, y: ColumnState, args, **kwargs):
        return y.C_star[-1] >= c_thresh
    return _event


def make_des_event(c_thresh):
    """Stop desorption when mean bed loading drops below *c_thresh*."""
    def _event(t, y: ColumnState, args, **kwargs):
        return jnp.mean(y.n_star) <= c_thresh
    return _event


# ---------------------------------------------------------------------------
# Shared solver settings
# ---------------------------------------------------------------------------
_SOLVER = diffrax.Tsit5()
_RTOL = 1e-3
_ATOL = 1e-6
_DT0 = 1e-4
_MAX_STEPS = 5_000_000


_CONTROLLER = diffrax.PIDController(
    pcoeff=0.3, icoeff=0.4, rtol=_RTOL, atol=_ATOL
)


def ads_state(c_threshold_des):
    return ColumnState(
        C_star=jnp.zeros(N_SPATIAL),
        n_star=jnp.full(N_SPATIAL, c_threshold_des),
        R_outlet=jnp.array(0.0)
    )



# ---------------------------------------------------------------------------
# Main cycle solver
# ---------------------------------------------------------------------------
@eqx.filter_jit
def run_cycle(
    non_dim: NonDimNumbers,
    c_thresh_ads: Float[Array, ""],
    c_thresh_des: Float[Array, ""],
):
    """
    Run a full adsorption-desorption cycle and return performance metrics.

    Parameters
    ----------
    non_dim : NonDimNumbers
        Physical dimensionless groups (Da, Lambda, theta, epsilon).
    c_thresh_ads : float
        Outlet concentration threshold to end adsorption  (0 < . < 1).
    c_thresh_des : float
        Outlet concentration threshold to end desorption   (0 < . < 1).
        Desorption stops when the elution peak has passed and the outlet
        drops below this value.
    Returns
    -------
    tau_ads   - dimensionless adsorption time at cutoff
    tau_des   - dimensionless desorption time at cutoff
    U_b       - bed utilisation at end of adsorption
    R_outlet_des - dimensionless Li collected at outlet during desorption ∫C*(ζ=1)dτ
    productivity - R_outlet_des / (tau_ads + tau_des), Li recovered per unit cycle time
    """
    zeta = jnp.linspace(0.0, 1.0, N_SPATIAL)
    # ---- Phase 1: Adsorption ------------------------------------------------
    ads_args = ColumnArgs.from_non_dim(non_dim, c_inlet=1.0)
    ads_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(column_ode),
        _SOLVER,
        t0=0.0,
        t1=jnp.inf,
        dt0=_DT0,
        y0=ads_state(c_thresh_des),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=_CONTROLLER,
        event=diffrax.Event(make_ads_event(c_thresh_ads)),
        args=ads_args,
        max_steps=_MAX_STEPS,
    )
    ads_final = ads_sol.ys
    tau_ads = ads_sol.ts[0]


    # Bed utilisation
    U_b = jnp.trapezoid(ads_final.n_star, zeta)[0]
    # ---- Phase 2: Desorption ------------------------------------------------
    des_args = ColumnArgs.from_non_dim(non_dim, c_inlet=0.0)
    des_y0 = ColumnState(
        C_star=ads_final.C_star[-1],
        n_star=ads_final.n_star[-1],
        R_outlet=jnp.array(0.0),       # reset accumulator for desorption
    )

    des_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(column_ode),
        _SOLVER,
        t0=0.0,
        t1=jnp.inf,
        dt0=_DT0,
        y0=des_y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=_CONTROLLER,
        event=diffrax.Event(make_des_event(c_thresh_des)),
        args=des_args,
        max_steps=_MAX_STEPS,
    )
    des_final = des_sol.ys
    tau_des = des_sol.ts[0]

    # Li collected at the outlet during desorption: ∫₀^τ_des C*(ζ=1, τ) dτ
    R_outlet_des = des_final.R_outlet[0]

    R_outlet_des = jnp.maximum(R_outlet_des, 1e-12)

    productivity = R_outlet_des / (tau_ads + tau_des)

    return tau_ads, tau_des, U_b, R_outlet_des, productivity
