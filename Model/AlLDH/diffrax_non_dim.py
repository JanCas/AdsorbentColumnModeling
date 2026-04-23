import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from collections.abc import Callable

N_SPATIAL = 20  # number of spatial grid points


# ---------------------------------------------------------------------------
# Spatial discretisation (unchanged)
# ---------------------------------------------------------------------------
class SpatialDiscretisation(eqx.Module):
    δx: Float[Array, ""]
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        δx = jnp.asarray((x_final - x0) / (n - 1), dtype=vals.dtype)
        return cls(δx, vals)

    def binop(self, other, fn):
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


# ---------------------------------------------------------------------------
# Isotherm
# ---------------------------------------------------------------------------
def langmuir_isotherm_non_dim(C_star, theta):
    return (1 + theta) * C_star / (1 + theta * C_star)


# ---------------------------------------------------------------------------
# Column state
# ---------------------------------------------------------------------------
class ColumnState(eqx.Module):
    C_star: SpatialDiscretisation
    n_star: SpatialDiscretisation
    R_outlet: Float[Array, ""]        # ∫ C*(ζ=1) dτ


def _spatial_tangent(sd: SpatialDiscretisation, dvals_dt: Float[Array, "n"]):
    return SpatialDiscretisation(jnp.zeros_like(sd.δx), dvals_dt)


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
# ODE args passed to diffrax (includes inlet BC)
# ---------------------------------------------------------------------------
class ColumnArgs(eqx.Module):
    Da: Float[Array, ""]
    Lambda: Float[Array, ""]
    theta: Float[Array, ""]
    epsilon: Float[Array, ""]
    c_inlet: Float[Array, ""]   # 1.0 for adsorption, 0.0 for desorption

    @classmethod
    def from_non_dim(cls, nd: NonDimNumbers, c_inlet: float):
        return cls(
            Da=nd.Da, Lambda=nd.Lambda, theta=nd.theta,
            epsilon=nd.epsilon, c_inlet=jnp.asarray(c_inlet),
        )


# ---------------------------------------------------------------------------
# ODE right-hand side  (standard LDF kinetics)
# ---------------------------------------------------------------------------
def column_ode(t, state: ColumnState, args: ColumnArgs):
    C = state.C_star
    n = state.n_star

    n_eq = langmuir_isotherm_non_dim(C.vals, args.theta)

    dn_dt = args.Da * (n_eq - n.vals)

    C_prev = jnp.roll(C.vals, 1).at[0].set(args.c_inlet)
    advection = (1.0 / args.epsilon) * (C.vals - C_prev) / C.δx
    dC_dt = -advection - args.Lambda * dn_dt
    
    return ColumnState(
        C_star=_spatial_tangent(C, dC_dt),
        n_star=_spatial_tangent(n, dn_dt),
        R_outlet=jnp.asarray(C.vals[-1]),           # dR_outlet/dτ = C*(ζ=1, τ)
    )


# ---------------------------------------------------------------------------
# Event functions (closure pattern — threshold captured at creation time)
# ---------------------------------------------------------------------------
def make_ads_event(c_thresh):
    """Stop adsorption when outlet concentration reaches *c_thresh*."""
    def _event(t, y: ColumnState, args, **kwargs):
        return y.C_star.vals[-1] >= c_thresh
    return _event


def make_des_event(c_thresh):
    """Stop desorption when mean bed loading drops below *c_thresh*."""
    def _event(t, y: ColumnState, args, **kwargs):
        return jnp.mean(y.n_star.vals) <= c_thresh
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
        C_star=SpatialDiscretisation.discretise_fn(0.0, 1.0, N_SPATIAL, lambda x: 0.0),
        n_star=SpatialDiscretisation.discretise_fn(0.0, 1.0, N_SPATIAL, lambda x: c_threshold_des),
        R_outlet=jnp.array(0.0)
    )



# ---------------------------------------------------------------------------
# Nondimensional SEC*
#
#   SEC* = 150 (1-ε)² / ε³  ·  (τ_ads + Ψ τ_des) / (η_p R_des)
#
#   Dimensional recovery:  SEC = SEC* · μ u_s N / (c₀ d_p)
# ---------------------------------------------------------------------------
def compute_sec_star(epsilon, eta_p, tau_ads, tau_des, R_des, Psi=1.0):
    hydraulic = 150.0 * (1.0 - epsilon) ** 2 / (epsilon ** 3)
    return hydraulic * (tau_ads + tau_des) / (R_des)


# ---------------------------------------------------------------------------
# Main cycle solver
# ---------------------------------------------------------------------------
@eqx.filter_jit
def run_cycle(
    non_dim: NonDimNumbers,
    c_thresh_ads: Float[Array, ""],
    c_thresh_des: Float[Array, ""],
    eta_p: Float[Array, ""],
    Psi: Float[Array, ""],
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
    eta_p : float
        Pump efficiency.
    Psi : float
        Viscosity-velocity ratio  mu_des u_des / (mu_ads u_ads).  Default 1.

    Returns
    -------
    tau_ads   - dimensionless adsorption time at cutoff
    tau_des   - dimensionless desorption time at cutoff
    U_b       - bed utilisation at end of adsorption
    eta_li    - average Li removal efficiency during adsorption
    R_outlet_des - dimensionless Li collected at outlet during desorption ∫C*(ζ=1)dτ
    sec_star     - nondimensional specific energy consumption
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
    U_b = jnp.trapezoid(ads_final.n_star.vals, zeta)[0]
    # ---- Phase 2: Desorption ------------------------------------------------
    des_args = ColumnArgs.from_non_dim(non_dim, c_inlet=0.0)
    des_y0 = ColumnState(
        C_star=SpatialDiscretisation(ads_final.C_star.δx, ads_final.C_star.vals[-1]),
        n_star=SpatialDiscretisation(ads_final.n_star.δx, ads_final.n_star.vals[-1]),
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

    # Guard against R_outlet_des ~ 0 (would blow up SEC*)
    R_outlet_des = jnp.maximum(R_outlet_des, 1e-12)

    # ---- Full-cycle metrics ---------------------------------------------------
    sec_star = compute_sec_star(non_dim.epsilon, eta_p, tau_ads, tau_des, R_outlet_des, Psi)
    productivity = R_outlet_des / (tau_ads + tau_des)

    return tau_ads, tau_des, U_b, R_outlet_des, sec_star, productivity

