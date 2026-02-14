import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from collections.abc import Callable
import numpy as np

# Represents values on a uniform spatial grid with fixed spacing δx.
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

def langmuir_isotherm_non_dim(C_star, theta):
    return (1+theta)*C_star/(1+theta*C_star)

def filter_inf(arr):
    if arr.ndim == 2:
        return arr[~np.isinf(arr).any(axis=1)]
    return arr[~np.isinf(arr)]

class ColumnState(eqx.Module):
    C_star: SpatialDiscretisation
    n_star: SpatialDiscretisation


def _spatial_tangent(sd: SpatialDiscretisation, dvals_dt: Float[Array, "n"]) -> SpatialDiscretisation:
    """Return ODE tangent with fixed grid spacing (dδx/dt = 0)."""
    return SpatialDiscretisation(jnp.zeros_like(sd.δx), dvals_dt)


class NonDimNumbers(eqx.Module):
    Da: Float[Array, ""] # Damkoehler number knL/u
    Lambda: Float[Array, ""] # Sorbent to fluid capacity ratio
    theta: Float[Array, ""] # isotherm steepness
    epsilon: Float[Array, ""] # porosity in the column

    def __post_init__(self):
        object.__setattr__(self, "Da", jnp.asarray(self.Da))
        object.__setattr__(self, "Lambda", jnp.asarray(self.Lambda))
        object.__setattr__(self, "theta", jnp.asarray(self.theta))
        object.__setattr__(self, "epsilon", jnp.asarray(self.epsilon))

def column_ode(t, state: ColumnState, non_dim_nums: NonDimNumbers):
    C_star = state.C_star
    n_star = state.n_star

    Da= non_dim_nums.Da
    Lambda = non_dim_nums.Lambda
    theta = non_dim_nums.theta
    epsilon = non_dim_nums.epsilon

    n_eq_star = jax.vmap(lambda c: langmuir_isotherm_non_dim(c, theta))(C_star.vals)
    dn_dt = Da * (n_eq_star - n_star.vals) ** 2

    C_star_prev = jnp.roll(C_star.vals, 1)
    C_star_prev = C_star_prev.at[0].set(1) # inlet boundary condition

    advection_dc_dx = 1 / epsilon * (C_star.vals - C_star_prev) / C_star.δx
    sorption = Lambda * dn_dt

    dC_dt = -advection_dc_dx - sorption

    # jax.debug.breakpoint()
    return ColumnState(
        C_star=_spatial_tangent(C_star, dC_dt),
        n_star=_spatial_tangent(n_star, dn_dt)
    )

def finish_event(t, y: ColumnState, *args, **kwargs):

    return y.C_star.vals[-1] >= .05

@jax.jit
def run_wrapper(non_dim_nums: NonDimNumbers):
    """
        Docstring for run_wrapper

        :param non_dim_nums: Description
        :type non_dim_nums: NonDimNumbers
    """
    ode_term = diffrax.ODETerm(column_ode)
    
    # Spatial discretisation
    x0 = 0
    x_final = 1
    n = 20
    y0 = ColumnState( 
        C_star = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0),
        n_star = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0)
    )

    # Temporal discretisation
    t0 = 0
    t_final = jnp.inf
    dt = 0.0001
    saveat = diffrax.SaveAt(t0=True, steps=True)

    # Tolerances
    rtol = 1e-3
    atol = 1e-6
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
    )

    event = diffrax.Event(finish_event)

    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        ode_term,
        solver,
        t0,
        t_final,
        dt,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        args=non_dim_nums,
        max_steps=5_000_000,
    )
     # Stay in JAX - get the last valid index using the event
    n_star_vals = sol.ys.n_star.vals
    ts = sol.ts
    
    # Use jnp.where to handle potential inf values
    valid_mask = jnp.isfinite(ts)
    last_idx = jnp.sum(valid_mask) - 1
    
    n_star_final = n_star_vals[last_idx]
    t_f = ts[last_idx]
    U_b = jnp.trapezoid(n_star_final, jnp.linspace(0, 1, n))
    
    return t_f, U_b

    
# if __name__ == "__main__":

#     non_dim_numbers = NonDimNumbers(
#         Da=9, Lambda=9, epsilon=.35, theta=9
#     )

#     sol = run_wrapper(non_dim_nums=non_dim_numbers)
#     x0 = np.asarray(sol.ys.C_star.vals)
#     n = np.asarray(sol.ys.n_star.vals)

#     x = filter_inf(x0)
#     n = filter_inf(n)
#     bed_util = np.trapezoid(n[-1, :], np.linspace(0,1,10))
#     print(bed_util)
#     t = filter_inf(np.asarray(sol.ts))
#     print("Done")