import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from collections.abc import Callable
import numpy as np

# Represents the interval [x0, x_final] discretised into n equally-spaced points.
class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

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

class ColumnState(eqx.Module):
    C_star: SpatialDiscretisation
    n_star: SpatialDiscretisation

class NonDimNumbers(eqx.Module):
    Da: float # Damkoehler number knL/u
    Lambda: float # Sorbent to fluid capacity ratio
    theta: float # isotherm steepness
    epsilon: float # porosity in the column

def column_ode(t, state: ColumnState, non_dim_nums: NonDimNumbers):
    C_star = state.C_star
    n_star = state.n_star

    Da= non_dim_nums.Da
    Lambda = non_dim_nums.Lambda
    theta = non_dim_nums.theta
    epsilon = non_dim_nums.epsilon

    n_eq_star = jax.vmap(lambda c: langmuir_isotherm_non_dim(c, theta))(C_star.vals)
    dn_dt = Da * (n_eq_star - n_star.vals)

    C_star_prev = jnp.roll(C_star.vals, 1)
    C_star_prev = C_star_prev.at[0].set(1) # inlet boundary condition

    advection_dc_dx = 1 / epsilon * (C_star.vals - C_star_prev) / C_star.δx
    sorption = Lambda * dn_dt

    dC_dt = -advection_dc_dx - sorption

    # jax.debug.breakpoint()
    return ColumnState(
        C_star=SpatialDiscretisation(C_star.x0, C_star.x_final, dC_dt),
        n_star=SpatialDiscretisation(n_star.x0, n_star.x_final, dn_dt)
    )

def finish_event(t, y: ColumnState, *args, **kwargs):

    return y.C_star.vals[-1] >= .05

def run_wrapper(non_dim_nums: NonDimNumbers):
    ode_term = diffrax.ODETerm(column_ode)
    
    # Spatial discretisation
    x0 = 0
    x_final = 1
    n = 10
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
    rtol = 1e-10
    atol = 1e-10
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
    return sol

def filter_inf(arr):
    if arr.ndim == 2:
        return arr[~np.isinf(arr).any(axis=1)]
    return arr[~np.isinf(arr)]
    

if __name__ == "__main__":

    non_dim_numbers = NonDimNumbers(
        Da=9, Lambda=9, epsilon=.35, theta=9
    )

    sol = run_wrapper(non_dim_nums=non_dim_numbers)
    x0 = np.asarray(sol.ys.C_star.vals)
    n = np.asarray(sol.ys.n_star.vals)

    x = filter_inf(x0)
    n = filter_inf(n)
    bed_util = np.trapezoid(n[-1, :], np.linspace(0,1,10))
    print(bed_util)
    t = filter_inf(np.asarray(sol.ts))
    print("Done")