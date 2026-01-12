import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float

from collections.abc import Callable

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

class ColumnState(eqx.Module):
    C: SpatialDiscretisation
    n: SpatialDiscretisation

class ColumnParameters(eqx.Module):
    u_inter: float
    k_s: float
    epsilon: float
    q_max: float
    b: float
    C_in: float
    L: float
    rho_p: float

def langmuir_isotherm(C, params: ColumnParameters):
    return params.q_max * params.b * C / (1 + params.b * C)

def column_ode(t, state: ColumnState, args: ColumnParameters):
    C = state.C
    q = state.q

    q_star = jax.vmap(lambda c: langmuir_isotherm(c, args))(C.vals)
    dq_dt = args.k_s * (q_star - q.vals)

    C_prev = jnp.roll(C.vals, shift=1)
    C_prev = C_prev.at[0].set(args.C_in)

    advection_dc_dx = (C.vals - C_prev) / C.δx
    sorption = (1 - args.epsilon) / args.epsilon * args.rho_s * dq_dt

    dC_dt = -args.u_inter * advection_dc_dx - sorption

    return ColumnState(
        C=SpatialDiscretisation(C.x0, C.x_final, dC_dt),
        q=SpatialDiscretisation(q.x0, q.x_final, dq_dt)
    )

column_params = ColumnParameters(
    u_inter=.0003,
    k_s=0.00000666,
    epsilon=0.35,
    q_max=.69,
    b=6.25,
    C_in=50,
    L=.6,
    rho_s=680
)

def finish_event(t,y: ColumnState, *args, **kwargs):

    total_adsorbed = y.n
    


def run_wrapper(column_params: ColumnParameters) ->  float:
    """
    Runs the diffrax model for the column sorption and returns the SEC
     
    :param column_params: Description
    :type column_params: ColumnParameters
    :return: SEC
    :rtype: float
    """

    #Spatial discretization
    x0 = 0
    x_final = column_params.L
    n = 100
    y0 = ColumnState(
        C = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0),
        n = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0)
    )

    #Temporal discretization
    t0 = 0
    t_final = jnp.inf
    dt = .00001
    saveat = diffrax.SaveAt(t0=True, steps=True)

    #Tolerances
    rtol=1e-6
    atol=1e-6
    stepsize_controller = diffrax.PIDController()

    event = diffrax.Event()