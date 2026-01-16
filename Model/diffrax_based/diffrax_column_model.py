import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.Dataclasses import ColumnParameters, Study, BreakthroughCurve

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

@jax.jit(static_argnums=2)
def column_ode(t, state: ColumnState, args: ColumnParameters):
    # jax.debug.print("args={a.C_in}", a=args)
    C = state.C
    n = state.n

    n_star = args.isotherm(C.vals)
    # jax.debug.print("n_star={n}", n=n_star)
    dn_dt = args.k_s * (n_star - n.vals)**2

    C_prev = jnp.roll(C.vals, shift=1)
    C_prev = C_prev.at[0].set(args.C_in)

    advection_dc_dx = (C.vals - C_prev) / C.δx
    sorption = (1 - args.epsilon) / args.epsilon * args.rho_p * dn_dt

    dC_dt = -args.u_inter * advection_dc_dx - sorption

    return ColumnState(
        C=SpatialDiscretisation(C.x0, C.x_final, dC_dt),
        n=SpatialDiscretisation(n.x0, n.x_final, dn_dt)
    )

def finish_event(t, y: ColumnState, params: ColumnParameters, **kwargs):
    total_adsorbed = jnp.sum(y.n.vals)
    potential_adsorbed = params.isotherm(params.C_in) * len(y.n.vals)
    ratio = total_adsorbed / potential_adsorbed

    # Print every 100 seconds
    lax.cond(
        (t % 100) < 0.1,
        lambda: jax.debug.print("t={t}, adsorption ratio={ratio}", t=t, ratio=ratio),
        lambda: None
    )
    return ratio > .85


def run_model(column_params: ColumnParameters) ->  float:
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
    dt = .1
    saveat = diffrax.SaveAt(t0=True, steps=True)

    #Tolerances
    rtol=1e-6
    atol=1e-6
    # stepsize_controller = diffrax.PIDController()
    stepsize_controller = diffrax.ConstantStepSize()

    event = diffrax.Event(finish_event)

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(column_ode),
        solver=diffrax.Euler(),
        t0=t0,
        t1=t_final,
        dt0=dt,
        y0=y0,
        args=column_params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=int(1e6),
    )

    return solution

def plot_breakthrough(solution, column_params: ColumnParameters, curve: BreakthroughCurve = None):
    """Plot the breakthrough curve (C_out/C_in vs BV)."""
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.array(solution.ts)
    # Filter out infinite time values
    valid_mask = np.isfinite(ts)
    ts = ts[valid_mask]

    # Convert time to Bed Volumes: BV = u_superficial * t / L = u_inter * epsilon * t / L
    BV = column_params.u_inter * column_params.epsilon * ts / column_params.L

    # Get outlet concentration (last spatial node) at each time - vectorized
    C_out = np.array(solution.ys.C.vals[valid_mask, -1])
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

if __name__ == "__main__":
    study = Study.from_json("LiteratureReview/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")

    column_length = study.column_experiments.column_properties.Length_si
    column_diameter = study.column_experiments.column_properties.Diameter_si

    # Get the experimental breakthrough curve
    exp_curve = study.column_experiments.breakthrough_curves.filter(flowrate=8)[0]

    flowrate = study.column_experiments.superficial_flowrate_si()[2]
    params = study.to_column_parameter(column_length, column_diameter, 50, flowrate)

    solution = run_model(params)

    plot_breakthrough(solution, params, exp_curve)


