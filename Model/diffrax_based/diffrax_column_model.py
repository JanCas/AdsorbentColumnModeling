import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float
import sys
from pathlib import Path
import numpy as np

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

def make_adsorption_event(bed_utilization):
    def adsorption_finish_event(t, y: ColumnState, params: ColumnParameters, **kwargs):
        total_adsorbed = jnp.sum(y.n.vals)
        potential_adsorbed = params.isotherm(params.C_in) * len(y.n.vals)
        ratio = total_adsorbed / potential_adsorbed

        return ratio > bed_utilization
    return adsorption_finish_event

def make_desorption_event(potential_adsorbed, threshold=0.02):
    """Create desorption event that stops when loading drops below threshold of max potential."""
    def desorption_finish_event(t, y: ColumnState, params: ColumnParameters, **kwargs):
        total_adsorbed = jnp.sum(y.n.vals)
        ratio = total_adsorbed / potential_adsorbed
        return ratio < threshold
    return desorption_finish_event


def get_finish_state(solution):
    t = np.array(solution.ts)
    t = t[np.isfinite(t)]

    C = np.array(solution.ys.C.vals)
    C = C[np.isfinite(C).all(axis=1)]

    n = np.array(solution.ys.n.vals)
    n = n[np.isfinite(n).all(axis=1)]

    return t, C, n

def run_model(column_params: ColumnParameters, bed_utilization: float = .5) ->  float:
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
    n = 10
    y0_ads = ColumnState(
        C = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0),
        n = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0)
    )

    #Temporal discretization
    t0 = 0
    t_final = jnp.inf
    dt = 1
    saveat = diffrax.SaveAt(t0=True, steps=True)

    #Tolerances
    rtol=1e-6
    atol=1e-6
    # stepsize_controller = diffrax.PIDController()
    stepsize_controller = diffrax.ConstantStepSize()

    ads_event = diffrax.Event(make_adsorption_event(bed_utilization))

    solution_ads = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(column_ode),
        solver=diffrax.Euler(),
        t0=t0,
        t1=t_final,
        dt0=dt,
        y0=y0_ads,
        args=column_params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=ads_event,
        max_steps=int(1e6),
        # progress_meter=diffrax.TqdmProgressMeter()
    )

    t_ads, C_ads, n_ads = get_finish_state(solution_ads)
    y0_des = ColumnState(
        C=SpatialDiscretisation(x0, x_final, C_ads[-1, :]),
        n=SpatialDiscretisation(x0, x_final, n_ads[-1, :])
    )

    # Calculate potential loading before modifying column_params
    potential_adsorbed = column_params.isotherm(column_params.C_in) * n
    des_event = diffrax.Event(make_desorption_event(potential_adsorbed))

    column_params = column_params.replace(C_in=0, k_s=-column_params.k_s)

    solution_des = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(column_ode),
        solver=diffrax.Euler(),
        t0 = t_ads[-1],
        t1=t_final,
        dt0=dt,
        y0=y0_des,
        args=column_params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=des_event,
        max_steps = int(1e6)
    )

    t_des, C_des, n_des = get_finish_state(solution=solution_des)

    return (t_ads, C_ads, n_ads), (t_des, C_des, n_des)

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

'''
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

    (t_ads, C_ads, n_ads), (t_des, C_des, n_des) = run_model(params)

    params = params.replace(C_in=50)

    plot_breakthrough(t_ads, C_ads, params, exp_curve)
    plot_desorption(t_des, C_des, n_des)

'''
