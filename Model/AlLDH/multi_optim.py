"""NSGA-II multi-objective optimizer for the dimensional AlLDH column model.

Searches column designs that trade off two competing objectives:
  * SEC          - specific energy consumption [J/mol Li], from Ergun pressure
                   drop x flow x cycle time / Li recovered  (minimise)
  * productivity - Li recovered per bed cross-section per cycle time
                   [mol/(m^2 s)]                             (maximise)

Each candidate is 5 design variables (column length L, superficial velocity
u_super, desorption threshold, PSO rate k_s, bed porosity epsilon). ``_evaluate``
builds a ``ColumnParameters``, runs a full adsorption/desorption cycle via
``diffrax_column_model.run_model``, and maps the result to (SEC, -productivity);
infeasible or solver-failed individuals are penalised with inf objectives.

pymoo's NSGA-II evolves the population to a Pareto front, which the ``__main__``
block prints, tags with extreme/elbow points, and saves as CSV + SVG. Physical
constants and the isotherm come from a literature ``Study`` loaded from JSON.

Run x64 on CPU (set below) — the model needs float64 for accuracy/stability.
"""
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import argparse
import logging
import sys
import time
from pathlib import Path

import JansPlottingStuff as JPS
import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from diffrax_column_model import run_model
from matplotlib import colormaps
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.Dataclasses import ColumnParameters, Study, BreakthroughCurve
from utils.logging_setup import configure_logging

_log = logging.getLogger(__name__)



class ColumnOptimizationProblem(ElementwiseProblem):
    """pymoo problem: 5 design vars -> 2 objectives (SEC, -productivity).

    Design vector x = [L, u_super, des_threshold, k_s, epsilon]; xl/xu are the
    per-variable lower/upper bounds. ElementwiseProblem evaluates one candidate
    per ``_evaluate`` call.
    """

    def __init__(self, study: Study, C_in: float, L_bounds: tuple, u_s_bounds: tuple, des_threshold_bounds: tuple, k_s_bounds: tuple, epsilon_bounds: tuple, loss_fraction: float = 0.01):
        super().__init__(
            n_var=5,
            n_obj=2,
            xl=[L_bounds[0], u_s_bounds[0], des_threshold_bounds[0], k_s_bounds[0], epsilon_bounds[0]],
            xu=[L_bounds[1], u_s_bounds[1], des_threshold_bounds[1], k_s_bounds[1], epsilon_bounds[1]]
        )

        self.study = study
        self.C_in = C_in
        self.loss_fraction = loss_fraction
        self.d_p = study.sorbent_properties.particle_diameter_si
        self.mu =  1.002e-3 #kg/m/s
        self.rho_water = 1000 #kg/m^3
        self.rho_p = study.sorbent_properties.density_si


    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate one candidate design; writes objectives into ``out['F']``."""
        L, u_super, des_threshold, k_s, epsilon = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
        # Build column params from the study, then override the swept variables.
        # u_inter = u_super / epsilon (interstitial from superficial velocity).
        params = self.study.to_column_parameter(L=L, C_in=self.C_in, u_super=u_super)
        params = params.replace(k_s=k_s, epsilon=epsilon, u_inter=u_super / epsilon)

        t0 = time.perf_counter()
        (t_ads, C_ads, n_ads), (t_des, C_des, n_des, cumulative_out_des), fraction_lost, solver_ok = run_model(params, self.loss_fraction, des_threshold)
        elapsed = time.perf_counter() - t0

        # Penalise a failed solve with infinite objectives so NSGA-II discards it.
        if not solver_ok:
            _log.warning(
                "individual rejected (solver failed): L=%.2f u_super=%.5f k_s=%.4e eps=%.3f  elapsed=%.2fs",
                L, u_super, k_s, epsilon, elapsed,
            )
            out["F"] = [np.inf, np.inf]
            return

        li_recovered = self._li_recovered(n_ads, n_des, L, epsilon)  # mol/m²

        # A design that captures no Li is infeasible -> also penalise with inf.
        if li_recovered <= 0 or not np.isfinite(li_recovered):
            _log.warning(
                "individual rejected (infeasible li_recovered=%.3e): L=%.2f u_super=%.5f k_s=%.4e eps=%.3f  elapsed=%.2fs",
                li_recovered, L, u_super, k_s, epsilon, elapsed,
            )
            out["F"] = [np.inf, np.inf]
            return

        sec = self._specific_energy_consumption(u_super=u_super, li_recovered=li_recovered, t_des=t_des, L=L, epsilon=epsilon)

        n_eq = float(params.isotherm(params.C_in))
        bed_util_ads = np.mean(n_ads) / n_eq
        bed_util_des = np.mean(n_des) / n_eq

        t_cycle = float(t_des)
        productivity = li_recovered / t_cycle / L # mol/(m²·s)

        _log.info(
            "indiv eval: L=%.2f u=%.5f k_s=%.4e eps=%.3f des_th=%.3f -> "
            "sec=%.3f prod=%.6f t_ads=%.1f t_des=%.1f U_ads=%.3f U_des=%.3f "
            "frac_lost=%.3f li_rec=%.3f  elapsed=%.2fs",
            L, u_super, k_s, epsilon, des_threshold,
            sec, productivity, float(t_ads), float(t_des) - float(t_ads),
            bed_util_ads, bed_util_des,
            float(fraction_lost), li_recovered, elapsed,
        )
        out["F"] = [sec, -productivity]  # Negative because we minimize (want max productivity)


    def _pressure_drop_per_unit_length(self, u_super, epsilon):
        """Ergun equation dP/dL [Pa/m]: viscous (term1) + inertial (term2) losses
        through the packed bed as a function of velocity, porosity, particle size."""
        term1 = (150 * self.mu * (1-epsilon)**2 * u_super) / (epsilon**3 * self.d_p**2)
        term2 = (1.75 * self.rho_water * (1-epsilon) * u_super**2) / (epsilon**3 * self.d_p)

        return term1 + term2

    def _li_recovered(self, n_ads, n_des, L, epsilon):
        """Compute lithium recovered per unit cross-sectional area [mol/m²]."""
        n_ads = np.asarray(n_ads)
        n_des = np.asarray(n_des)
        x = np.linspace(0.0, L, n_ads.shape[0])
        # Axial integral of (loaded - stripped) loading = net Li per kg sorbent
        # times bed length; convert to per cross-section via solid bulk density.
        delta_n = (np.trapezoid(n_ads, x) - np.trapezoid(n_des, x))  # mol·m/kg
        return delta_n * self.rho_p * (1 - epsilon)  # mol/m²

    def _specific_energy_consumption(self, u_super, li_recovered, t_des, L, epsilon):
        """Compute SEC [J/mol]. Both pumping energy and li_recovered are per unit cross-sectional area."""
        dP_dL = self._pressure_drop_per_unit_length(u_super=u_super, epsilon=epsilon)

        pumping_energy = dP_dL * L * u_super * float(t_des)  # J/m²

        return pumping_energy / li_recovered  # J/mol


def plot_pareto_front(res, save_path=None):
    """Plot the Pareto front: SEC vs Productivity."""
    sec = res.F[:, 0]
    productivity = -res.F[:, 1]  # Convert back to positive

    fig, ax = plt.subplots()
    ax.scatter(sec, productivity, c=range(len(sec)), cmap='batlow')
    ax.set_xlabel('SEC (J/mol)')
    ax.set_ylabel('Productivity (mol/(m²·s))')
    ax.set_title('Pareto Front: SEC vs Productivity')

    if save_path:
        fig.savefig(save_path)
    plt.show()


def run_optimization(study: Study, C_in: float, L_bounds: tuple, u_bounds: tuple, des_threshold_bounds: tuple, k_s_bounds: tuple, epsilon_bounds: tuple, loss_fraction: float = 0.01, pop_size=40, seed=1):
    """Build the problem and run NSGA-II; returns (pymoo result, problem)."""
    _log.info(
        "run_optimization: C_in=%.3g loss_fraction=%.3g pop_size=%d seed=%d  "
        "bounds: L=%s u=%s des_thresh=%s k_s=%s eps=%s",
        C_in, loss_fraction, pop_size, seed,
        L_bounds, u_bounds, des_threshold_bounds, k_s_bounds, epsilon_bounds,
    )
    problem = ColumnOptimizationProblem(study, C_in, L_bounds, u_bounds, des_threshold_bounds, k_s_bounds, epsilon_bounds, loss_fraction)

    # X = np.random.uniform(problem.xl, problem.xu, size=(pop_size, 2))
    # X[0, :] = [1, .0001]

    algorithm = NSGA2(pop_size=pop_size)
    termination = DefaultMultiObjectiveTermination(
        # xtol=1e-3,      # stop when design variables change < 0.1%
        # ftol=1e-3,      # stop when objective changes < 0.1%
        # period=10,      # check over last 10 generations
        # n_max_gen=50   # hard limit on generations
    )
    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    return res, problem

if __name__ == "__main__":
    configure_logging()
    _log.info("multi_optim.py starting")
    JPS.apply()
    parser = argparse.ArgumentParser(description="Optimize column parameters for minimum SEC")
    parser.add_argument("-lf", "--loss-fraction", type=float, default=0.01, help="Fraction of incoming material lost before stopping adsorption (default: 0.01)")
    parser.add_argument("-br", "--brine_concentration", type=float, default=50, help="Incoming brine concentration in mol/m^3")
    parser.add_argument("--k_s_bounds", nargs=2, type=float, default=None, help="Lower and upper bounds for k_s (mol/kg/s). Defaults to 0.5x-2x the study value at the chosen brine concentration.")
    parser.add_argument("--epsilon_bounds", nargs=2, type=float, default=(0.35, 0.45), help="Lower and upper bounds for bed porosity epsilon (default: 0.35 0.45)")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write Pareto CSV and plot into. Created if missing (default: current directory).")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = Study.from_json("LiteratureReview/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")

    if args.k_s_bounds is None:
        k_s_study = study.get_kinetics_experiment_from_concentration(args.brine_concentration).kinetics_params.k2_si
        k_s_bounds = (0.5 * k_s_study, 2.0 * k_s_study)
    else:
        k_s_bounds = tuple(args.k_s_bounds)
    epsilon_bounds = tuple(args.epsilon_bounds)
    _log.info("k_s_bounds=%s epsilon_bounds=%s", k_s_bounds, epsilon_bounds)

    res, problem = run_optimization(study, args.brine_concentration, (.5, 500), (0.00005, .0099), (.02, .5), k_s_bounds, epsilon_bounds, loss_fraction=args.loss_fraction)

    print("\n" + "="*80)
    print("PARETO FRONT")
    print("="*80)
    print(f"{'L':>8} {'u_super':>12} {'des_thresh':>12} {'k_s':>12} {'eps':>8} {'SEC':>12} {'Productivity':>14}")
    print("-"*80)
    for i in range(len(res.X)):
        L, u, d, k, eps = res.X[i]
        sec, neg_prod = res.F[i]
        print(f"{L:8.4f} {u:12.6f} {d:12.4f} {k:12.4e} {eps:8.4f} {sec:12.4f} {-neg_prod:14.6f}")
    print("="*80)

    # Find and print extremes
    min_sec_idx = np.argmin(res.F[:, 0])
    max_prod_idx = np.argmin(res.F[:, 1])  # Most negative = highest productivity

    # Find elbow point (max perpendicular distance from line connecting extremes)
    # Normalize objectives to [0,1] for fair distance calculation
    sec_vals = res.F[:, 0]
    prod_vals = -res.F[:, 1]  # Convert back to positive productivity
    sec_norm = (sec_vals - sec_vals.min()) / (sec_vals.max() - sec_vals.min() + 1e-10)
    prod_norm = (prod_vals - prod_vals.min()) / (prod_vals.max() - prod_vals.min() + 1e-10)

    # Line connecting the two extreme points in normalized space
    p1 = np.array([sec_norm[min_sec_idx], prod_norm[min_sec_idx]])
    p2 = np.array([sec_norm[max_prod_idx], prod_norm[max_prod_idx]])
    # Distance from each point to the line through p1 and p2
    d = p2 - p1
    distances = np.abs(d[1] * (sec_norm - p1[0]) - d[0] * (prod_norm - p1[1])) / np.linalg.norm(d)
    elbow_idx = np.argmax(distances)

    print("\nEXTREME SOLUTIONS")
    print("-"*80)
    print("Minimum SEC:")
    print(f"  L={res.X[min_sec_idx, 0]:.4f} m, u={res.X[min_sec_idx, 1]:.6f} m/s, des_thresh={res.X[min_sec_idx, 2]:.4f}, k_s={res.X[min_sec_idx, 3]:.4e}, eps={res.X[min_sec_idx, 4]:.4f}")
    print(f"  SEC={res.F[min_sec_idx, 0]:.4f}, Productivity={-res.F[min_sec_idx, 1]:.6f} mol/(m²·s)")
    print("\nMaximum Productivity:")
    print(f"  L={res.X[max_prod_idx, 0]:.4f} m, u={res.X[max_prod_idx, 1]:.6f} m/s, des_thresh={res.X[max_prod_idx, 2]:.4f}, k_s={res.X[max_prod_idx, 3]:.4e}, eps={res.X[max_prod_idx, 4]:.4f}")
    print(f"  SEC={res.F[max_prod_idx, 0]:.4f}, Productivity={-res.F[max_prod_idx, 1]:.6f} mol/(m²·s)")
    print("\nElbow (Best Trade-off):")
    print(f"  L={res.X[elbow_idx, 0]:.4f} m, u={res.X[elbow_idx, 1]:.6f} m/s, des_thresh={res.X[elbow_idx, 2]:.4f}, k_s={res.X[elbow_idx, 3]:.4e}, eps={res.X[elbow_idx, 4]:.4f}")
    print(f"  SEC={res.F[elbow_idx, 0]:.4f}, Productivity={-res.F[elbow_idx, 1]:.6f} mol/(m²·s)")
    print("-"*80)
    print(f"Generations: {res.algorithm.n_gen}")
    print("="*80)

    # Save Pareto front data to CSV
    pareto_data = np.column_stack([res.X, res.F[:, 0], -res.F[:, 1]])
    header = "L,u_super,des_thresh,k_s,epsilon,SEC_J_per_mol,Productivity_mol_per_m2_per_s"
    np.savetxt(output_dir / f"pareto_front_{args.loss_fraction}.csv", pareto_data, delimiter=",", header=header, comments="")

    plot_pareto_front(res, save_path=output_dir / f"pareto_front_{args.loss_fraction}.svg")
