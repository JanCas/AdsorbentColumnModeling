from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
import sys
import argparse
from pathlib import Path
from diffrax_column_model import run_model
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from matplotlib import colormaps

colormaps.register(cm.batlow, name="batlow")
plt.style.use('natcomm_paper.mplstyle')


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.Dataclasses import ColumnParameters, Study, BreakthroughCurve



class ColumnOptimizationProblem(ElementwiseProblem):
    
    def __init__(self, study: Study, C_in: float, L_bounds: tuple, u_s_bounds: tuple, des_threshold_bounds: tuple, loss_fraction: float = 0.01):
        super().__init__(
            n_var=3,
            n_obj=2,
            xl=[L_bounds[0], u_s_bounds[0], des_threshold_bounds[0]],
            xu=[L_bounds[1], u_s_bounds[1], des_threshold_bounds[1]]
        )

        self.study = study
        self.C_in = C_in
        self.loss_fraction = loss_fraction
        self.epsilon = study.column_experiments.column_properties.porosity
        self.d_p = study.sorbent_properties.particle_diameter_si
        self.mu =  1.002e-3 #kg/m/s
        self.rho_water = 1000 #kg/m^3
        self.rho_p = study.sorbent_properties.density_si


    def _evaluate(self, x, out, *args, **kwargs):
        L, u_super, des_threshold = float(x[0]), float(x[1]), float(x[2])
        params = self.study.to_column_parameter(L=L, C_in=self.C_in, u_super=u_super)

        (t_ads, C_ads, n_ads), (t_des, C_des, n_des), fraction_lost = run_model(params, self.loss_fraction, des_threshold)

        sec, li_recovered = self._specific_energy_consumption(u_super=u_super, n_des=n_des, t_des=t_des)

        n_eq = float(params.isotherm(params.C_in))
        bed_util_ads = np.mean(n_ads[-1, :]) / n_eq
        bed_util_des = np.mean(n_des[-1, :]) / n_eq

        t_cycle = t_des[-1]
        productivity = li_recovered / t_cycle  # mol/m³/s

        print(f"L: {L:.2f}, u_super: {u_super:.5f}, sec: {sec:.3f}, prod: {productivity:.6f}, t_ads: {t_ads[-1]:.1f}, t_des: {t_des[-1] - t_ads[-1]:.1f}, bed_util_ads: {bed_util_ads:.3f}, bed_util_des: {bed_util_des:.3f}, frac_lost: {fraction_lost:.3f}, li_recovered: {li_recovered: .3f}")
        out["F"] = [sec, -productivity]  # Negative because we minimize (want max productivity)

    
    def _pressure_drop_per_unit_length(self, u_super):
        term1 = (150 * self.mu * (1-self.epsilon)**2 * u_super) / (self.epsilon**3 * self.d_p**2)
        term2 = (1.75 * self.rho_water * (1-self.epsilon) * u_super**2) / (self.epsilon**3 * self.d_p)

        return term1 + term2
    
    def _li_recovered(self, n_des):
        n_start = np.mean(n_des[0, :])
        n_end = np.mean(n_des[-1, :])

        total = (n_start - n_end) * (1-self.epsilon) * self.rho_p
        return total

    def _specific_energy_consumption(self, u_super, n_des, t_des):
        dP_dL = self._pressure_drop_per_unit_length(u_super=u_super)

        pumping_power = dP_dL * u_super * t_des[-1]

        li_recovered = self._li_recovered(n_des=n_des)

        return pumping_power / li_recovered, li_recovered


def plot_pareto_front(res, save_path=None):
    """Plot the Pareto front: SEC vs Productivity."""
    sec = res.F[:, 0]
    productivity = -res.F[:, 1]  # Convert back to positive

    fig, ax = plt.subplots()
    ax.scatter(sec, productivity, c=range(len(sec)), cmap='batlow')
    ax.set_xlabel('SEC (J/mol)')
    ax.set_ylabel('Productivity (mol/m³/s)')
    ax.set_title('Pareto Front: SEC vs Productivity')

    if save_path:
        fig.savefig(save_path)
    plt.show()


def run_optimization(study: Study, C_in: float, L_bounds: tuple, u_bounds: tuple, des_threshold_bounds: tuple, loss_fraction: float = 0.01, pop_size=40, seed=1):
    problem = ColumnOptimizationProblem(study, C_in, L_bounds, u_bounds, des_threshold_bounds, loss_fraction)

    # X = np.random.uniform(problem.xl, problem.xu, size=(pop_size, 2))
    # X[0, :] = [1, .0001]

    algorithm = NSGA2(pop_size=pop_size)
    termination = DefaultMultiObjectiveTermination(
        # xtol=1e-3,      # stop when design variables change < 0.1%
        # ftol=1e-3,      # stop when objective changes < 0.1%
        period=10,      # check over last 10 generations
        n_max_gen=50   # hard limit on generations
    )
    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    return res, problem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize column parameters for minimum SEC")
    parser.add_argument("-lf", "--loss-fraction", type=float, default=0.01, help="Fraction of incoming material lost before stopping adsorption (default: 0.01)")
    args = parser.parse_args()

    study = Study.from_json("LiteratureReview/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")

    res, problem = run_optimization(study, 50, (.5, 4), (0.00005, .0099), (.02, .5), loss_fraction=args.loss_fraction)

    print("\n" + "="*80)
    print("PARETO FRONT")
    print("="*80)
    print(f"{'L':>8} {'u_super':>12} {'des_thresh':>12} {'SEC':>12} {'Productivity':>14}")
    print("-"*80)
    for i in range(len(res.X)):
        L, u, d = res.X[i]
        sec, neg_prod = res.F[i]
        print(f"{L:8.4f} {u:12.6f} {d:12.4f} {sec:12.4f} {-neg_prod:14.6f}")
    print("="*80)

    # Find and print extremes
    min_sec_idx = np.argmin(res.F[:, 0])
    max_prod_idx = np.argmin(res.F[:, 1])  # Most negative = highest productivity

    # Find elbow point (max distance from line connecting extremes)
    # Normalize objectives to [0,1] for fair distance calculation
    sec_vals = res.F[:, 0]
    prod_vals = -res.F[:, 1]  # Convert back to positive productivity
    sec_norm = (sec_vals - sec_vals.min()) / (sec_vals.max() - sec_vals.min() + 1e-10)
    prod_norm = (prod_vals - prod_vals.min()) / (prod_vals.max() - prod_vals.min() + 1e-10)

    # Line from (0,1) to (1,0) in normalized space (min SEC has high prod_norm, max prod has high sec_norm)
    # Distance from point (x,y) to line ax + by + c = 0: |ax + by + c| / sqrt(a² + b²)
    # Line: x + y - 1 = 0 (connects (0,1) and (1,0))
    distances = np.abs(sec_norm + prod_norm - 1) / np.sqrt(2)
    elbow_idx = np.argmax(distances)

    print("\nEXTREME SOLUTIONS")
    print("-"*80)
    print("Minimum SEC:")
    print(f"  L={res.X[min_sec_idx, 0]:.4f} m, u={res.X[min_sec_idx, 1]:.6f} m/s, des_thresh={res.X[min_sec_idx, 2]:.4f}")
    print(f"  SEC={res.F[min_sec_idx, 0]:.4f}, Productivity={-res.F[min_sec_idx, 1]:.6f} mol/m³/s")
    print("\nMaximum Productivity:")
    print(f"  L={res.X[max_prod_idx, 0]:.4f} m, u={res.X[max_prod_idx, 1]:.6f} m/s, des_thresh={res.X[max_prod_idx, 2]:.4f}")
    print(f"  SEC={res.F[max_prod_idx, 0]:.4f}, Productivity={-res.F[max_prod_idx, 1]:.6f} mol/m³/s")
    print("\nElbow (Best Trade-off):")
    print(f"  L={res.X[elbow_idx, 0]:.4f} m, u={res.X[elbow_idx, 1]:.6f} m/s, des_thresh={res.X[elbow_idx, 2]:.4f}")
    print(f"  SEC={res.F[elbow_idx, 0]:.4f}, Productivity={-res.F[elbow_idx, 1]:.6f} mol/m³/s")
    print("-"*80)
    print(f"Generations: {res.algorithm.n_gen}")
    print("="*80)

    plot_pareto_front(res, save_path=f"pareto_front_{args.loss_fraction}.svg")