from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
import sys
import argparse
from pathlib import Path
from diffrax_column_model import run_model
import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.Dataclasses import ColumnParameters, Study, BreakthroughCurve



class ColumnOptimizationProblem(ElementwiseProblem):
    
    def __init__(self, study: Study, C_in: float, L_bounds: tuple, u_s_bounds: tuple, des_threshold_bounds: tuple, loss_fraction: float = 0.01):
        super().__init__(
            n_var=3,
            n_obj=1,
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

        # Store best results
        self.best_sec = np.inf
        self.best_t_ads = None
        self.best_t_des = None
        self.best_li_recovered = None
        self.best_frac_lost = None
        self.best_bed_util_ads = None

    def _evaluate(self, x, out, *args, **kwargs):
        L, u_super, des_threshold = float(x[0]), float(x[1]), float(x[2])
        params = self.study.to_column_parameter(L=L, C_in=self.C_in, u_super=u_super)

        (t_ads, C_ads, n_ads), (t_des, C_des, n_des), fraction_lost = run_model(params, self.loss_fraction, des_threshold)

        sec, li_recovered = self._specific_energy_consumption(u_super=u_super, n_des=n_des, t_des=t_des)

        n_eq = float(params.isotherm(params.C_in))
        bed_util_ads = np.mean(n_ads[-1, :]) / n_eq
        bed_util_des = np.mean(n_des[-1, :]) / n_eq

        print(f"L: {L:.2f}, u_super: {u_super:.5f}, sec: {sec:.3f}, t_ads: {t_ads[-1]:.1f}, t_des: {t_des[-1] - t_ads[-1]:.1f}, bed_util_ads: {bed_util_ads:.3f}, bed_util_des: {bed_util_des:.3f}, frac_lost: {fraction_lost:.3f}, li_recovered: {li_recovered: .3f}")
        out["F"] = [sec]

        # Track best results
        if sec < self.best_sec:
            self.best_sec = sec
            self.best_t_ads = t_ads[-1]
            self.best_t_des = t_des[-1] - t_ads[-1]
            self.best_li_recovered = li_recovered
            self.best_frac_lost = fraction_lost
            self.best_bed_util_ads = bed_util_ads

    
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


def run_optimization(study: Study, C_in: float, L_bounds: tuple, u_bounds: tuple, des_threshold_bounds: tuple, loss_fraction: float = 0.01, pop_size=40, seed=1):
    problem = ColumnOptimizationProblem(study, C_in, L_bounds, u_bounds, des_threshold_bounds, loss_fraction)

    # X = np.random.uniform(problem.xl, problem.xu, size=(pop_size, 2))
    # X[0, :] = [1, .0001]

    algorithm = DE(pop_size=pop_size)
    termination = DefaultSingleObjectiveTermination(
        xtol=1e-3,      # stop when design variables change < 0.1%
        ftol=1e-3,      # stop when objective changes < 0.1%
        period=10,      # check over last 10 generations
        n_max_gen=100   # hard limit on generations
    )
    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    return res, problem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize column parameters for minimum SEC")
    parser.add_argument("-lf", "--loss-fraction", type=float, default=0.01, help="Fraction of incoming material lost before stopping adsorption (default: 0.01)")
    args = parser.parse_args()

    study = Study.from_json("LiteratureReview/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")

    res, problem = run_optimization(study, 50, (.3, 1), (0.00001, .0005), (.02, .5), loss_fraction=args.loss_fraction)

    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Best SEC:              {res.F[0]:.4f}")
    print("-"*60)
    print("Optimal Parameters:")
    print(f"  Column length (L):     {res.X[0]:.4f} m")
    print(f"  Superficial velocity:  {res.X[1]:.6f} m/s")
    print(f"  Desorption threshold:  {res.X[2]:.4f}")
    print("-"*60)
    print("Cycle Times:")
    print(f"  Adsorption time:       {problem.best_t_ads:.1f} s")
    print(f"  Desorption time:       {problem.best_t_des:.1f} s")
    print(f"  Total cycle time:      {problem.best_t_ads + problem.best_t_des:.1f} s")
    print("-"*60)
    print("Performance:")
    print(f"  Li recovered:          {problem.best_li_recovered:.4f} mol/m³")
    print(f"  Fraction lost:         {problem.best_frac_lost:.4f}")
    print(f"  Bed util (ads):        {problem.best_bed_util_ads:.4f}")
    print(f"  Generations:           {res.algorithm.n_gen}")
    print("="*60)