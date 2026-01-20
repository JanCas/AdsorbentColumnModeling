from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
import sys
from pathlib import Path
from diffrax_column_model import run_model
import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.Dataclasses import ColumnParameters, Study, BreakthroughCurve



class ColumnOptimizationProblem(ElementwiseProblem):
    
    def __init__(self, study: Study, C_in: float, L_bounds: tuple, u_s_bounds: tuple, bed_util_bounds: tuple):
        super().__init__(
            n_var=3,
            n_obj=1,
            xl=[L_bounds[0], u_s_bounds[0], bed_util_bounds[0]],
            xu=[L_bounds[1], u_s_bounds[1], bed_util_bounds[1]]
        )

        self.study = study
        self.C_in = C_in
        self.epsilon = study.column_experiments.column_properties.porosity
        self.d_p = study.sorbent_properties.particle_diameter_si
        self.mu =  1.002e-3 #kg/m/s
        self.rho_water = 1000 #kg/m^3
        self.rho_p = study.sorbent_properties.density_si

    def _evaluate(self, x, out, *args, **kwargs):
        L, u_super, bed_util = float(x[0]), float(x[1]), float(x[2])
        params = self.study.to_column_parameter(L=L, C_in=self.C_in, u_super=u_super)

        (t_ads, C_ads, n_ads), (t_des, C_des, n_des) = run_model(params, bed_util)

        sec = self._specific_energy_consumption(u_super=u_super, n_des=n_des, t_des=t_des)


        print(f"L: {L:.2f}, u_super: {u_super:.5f}, bed_util: {bed_util:.4f}, sec: {sec:.3f}, t_ads: {t_ads[-1]:.1f},t_des: {t_des[-1]:.1f}")
        out["F"] = [sec]

    
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

        return pumping_power / li_recovered


def run_optimization(study: Study, C_in: float, L_bounds: tuple, u_bounds: tuple, bed_util_bounds: tuple,pop_size=40, seed=1):
    problem = ColumnOptimizationProblem(study, C_in, L_bounds, u_bounds, bed_util_bounds)

    # X = np.random.uniform(problem.xl, problem.xu, size=(pop_size, 2))
    # X[0, :] = [1, .0001]

    algorithm = DE(pop_size=pop_size)
    termination = DefaultSingleObjectiveTermination(ftol=1e-3, xtol=1e-3, period=10)
    res = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    return res

if __name__ == "__main__":
    study = Study.from_json("LiteratureReview/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")

    res = run_optimization(study, 50, (.6, 2), (0.0001, .0099), (.1, .9))

    print()