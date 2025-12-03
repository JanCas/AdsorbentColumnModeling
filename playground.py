from utils.SensitivityStudy import SensitivityAnalyzer
from Model.non_dim_model import simulate_column_temkin_star

if __name__ == "__main__":

    sens = SensitivityAnalyzer(model_func=simulate_column_temkin_star,
                                parameters= {"phi_star": 10, "Da2_star": 10, "Lambda_star": 0.1},
                                bounds= {"phi_star": (-1, 2), "Da2_star": (-2, 2), "Lambda_star": (0.5, 5)},
                                parameters_log_scale= ["phi_star", "Da2_star"],
                                fixed_params={"C_star_thresh": 0.5},
                                n_workers=1
                                )
    
    x=sens.sobol(n_samples=200)
    print(x)
