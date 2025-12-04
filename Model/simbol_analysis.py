# sobol_temkin_star.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol
import tqdm

from non_dim_model import simulate_column_temkin_star


# --------------------------------------------------
# 1) Sobol problem definition
# --------------------------------------------------

problem = {
    "num_vars": 3,
    "names": ["log10_phi_star", "log10_Da2_star", "Lambda_star"],
    "bounds": [
        [.01,  10],   # φ* ∈ [0.1, 100]
        [.01,  10],   # Da2* ∈ [1e−2, 1e2]
        [.1,  10],   # Λ* ∈ [0.5, 5]  (tune to your system)
    ],
    "dists": ["unif", "unif", "unif"]
}

pretty_names = [r"$\phi^\ast$", r"$\mathrm{Da}_2^\ast$", r"$\Lambda^\ast$"]


# --------------------------------------------------
# 2) Wrapper that calls the Euler solver
# --------------------------------------------------

def run_model_from_sample(log10_phi_star, log10_Da2_star, Lambda_star):

    tau_star_break, C_star, n_star = simulate_column_temkin_star(
        phi_star=log10_phi_star,
        Da2_star=log10_Da2_star,
        Lambda_star=Lambda_star,
        eps=0.4,
        Nx=150,
        tau_star_max=100,
        C_star_thresh=0.5,
        cfl=0.5,
        store_history=True,
    )

    bed_util = np.trapezoid(n_star[-1, :], np.linspace(0, 1, 150))  # final loading

    return tau_star_break, bed_util


# --------------------------------------------------
# 3) Plotting utility
# --------------------------------------------------

def plot_sobol_indices(df, filename="sobol_indices_langmuir5.png", title="Sobol sensitivity"):
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # First-order
    axes[0].bar(x, df["S1"], yerr=df["S1_conf"], width=0.6, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["param"])
    axes[0].set_ylabel("S1")
    axes[0].set_title("First-order")

    # Total-order
    axes[1].bar(x, df["ST"], yerr=df["ST_conf"], width=0.6, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["param"])
    axes[1].set_ylabel("ST")
    axes[1].set_title("Total-order")

    ymax = max(1.0, 1.1 * df[["S1", "ST"]].to_numpy().max())
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.savefig(filename, dpi=300)
    fig.savefig(filename.replace(".png", ".svg"))
    plt.close(fig)


# --------------------------------------------------
# 4) Main execution
# --------------------------------------------------

if __name__ == "__main__":
    print("Generating Saltelli samples...")

    N = 2048  # base sample size
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    print(f"Total model evaluations: {param_values.shape[0]}")

    Y = np.zeros(param_values.shape[0])
    bed_utils = np.zeros(param_values.shape[0])
    with tqdm.tqdm(total=len(Y), desc="Evaluating model", unit="eval") as pbar:
        for i, (log10_phi_star, log10_Da2_star, Lambda_star) in enumerate(param_values):
            Y[i], bed_utils[i] = run_model_from_sample(log10_phi_star, log10_Da2_star, Lambda_star)
            pbar.set_postfix({
            "phi*":     f"{log10_phi_star:.3g}",
            "Da2*":     f"{log10_Da2_star:.3g}",
            "Lambda*":  f"{Lambda_star:.3g}",
            "tau*_break": f"{Y[i]:.3g}",
            "bed_util": f"{bed_utils[i]:.3g}",
        })
            pbar.update(1)



    print("Performing Sobol analysis for breakthrough time...")
    Si_breakthrough = sobol.analyze(
        problem,
        Y,
        calc_second_order=False,
        print_to_console=True,
    )

    S1 = Si_breakthrough["S1"]
    ST = Si_breakthrough["ST"]
    S1_conf = Si_breakthrough["S1_conf"]
    ST_conf = Si_breakthrough["ST_conf"]

    print("\nPerforming Sobol analysis for bed utilization...")
    Si_bed_util = sobol.analyze(
        problem,
        bed_utils,
        calc_second_order=False,
        print_to_console=True,
    )

    S1_bed = Si_bed_util["S1"]
    ST_bed = Si_bed_util["ST"]
    S1_conf_bed = Si_bed_util["S1_conf"]
    ST_conf_bed = Si_bed_util["ST_conf"]

    # Convert back from log10-space for saving
    phi_star_vals     = param_values[:, 0]
    Da2_star_vals     = param_values[:, 1]
    Lambda_star_vals  = param_values[:, 2]

    print("Saving sample data...")
    df_samples = pd.DataFrame({
        "phi_star": phi_star_vals,
        "Da2_star": Da2_star_vals,
        "Lambda_star": Lambda_star_vals,
        "tau_star_break": Y,
        "bed_utilization": bed_utils,
    })
    df_samples.to_csv("sobol_samples_langmuir_5.csv", index=False)

    print("Saving Sobol indices...")
    df_sobol_breakthrough = pd.DataFrame({
        "param": pretty_names,
        "S1": S1,
        "S1_conf": S1_conf,
        "ST": ST,
        "ST_conf": ST_conf,
    })
    df_sobol_breakthrough.to_csv("sobol_indices_breakthrough_langmuir_5.csv", index=False)

    df_sobol_bed_util = pd.DataFrame({
        "param": pretty_names,
        "S1": S1_bed,
        "S1_conf": S1_conf_bed,
        "ST": ST_bed,
        "ST_conf": ST_conf_bed,
    })
    df_sobol_bed_util.to_csv("sobol_indices_bed_util_langmuir_5.csv", index=False)

    print("Plotting Sobol indices...")
    plot_sobol_indices(df_sobol_breakthrough, filename="sobol_indices_breakthrough_5.png",
                       title="Sobol sensitivity - Breakthrough time")
    plot_sobol_indices(df_sobol_bed_util, filename="sobol_indices_bed_util_5.png",
                       title="Sobol sensitivity - Bed utilization")

    print("\nDone.")
