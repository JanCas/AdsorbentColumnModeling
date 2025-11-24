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
        [-1.0,  2.0],   # φ* ∈ [0.1, 100]
        [-2.0,  2.0],   # Da2* ∈ [1e−2, 1e2]
        [ 0.5,  5.0],   # Λ* ∈ [0.5, 5]  (tune to your system)
    ],
}

pretty_names = [r"$\phi^\ast$", r"$\mathrm{Da}_2^\ast$", r"$\Lambda^\ast$"]


# --------------------------------------------------
# 2) Wrapper that calls the Euler solver
# --------------------------------------------------

def run_model_from_sample(log10_phi_star, log10_Da2_star, Lambda_star):
    phi_star = 10.0 ** log10_phi_star
    Da2_star = 10.0 ** log10_Da2_star

    tau_star_break, _, _ = simulate_column_temkin_star(
        phi_star=phi_star,
        Da2_star=Da2_star,
        Lambda_star=Lambda_star,
        eps=0.4,
        Nx=100,
        tau_star_max=15.0,
        C_star_thresh=0.99,
        cfl=0.5,
        store_history=False,
    )

    return tau_star_break


# --------------------------------------------------
# 3) Plotting utility
# --------------------------------------------------

def plot_sobol_indices(df, filename="sobol_indices_temkin_star_99.png"):
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

    fig.suptitle("Sobol sensitivity (Temkin, explicit Euler, starred model)")
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# --------------------------------------------------
# 4) Main execution
# --------------------------------------------------

if __name__ == "__main__":
    print("Generating Saltelli samples...")

    N = 512  # base sample size
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    print(f"Total model evaluations: {param_values.shape[0]}")

    Y = np.zeros(param_values.shape[0])
    with tqdm.tqdm(total=len(Y), desc="Evaluating model", unit="eval") as pbar:
        for i, (log10_phi_star, log10_Da2_star, Lambda_star) in enumerate(param_values):
            Y[i] = run_model_from_sample(log10_phi_star, log10_Da2_star, Lambda_star)
            # if i % 100 == 0:
            #     print(f"  Progress: {i}/{len(Y)}")
            phi_star = 10.0 ** log10_phi_star
            Da2_star = 10.0 ** log10_Da2_star
            pbar.set_postfix({
            "phi*":     f"{phi_star:.3g}",
            "Da2*":     f"{Da2_star:.3g}",
            "Lambda*":  f"{Lambda_star:.3g}",
        })
            pbar.update(1)



    print("Performing Sobol analysis...")
    Si = sobol.analyze(
        problem,
        Y,
        calc_second_order=False,
        print_to_console=True,
    )

    S1 = Si["S1"]
    ST = Si["ST"]
    S1_conf = Si["S1_conf"]
    ST_conf = Si["ST_conf"]

    # Convert back from log10-space for saving
    phi_star_vals     = 10.0 ** param_values[:, 0]
    Da2_star_vals     = 10.0 ** param_values[:, 1]
    Lambda_star_vals  = param_values[:, 2]

    print("Saving sample data...")
    df_samples = pd.DataFrame({
        "phi_star": phi_star_vals,
        "Da2_star": Da2_star_vals,
        "Lambda_star": Lambda_star_vals,
        "tau_star_break": Y,
    })
    df_samples.to_csv("sobol_samples_temkin_star_99.csv", index=False)

    print("Saving Sobol indices...")
    df_sobol = pd.DataFrame({
        "param": pretty_names,
        "S1": S1,
        "S1_conf": S1_conf,
        "ST": ST,
        "ST_conf": ST_conf,
    })
    df_sobol.to_csv("sobol_indices_temkin_star_99.csv", index=False)

    print("Plotting Sobol indices...")
    plot_sobol_indices(df_sobol)

    print("\nDone.")
