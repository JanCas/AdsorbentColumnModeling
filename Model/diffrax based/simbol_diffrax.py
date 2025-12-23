from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from diffrax_non_dim import run_wrapper, NonDimNumbers
import cmcrameri.cm as cm
from matplotlib import colormaps
import jax

colormaps.register(cm.batlow, name="batlow")
plt.style.use('natcomm_paper.mplstyle')
 
problem = {
    "num_vars": 3,
    "names": ["log10_phi_star", "log10_Da2_star", "Lambda_star"],
    "bounds": [
        [.01,  10],   # lambda 
        [.005,  10],   # Da2* 
        [.1,  10],   # theta 
    ],
    "dists": ["unif", "unif", "unif"]
}

pretty_names = [r"$\Lambda^*$", r"$Da^*$", r"$\Theta^*$"]

def plot_sobol_indices(df, filename="sobol_indices_langmuir_5.png", title="Sobol sensitivity (LDF)"):
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

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    print(jax.devices())
    print("Generating Sobol Samples...")
    
    N=2048 # base sample size
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    print(f"Total number of evaluations: {param_values.shape[0]}")

    tau = np.zeros(param_values.shape[0])
    bed_util = np.zeros(param_values.shape[0])

    with tqdm.tqdm(total=len(tau), desc="Evaluating Model", unit="eval") as pbar:
        for i, (Lambda, Da, theta) in enumerate(param_values):
            
            tau[i], bed_util[i] = run_wrapper(
                NonDimNumbers(
                    Da=Da, Lambda=Lambda, theta=theta, epsilon=.35
                )
            )

            pbar.set_postfix({
                "Lambda": f"{Lambda:.2g}",
                "Da": f"{Da:.2g}",
                "theta": f"{theta:.2g}",
                "U_b": f"{bed_util[i]:.3g}",
                "tau": f"{tau[i]}"
            })
            pbar.update(1)


    print("Performing Sobol analysis for breakthrough time...")
    Si_breakthrough = sobol.analyze(
        problem,
        tau,
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
        bed_util,
        calc_second_order=False,
        print_to_console=True,
    )

    S1_bed = Si_bed_util["S1"]
    ST_bed = Si_bed_util["ST"]
    S1_conf_bed = Si_bed_util["S1_conf"]
    ST_conf_bed = Si_bed_util["ST_conf"]

    # Convert back from log10-space for saving
    Lambda     = param_values[:, 0]
    Da     = param_values[:, 1]
    Theta  = param_values[:, 2]

    print("Saving sample data...")
    df_samples = pd.DataFrame({
        "Lambda": Lambda,
        "Da2_star": Da,
        "Theta": Theta,
        "tau_star_break": tau,
        "bed_utilization": bed_util,
    })
    df_samples.to_csv("sobol_samples_langmuir_05.csv", index=False)

    print("Saving Sobol indices...")
    df_sobol_breakthrough = pd.DataFrame({
        "param": pretty_names,
        "S1": S1,
        "S1_conf": S1_conf,
        "ST": ST,
        "ST_conf": ST_conf,
    })
    df_sobol_breakthrough.to_csv("sobol_indices_breakthrough_langmuir_05.csv", index=False)

    df_sobol_bed_util = pd.DataFrame({
        "param": pretty_names,
        "S1": S1_bed,
        "S1_conf": S1_conf_bed,
        "ST": ST_bed,
        "ST_conf": ST_conf_bed,
    })
    df_sobol_bed_util.to_csv("sobol_indices_bed_util_langmuir_05.csv", index=False)

    print("Plotting Sobol indices...")
    plot_sobol_indices(df_sobol_breakthrough, filename="sobol_indices_breakthrough_05.png",
                       title="Sobol sensitivity - Breakthrough time")
    plot_sobol_indices(df_sobol_bed_util, filename="sobol_indices_bed_util_05.png",
                       title="Sobol sensitivity - Bed utilization")

    print("\nDone.")
