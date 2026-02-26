import JansPlottingStuff as JPS
import jax.numpy as jnp
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from diffrax_non_dim import run_cycle, NonDimNumbers
import cmcrameri.cm as cm
from matplotlib import colormaps
import jax


# ---------------------------------------------------------------------------
# Sobol problem definition  (5 parameters)
# ---------------------------------------------------------------------------
problem = {
    "num_vars": 5,
    "names": ["Lambda", "Da", "theta", "C_thresh_ads", "C_thresh_des"],
    "bounds": [
        [0.01,  10.0],    # Lambda  — sorbent/fluid capacity ratio
        [0.005, 10.0],    # Da      — Damkoehler number
        [0.1,   10.0],    # theta   — isotherm steepness
        [0.01,   0.5],    # C_thresh_ads — adsorption outlet cutoff
        [0.01,   0.5],    # C_thresh_des — desorption eluate cutoff
    ],
    "dists": ["unif", "unif", "unif", "unif", "unif"],
}

pretty_names = [
    r"$\Lambda$", r"$Da$", r"$\Theta$",
    r"$C^*_{th,ads}$", r"$C^*_{th,des}$",
]

ETA_P = 1      # pump efficiency
EPSILON = 0.35   # bed porosity
PSI = 1.0        # viscosity ratio (same fluid)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sobol_indices(df, filename="sobol_indices.png", title="Sobol sensitivity"):
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].bar(x, df["S1"], yerr=df["S1_conf"], width=0.6, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["param"], fontsize=9)
    axes[0].set_ylabel("S1")
    axes[0].set_title("First-order")

    axes[1].bar(x, df["ST"], yerr=df["ST_conf"], width=0.6, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["param"], fontsize=9)
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


def build_sobol_df(Si, names):
    return pd.DataFrame({
        "param": names,
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    JPS.apply()
    jax.config.update("jax_platform_name", "cpu")
    print(jax.devices())
    print("Generating Sobol samples …")

    N = 2 ** 12
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    n_evals = param_values.shape[0]
    print(f"Total evaluations: {n_evals}")

    # Pre-allocate output arrays
    tau_ads      = np.zeros(n_evals)
    tau_des      = np.zeros(n_evals)
    U_b          = np.zeros(n_evals)
    eta_li       = np.zeros(n_evals)
    R_outlet_des        = np.zeros(n_evals)
    sec_star     = np.zeros(n_evals)
    productivity = np.zeros(n_evals)

    with tqdm.tqdm(total=n_evals, desc="Evaluating", unit="eval") as pbar:
        for i, (Lambda, Da, theta, c_th_ads, c_th_des) in enumerate(param_values):

            result = run_cycle(
                non_dim=NonDimNumbers(
                    Da=Da, Lambda=Lambda, theta=theta, epsilon=EPSILON,
                ),
                c_thresh_ads=jnp.asarray(c_th_ads),
                c_thresh_des=jnp.asarray(c_th_des),
                eta_p=jnp.asarray(ETA_P),
                Psi=jnp.asarray(PSI),
            )

            tau_ads[i]      = result[0]
            tau_des[i]      = result[1]
            U_b[i]          = result[2]
            eta_li[i]       = result[3]
            R_outlet_des[i]        = result[4][0]
            sec_star[i]     = result[5][0]
            productivity[i] = result[6][0]

            pbar.set_postfix({
                "Λ": f"{Lambda:.2g}",
                "Da": f"{Da:.2g}",
                "θ": f"{theta:.2g}",
                "c_ads": f"{c_th_ads:.2f}",
                "c_des": f"{c_th_des:.2f}",
                "SEC*": f"{sec_star[i]:.3g}",
            })
            pbar.update(1)

    # ---- Sobol analysis on each output -----------------------------------
    outputs = {
        "tau_ads":  ("Sobol — Adsorption time",         tau_ads),
        "tau_des":  ("Sobol — Desorption time",          tau_des),
        "U_b":      ("Sobol — Bed utilisation",           U_b),
        "eta_li":   ("Sobol — Li removal efficiency",     eta_li),
        "R_outlet_des":    ("Sobol — Li recovered (desorption)", R_outlet_des),
        "sec_star":     ("Sobol — SEC*",                      sec_star),
        "productivity": ("Sobol — Productivity (R_outlet_des/τ_cycle)", productivity),
    }

    for key, (title, values) in outputs.items():
        # Skip if all values are identical (no variance to analyse)
        if np.ptp(values) < 1e-15:
            print(f"\nSkipping {key}: no variance.")
            continue

        print(f"\nSobol analysis: {key}")
        Si = sobol.analyze(
            problem, values, calc_second_order=False, print_to_console=True,
        )
        df = build_sobol_df(Si, pretty_names)
        df.to_csv(f"sobol_{key}.csv", index=False)
        plot_sobol_indices(df, filename=f"sobol_{key}.png", title=title)

    # ---- Save raw sample data --------------------------------------------
    print("\nSaving sample data …")
    pd.DataFrame({
        "Lambda":       param_values[:, 0],
        "Da":           param_values[:, 1],
        "theta":        param_values[:, 2],
        "C_thresh_ads": param_values[:, 3],
        "C_thresh_des": param_values[:, 4],
        "tau_ads":      tau_ads,
        "tau_des":      tau_des,
        "U_b":          U_b,
        "eta_li":       eta_li,
        "R_outlet_des":        R_outlet_des,
        "sec_star":     sec_star,
        "productivity": productivity,
    }).to_csv("sobol_cycle_samples.csv", index=False)

    print("\nDone.")
