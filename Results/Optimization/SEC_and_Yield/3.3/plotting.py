"""Scatter the SEC-vs-Productivity Pareto front for one brine/extraction case.

Reads the NSGA-II Pareto-front CSVs (``pareto_front_<loss>.csv``) produced by
the dimensional AlLDH optimizer and plots the two objectives against each other:
    x = SEC_J_per_mol   -- specific energy consumption (pumping energy / mol Li
                           recovered) [J/mol]; objective is to MINIMIZE it.
    y = Productivity_mol_per_m2_per_s -- Li recovered per cycle time per bed
                           length [mol/m^2/s]; objective is to MAXIMIZE it.
Points are coloured by residence time L / u_super [s] (column length / superficial
velocity). This exploratory script keeps only the brine=50 mol/m^3, loss-fraction
0.5 case. Filenames encode brine concentration (``bNN``) and the loss fraction
(fraction of feed lost, i.e. lower loss = higher extraction efficiency)."""
import JansPlottingStuff as JPS
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re
from typing import NamedTuple

search_path = Path(".")

class df_label(NamedTuple):
    """Pairs a (brine, loss_fraction) key with the Pareto-front DataFrame."""
    label: tuple
    df: pd.DataFrame

def get_brine_and_lf(path):
    """Parse brine concentration and loss fraction out of a Pareto-front filename.

    e.g. ``.../b50/pareto_front_0.5.csv`` -> (50.0, 0.5)."""
    b_num = re.search(r'b(\d+)', str(path))
    lf = re.search(r'_([\d.]+)\.csv$', str(path))

    return float(b_num.group(1)), float(lf.group(1))

if __name__ == "__main__":
    JPS.apply()  # project-wide matplotlib styling
    paths = search_path.rglob("*.csv")

    l_df = []
    for path in paths:
        label = get_brine_and_lf(path)
        df = pd.read_csv(path)
        l_df.append(df_label(label, df))

    # Keep only the brine=50 mol/m^3, loss-fraction=0.5 (50% extraction) case.
    b50_dfs = [(label, df) for label, df in l_df if label[0] == 50 and label[1] == 0.5]
    # Colour scale = residence time L/u_super [s]; normalise over the pooled data.
    all_L_over_u = pd.concat([df["L"] / df["u_super"] for _, df in b50_dfs])
    norm = mpl.colors.Normalize(vmin=all_L_over_u.min(), vmax=all_L_over_u.max())
    cmap = plt.cm.get_cmap("batlow")

    fig, ax = plt.subplots()

    # One scatter series per loss fraction; marker encodes extraction efficiency
    # (n_extr = 1 - loss_fraction), colour encodes residence time L/u_super.
    for label, df in b50_dfs:
        match label[1]:
            case 0.5:
                marker = "+"
                label = r"$n_{extr}=50\%$"
            case 0.1:
                marker = "*"
                label = r"$n_{extr}=90\%$"
            case 0.01:
                marker = "."
                label = r"$n_{extr}=99\%$"

        ax.scatter(df["SEC_J_per_mol"], df["Productivity_mol_per_m2_per_s"],
                   c=df["L"] / df["u_super"], cmap=cmap, norm=norm,
                   marker=marker, label=label)

    ax.set_xlabel("SEC")
    ax.set_ylabel("Productivity (mol/m^3 s)")
    ax.set_title("b50")
    plt.legend()
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="L/u (s)")
    plt.show()
