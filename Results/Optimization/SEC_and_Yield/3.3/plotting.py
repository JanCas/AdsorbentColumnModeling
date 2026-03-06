import JansPlottingStuff as JPS
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re
from typing import NamedTuple

search_path = Path(".")

class df_label(NamedTuple):
    label: tuple
    df: pd.DataFrame

def get_brine_and_lf(path):
    b_num = re.search(r'b(\d+)', str(path))
    lf = re.search(r'_([\d.]+)\.csv$', str(path))

    return float(b_num.group(1)), float(lf.group(1))

if __name__ == "__main__":
    JPS.apply()
    paths = search_path.rglob("*.csv")

    l_df = []
    for path in paths:
        label = get_brine_and_lf(path)
        df = pd.read_csv(path)
        l_df.append(df_label(label, df))
    
    b50_dfs = [(label, df) for label, df in l_df if label[0] == 50 and label[1] == 0.5]
    all_L_over_u = pd.concat([df["L"] / df["u_super"] for _, df in b50_dfs])
    norm = mpl.colors.Normalize(vmin=all_L_over_u.min(), vmax=all_L_over_u.max())
    cmap = plt.cm.get_cmap("batlow")

    fig, ax = plt.subplots()

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
