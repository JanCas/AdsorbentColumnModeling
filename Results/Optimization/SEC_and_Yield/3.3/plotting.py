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

    all_L = pd.concat([df["des_thresh"] for _, df in l_df])
    norm = mpl.colors.Normalize(vmin=all_L.min(), vmax=all_L.max())
    cmap = plt.cm.get_cmap("batlow")

    fig, ax = plt.subplots(1,2, sharey=True)

    for label, df in l_df:
        ax_used = None
        match label[0]:
            case 50:
                ax_used = ax[0]
            case 25:
                ax_used = ax[1]

        match label[1]:
            case 0.5:
                marker = "+"
                label = "lf=50"
            case 0.1:
                marker = "*"
                label = "lf=10"
            case 0.01:
                marker = "."
                label = "lf=1"

        ax_used.scatter(df["SEC_J_per_mol"], df["Productivity_mol_per_m2_per_s"],
                        c=df["des_thresh"], cmap=cmap, norm=norm, marker=marker, label=label)

    ax[0].set_xlabel("SEC")
    ax[1].set_xlabel("SEC")
    ax[0].set_ylabel("Prodcutivity")

    ax[0].set_title("b50")
    ax[1].set_title("b25")
    plt.legend()
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="des_thresh")
    plt.show()
