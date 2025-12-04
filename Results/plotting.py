import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from matplotlib import colormaps

colormaps.register(cm.batlow, name="batlow")
plt.style.use('natcomm_paper.mplstyle')

pretty_names = {
    "Da2_star": r"$Da_2^*$",
    "phi_star" : r"$\phi^*$",
    "Lambda_star": r"$\lambda^*$",
    "tau_star_break": r"$\tau_b$",
    "bed_utilization": r"$U_b$"
}

def scatter_plot(data: pd.DataFrame, x: str, y: str, c: str, filename: str):
    fig, ax = plt.subplots()

    if c is None:
        ax.scatter(data[x], data[y])
    else:
        sc = ax.scatter(data[x], data[y], c=data[c])
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(pretty_names[c])

    ax.set_xlabel(pretty_names[x])
    ax.set_ylabel(pretty_names[y])
    
    ax.legend()
    fig.savefig(filename)

if __name__ == "__main__":
    df = pd.read_csv("Sensitivity/Sobol/Good Ranges/sobol_samples_langmuir_05.csv")
    scatter_plot(df, 'Da2_star', 'tau_star_break', 'phi_star', 'test2.svg')