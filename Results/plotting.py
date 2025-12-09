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
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()

    if c is None:
        ax.scatter(data[x], data[y])
    else:
        sc = ax.scatter(data[x], data[y], c=data[c])
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(pretty_names[c])

    ax.set_xlabel(pretty_names[x])
    ax.set_ylabel(pretty_names[y])
    
    # ax.legend()
    fig.savefig(filename)
    plt.close(fig)

def scatter_plot_data(x, y, c, xlabel, ylabel, clabel, filename):
    fig, ax = plt.subplots()

    if c is None:
        ax.scatter(x, y)
    else:
        sc = ax.scatter(x, y, c=c)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(pretty_names[clabel])

    ax.set_xlabel(pretty_names[xlabel])
    ax.set_ylabel(pretty_names[ylabel])
    
    ax.legend()
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("Sensitivity/Sobol/1st Order/sobol_samples_langmuir_LDF_05.csv")
    df = df[
        ~((df['phi_star'] > 7) & 
        (df['Da2_star'] > 7 ) &
        (df['Lambda_star'] > 7 ) &
        ((df['tau_star_break'] > .2) & (df['tau_star_break'] < .35 )) &
        ((df['bed_utilization'] > .05) & (df['bed_utilization'] < .1 )))
    ]

    scatter_plot(df, 'Da2_star', 'tau_star_break', 'phi_star', 'Da_tau_phi_05.svg')
    scatter_plot(df, 'Da2_star', 'bed_utilization', 'phi_star', 'Da_bed_phi_05.svg')
    
    scatter_plot(df, 'phi_star', 'tau_star_break', 'Da2_star', 'phi_tau_Da_05.svg')
    scatter_plot(df, 'phi_star', 'bed_utilization', 'Da2_star', 'phi_bed_Da_05.svg')
    
    scatter_plot(df, 'Da2_star', 'tau_star_break', 'Lambda_star', 'Da_tau_Lambda_05.svg')
    scatter_plot(df, 'Da2_star', 'bed_utilization', 'Lambda_star', 'Da_bed_Lambda_05.svg')
    
    scatter_plot(df, 'phi_star', 'tau_star_break', 'Lambda_star', 'phi_tau_Lambda_05.svg')
    scatter_plot(df, 'phi_star', 'bed_utilization', 'Lambda_star', 'phi_bed_Lambda_05.svg')

    scatter_plot(df, 'Lambda_star', 'tau_star_break', 'Da2_star', 'Lambda_tau_Da_05.svg')
    scatter_plot(df, 'Lambda_star', 'tau_star_break', 'phi_star', 'Lambda_tau_phi_05.svg')
    scatter_plot(df, 'Lambda_star', 'bed_utilization', 'phi_star', 'Lambda_bed_phi_05.svg')
    scatter_plot(df, 'Lambda_star', 'bed_utilization', 'Da2_star', 'Lambda_bed_Da_05.svg')
     
    df = pd.read_csv("Sensitivity/Sobol/1st Order/sobol_samples_langmuir_LDF_50.csv")
    scatter_plot(df, 'Da2_star', 'tau_star_break', 'phi_star', 'Da_tau_phi_50.svg')
    scatter_plot(df, 'Da2_star', 'bed_utilization', 'phi_star', 'Da_bed_phi_50.svg')
    
    scatter_plot(df, 'phi_star', 'tau_star_break', 'Da2_star', 'phi_tau_Da_50.svg')
    scatter_plot(df, 'phi_star', 'bed_utilization', 'Da2_star', 'phi_bed_Da_50.svg')
    
    scatter_plot(df, 'Da2_star', 'tau_star_break', 'Lambda_star', 'Da_tau_Lambda_50.svg')
    scatter_plot(df, 'Da2_star', 'bed_utilization', 'Lambda_star', 'Da_bed_Lambda_50.svg')
    
    scatter_plot(df, 'phi_star', 'tau_star_break', 'Lambda_star', 'phi_tau_Lambda_50.svg')
    scatter_plot(df, 'phi_star', 'bed_utilization', 'Lambda_star', 'phi_bed_Lambda_50.svg')
    
    scatter_plot(df, 'Lambda_star', 'tau_star_break', 'Da2_star', 'Lambda_tau_Da_50.svg')
    scatter_plot(df, 'Lambda_star', 'tau_star_break', 'phi_star', 'Lambda_tau_phi_50.svg')
    scatter_plot(df, 'Lambda_star', 'bed_utilization', 'phi_star', 'Lambda_bed_phi_50.svg')
    scatter_plot(df, 'Lambda_star', 'bed_utilization', 'Da2_star', 'Lambda_bed_Da_50.svg')
    