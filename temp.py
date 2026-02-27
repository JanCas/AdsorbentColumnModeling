import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import numpy as np
import JansPlottingStuff as JPS
import pandas as pd

def heatmap(df, xcol, ycol, zcol):

     # bin edges
     xcol_bins = np.linspace(df[xcol].min(), df[xcol].max(), 31)
     ycol_bins = np.linspace(df[ycol].min(), df[ycol].max(), 31)

     # bin and average
     df["xcol_bin"] = pd.cut(df[xcol], bins=xcol_bins)
     df["ycol_bin"] = pd.cut(df[ycol], bins=ycol_bins)
     heat = df.groupby(["ycol_bin", "xcol_bin"])[zcol].mean().unstack()

     # plot
     fig, ax = plt.subplots(figsize=(8, 6))
     im = ax.pcolormesh(
         xcol_bins, ycol_bins, heat.values,
         shading="flat"
     )
     ax.set_xlabel(xcol)
     ax.set_ylabel(ycol)
     fig.colorbar(im, ax=ax, label=zcol)
     plt.tight_layout()

def contour_heatmap(
    df,
    x_col,
    y_col,
    z_col,
    n_bins=30,
    smooth=3,
    levels=15,
    contour_levels=8,
    figsize=(8, 6),
    x_label=None,
    y_label=None,
    z_label=None,
    contour_fmt="%.2f",
    x_log=False,
    reverse_cmap=False,
    figname=None
):

    if x_log:
        x_bins = np.logspace(np.log10(df[x_col].min()), np.log10(df[x_col].max()), n_bins + 1)
    else:
        x_bins = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
    y_bins = np.linspace(df[y_col].min(), df[y_col].max(), n_bins + 1)

    df = df.copy()
    df["_xbin"] = pd.cut(df[x_col], bins=x_bins)
    df["_ybin"] = pd.cut(df[y_col], bins=y_bins)
    heat = df.groupby(["_ybin", "_xbin"])[z_col].mean().unstack().values.astype(float)

    if smooth:
        heat = uniform_filter(heat, size=smooth)

    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    Xm, Ym = np.meshgrid(x_centers, y_centers)

    cmap = plt.rcParams["image.cmap"]
    if reverse_cmap:
        cmap = 'cmc.batlow_r'

    plt.figure(figsize=figsize)
    cf = plt.contourf(Xm, Ym, heat, levels=levels, cmap=cmap)
    cs = plt.contour(Xm, Ym, heat, colors="white", linewidths=1.5, levels=contour_levels)
    plt.clabel(cs, inline=True, fontsize=8, fmt=contour_fmt)
    plt.colorbar(cf, label=z_label or z_col)
    plt.xlabel(x_label or x_col)
    plt.ylabel(y_label or y_col)
    if x_log:
        plt.xscale("log")
    plt.tight_layout()

    if figname:
        plt.savefig(figname)


if __name__ == "__main__":
    JPS.apply()
    df = pd.read_csv("sobol_cycle_samples.csv")


    #heatmap(df, 'Da', 'theta', 'productivity')
    #heatmap(df, 'Da', 'epsilon', 'sec_star')

    contour_heatmap(df, 'Da', 'theta', 'productivity', x_label=r"$Da$", y_label=r"$\theta$", z_label="Productivity",n_bins=15, levels=25, figname="Da_theta_prod.svg")
    contour_heatmap(df, 'Da', 'epsilon', 'sec_star', x_label=r"$Da$", y_label=r"$\epsilon$", z_label=r"$SEC^{*}$",reverse_cmap=True, contour_levels=8, n_bins=15, levels=25, contour_fmt="%d", figname="Da_epsilon_sec.svg")
    contour_heatmap(df, 'Da', 'epsilon', 'sec_star', x_label=r"$Da$", y_label=r"$\epsilon$", z_label=r"$SEC^{*}$",reverse_cmap=True, contour_levels=8, n_bins=15, levels=25, contour_fmt="%d", figname="Da_epsilon_sec_log.svg", x_log=True)
    contour_heatmap(df, 'Da', 'theta', 'sec_star', x_label=r"$Da$", y_label=r"$\theta$", z_label=r"$SEC^{*}$",reverse_cmap=True, contour_levels=8, n_bins=15, levels=25, contour_fmt="%d", figname="Da_theta_sec.svg")


