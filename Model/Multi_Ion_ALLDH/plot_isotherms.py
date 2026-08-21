"""Write the illustrative multi-ion ALLDH isotherms to Results as SVGs."""

from pathlib import Path

import JansPlottingStuff as JPS
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from Model.Multi_Ion_ALLDH import ALLDHIsotherm, log_reaction_activity
from utils.StreamData import Composition


Q_MAX = 3.98
RESULT_DIR = Path(__file__).resolve().parents[2] / "Results" / "Multi_Ion_ALLDH"


def _occupancy(model, m_li_cl, m_na_cl, m_mg_cl2):
    molalities = jnp.broadcast_arrays(m_li_cl, m_na_cl, m_mg_cl2)
    shape = molalities[0].shape
    composition = Composition(*(m.reshape(-1) for m in molalities))
    return np.asarray(jax.vmap(model)(composition)).reshape(shape) / Q_MAX


def _reaction_activity(n_h2o, m_li_cl, m_na_cl, m_mg_cl2):
    molalities = jnp.broadcast_arrays(m_li_cl, m_na_cl, m_mg_cl2)
    shape = molalities[0].shape
    composition = Composition(*(m.reshape(-1) for m in molalities))
    log_aq = jax.vmap(lambda c: log_reaction_activity(c, n_h2o))(composition)
    return np.exp(np.asarray(log_aq)).reshape(shape)


def _scenarios():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return (
        ("Water + LiCl", 0.0, 0.0, colors[0], "-"),
        ("+ 0.10 m NaCl", 0.10, 0.0, colors[3], "-"),
        (r"+ 0.05 m MgCl$_2$", 0.0, 0.05, colors[3], "--"),
        ("+ 0.50 m NaCl", 0.50, 0.0, colors[4], "-"),
        (r"+ 0.25 m MgCl$_2$", 0.0, 0.25, colors[4], "--"),
        (r"+ 0.50 m MgCl$_2$", 0.0, 0.50, colors[1], "--"),
        (r"+ 1.00 m MgCl$_2$", 0.0, 1.00, colors[5], "--"),
        (
            r"+ 0.50 m NaCl + 0.05 m MgCl$_2$",
            0.50,
            0.05,
            colors[2],
            "-.",
        ),
    )


def _plot(model, n_h2o, m_li_cl, filename, title, log_x=False):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for label, m_na_cl, m_mg_cl2, color, linestyle in _scenarios():
        ax.plot(
            m_li_cl,
            _occupancy(model, m_li_cl, m_na_cl, m_mg_cl2),
            label=label,
            color=color,
            linestyle=linestyle,
        )

    if log_x:
        ax.set_xscale("log")
    else:
        ax.plot(0.0, 0.0, "o", color="black", markersize=3, zorder=5)
        ax.annotate(
            "Pure water",
            xy=(0.0, 0.0),
            xytext=(0.08, 0.08),
            arrowprops={"arrowstyle": "-", "linewidth": 0.6},
            fontsize=10,
        )

    ax.axhline(1.0, color="0.45", linestyle=":", linewidth=0.8)
    ax.grid(which="major", color="0.75", linewidth=0.5, alpha=0.35)
    ax.set_xlim(float(m_li_cl[0]), float(m_li_cl[-1]))
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel(r"LiCl molality [mol kg$^{-1}$ water]", fontsize=14)
    ax.set_ylabel(r"Reversible capacity fraction $q/q_{\max}$", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(
        loc="upper left" if log_x else "lower right",
        ncol=2,
        fontsize=10,
        title=rf"Illustrative, unfitted: $K_i=12$, $n_{{H_2O}}={n_h2o:g}$, "
        r"$T=298.15$ K",
        title_fontsize=10,
    )
    fig.tight_layout()

    output = RESULT_DIR / filename
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_reaction_activity_linear():
    n_h2o = 2.0
    m_li_cl = np.linspace(0.0, 1.0, 401)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for label, m_na_cl, m_mg_cl2, color, linestyle in _scenarios():
        ax.plot(
            m_li_cl,
            _reaction_activity(n_h2o, m_li_cl, m_na_cl, m_mg_cl2),
            label=label,
            color=color,
            linestyle=linestyle,
        )

    ax.plot(0.0, 0.0, "o", color="black", markersize=3, zorder=5)
    ax.grid(which="major", color="0.75", linewidth=0.5, alpha=0.35)
    ax.set_xlim(float(m_li_cl[0]), float(m_li_cl[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"LiCl molality [mol kg$^{-1}$ water]", fontsize=14)
    ax.set_ylabel(r"Aqueous reaction activity $A_{aq}$", fontsize=14)
    ax.set_title("Multi-ion $A_{aq}$: full LiCl sweep", fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=10,
        title=rf"Pitzer activities: $n_{{H_2O}}={n_h2o:g}$, $T=298.15$ K",
        title_fontsize=10,
    )
    fig.tight_layout()

    output = RESULT_DIR / "multi_ion_aqueous_activity_linear.svg"
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_mg_surface():
    m_li_cl = np.geomspace(1e-3, 1.2, 241)
    m_mg_cl2 = np.linspace(0.0, 1.5, 201)
    li_grid, mg_grid = np.meshgrid(m_li_cl, m_mg_cl2)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.8),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    filled = None
    for ax, n_h2o in zip(axes, (1.0, 2.0), strict=True):
        model = ALLDHIsotherm(
            q_max=Q_MAX,
            log_K_i=np.log(12.0),
            n_h2o=n_h2o,
        )
        theta = _occupancy(model, li_grid, 0.0, mg_grid)
        filled = ax.contourf(
            li_grid,
            mg_grid,
            theta,
            levels=np.linspace(0.0, 1.0, 21),
            cmap="batlow",
        )
        contours = ax.contour(
            li_grid,
            mg_grid,
            theta,
            levels=(0.1, 0.5, 0.9),
            colors="0.2",
            linewidths=0.7,
        )
        ax.clabel(contours, fmt="%.1f", fontsize=7)
        ax.set_xscale("log")
        ax.set_title(rf"$n_{{H_2O}}={n_h2o:g}$")
        ax.set_xlabel(r"LiCl molality [mol kg$^{-1}$ water]")

    axes[0].set_ylabel(r"MgCl$_2$ molality [mol kg$^{-1}$ water]")
    fig.suptitle("MgCl$_2$--LiCl capacity map (no NaCl)")
    colorbar = fig.colorbar(filled, ax=axes, pad=0.02)
    colorbar.set_label(r"Reversible capacity fraction $q/q_{\max}$")

    output = RESULT_DIR / "multi_ion_mgcl2_surface.svg"
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_matched_background():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fixed_li_cl = 0.05
    comparisons = (
        (
            "Hold added chloride fixed",
            np.linspace(0.0, 3.0, 401),
            lambda x: (x, x / 2.0),
            r"Added background chloride [mol kg$^{-1}$ water]",
        ),
        (
            "Hold added ionic strength fixed",
            np.linspace(0.0, 4.5, 401),
            lambda x: (x, x / 3.0),
            r"Background ionic strength [mol kg$^{-1}$ water]",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    for ax, (title, coordinate, salt_paths, xlabel) in zip(
        axes, comparisons, strict=True
    ):
        m_na_cl, m_mg_cl2 = salt_paths(coordinate)
        for n_h2o, linestyle in ((1.0, "-"), (2.0, "--")):
            model = ALLDHIsotherm(
                q_max=Q_MAX,
                log_K_i=np.log(12.0),
                n_h2o=n_h2o,
            )
            ax.plot(
                coordinate,
                _occupancy(model, fixed_li_cl, m_na_cl, 0.0),
                color=colors[0],
                linestyle=linestyle,
                label=rf"NaCl, $n_{{H_2O}}={n_h2o:g}$",
            )
            ax.plot(
                coordinate,
                _occupancy(model, fixed_li_cl, 0.0, m_mg_cl2),
                color=colors[3],
                linestyle=linestyle,
                label=rf"MgCl$_2$, $n_{{H_2O}}={n_h2o:g}$",
            )

        ax.set_xlim(float(coordinate[0]), float(coordinate[-1]))
        ax.set_ylim(0.0, 0.80)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel(r"Reversible capacity fraction $q/q_{\max}$")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(r"NaCl versus MgCl$_2$ at $m_{LiCl}=0.05$ mol kg$^{-1}$")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output = RESULT_DIR / "multi_ion_matched_background.svg"
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_aqueous_activity():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fixed_li_cl = 0.05
    n_h2o = 2.0
    coordinate = np.linspace(0.0, 3.0, 401)
    comparisons = (
        (
            "Equal added chloride",
            coordinate,
            coordinate / 2.0,
            r"Added background chloride [mol kg$^{-1}$ water]",
        ),
        (
            "Equal added ionic strength",
            coordinate,
            coordinate / 3.0,
            r"Background ionic strength [mol kg$^{-1}$ water]",
        ),
    )

    baseline = float(_reaction_activity(n_h2o, fixed_li_cl, 0.0, 0.0))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    for ax, (title, m_na_cl, m_mg_cl2, xlabel) in zip(
        axes, comparisons, strict=True
    ):
        ax.plot(
            coordinate,
            _reaction_activity(n_h2o, fixed_li_cl, m_na_cl, 0.0),
            color=colors[0],
            label="NaCl background",
        )
        ax.plot(
            coordinate,
            _reaction_activity(n_h2o, fixed_li_cl, 0.0, m_mg_cl2),
            color=colors[3],
            linestyle="--",
            label=r"MgCl$_2$ background",
        )
        ax.axhline(
            baseline,
            color="0.45",
            linestyle=":",
            linewidth=0.8,
            label="LiCl only",
        )
        ax.set_yscale("log")
        ax.grid(which="major", color="0.75", linewidth=0.5, alpha=0.35)
        ax.set_xlim(float(coordinate[0]), float(coordinate[-1]))
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis="both", labelsize=11)

    axes[0].set_ylabel(r"Aqueous reaction activity $A_{aq}$", fontsize=14)
    axes[0].legend(
        loc="upper left",
        fontsize=10,
        title=rf"$m_{{LiCl}}={fixed_li_cl:g}$ m, $n_{{H_2O}}={n_h2o:g}$",
        title_fontsize=10,
    )
    fig.suptitle("Effect of NaCl and MgCl$_2$ on $A_{aq}$", fontsize=15)
    fig.tight_layout()

    output = RESULT_DIR / "multi_ion_aqueous_activity.svg"
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_reference_brines():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Table 6 of "Setting the Boundaries of the Analysis". Ranged Na/Mg
    # concentrations use their midpoint for this reduced three-salt projection.
    brines = (
        ("Atacama (3.65, 0.39)", (0.20, 0.30), True, 3.65, 0.39),
        ("Uyuni / Qinghai* (4.60, 0.69)", (0.04, 0.13), True, 4.60, 0.69),
        ("Salton Sea A (2.30, 0.0015)", (0.030, 0.030), False, 2.30, 0.0015),
        ("Salton Sea B (3.10, 0.049)", (0.042, 0.042), False, 3.10, 0.049),
        ("Smackover* (3.45, 0.076)", (0.016, 0.097), True, 3.45, 0.076),
        ("Alberta Devonian* (2.35, 0.125)", (0.005, 0.011), False, 2.35, 0.125),
    )
    m_li_cl = np.geomspace(1e-3, 0.5, 401)

    n_h2o = 2.0
    model = ALLDHIsotherm(
        q_max=Q_MAX,
        log_K_i=np.log(12.0),
        n_h2o=n_h2o,
    )
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    for (label, li_values, is_range, m_na_cl, m_mg_cl2), color in zip(
        brines, colors
    ):
        ax.plot(
            m_li_cl,
            _occupancy(model, m_li_cl, m_na_cl, m_mg_cl2),
            color=color,
            label=label,
        )
        reported_li = (
            np.geomspace(*li_values, 41) if is_range else np.asarray(li_values)
        )
        reported_theta = _occupancy(model, reported_li, m_na_cl, m_mg_cl2)
        if is_range:
            ax.plot(
                reported_li,
                reported_theta,
                color=color,
                linewidth=5,
                alpha=0.45,
                solid_capstyle="round",
            )
        else:
            ax.plot(reported_li, reported_theta, "o", color=color, markersize=4)

    ax.set_xscale("log")
    ax.set_xlim(float(m_li_cl[0]), float(m_li_cl[-1]))
    ax.set_ylim(0.0, 1.02)
    ax.set_title(rf"$n_{{H_2O}}={n_h2o:g}$")
    ax.set_xlabel(r"LiCl molality [mol kg$^{-1}$ water]", fontsize=14)
    ax.set_ylabel(r"Reversible capacity fraction $q/q_{\max}$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=2,
        fontsize=9,
        title=r"Representative Table 6 background (Na, Mg; mol L$^{-1}$)",
        title_fontsize=9,
    )
    fig.suptitle("Table 6 archetypes - qualitative three-salt model")
    fig.text(
        0.5,
        0.005,
        "Heavy segments/points show reported Li ranges/values; * includes inferred "
        "Table 6 data; ranged Na/Mg use midpoints.\nIllustrative only: mol/L "
        "values are copied as mol/kg-water inputs, not converted.\nOnly Li/Na/Mg "
        "are retained as chloride salts; omitted ions mean Cl is model-implied. "
        r"$K_i=12$, $T=298.15$ K, unfitted.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.25, 1.0, 0.94))

    output = RESULT_DIR / "multi_ion_table6_brines.svg"
    fig.savefig(output)
    plt.close(fig)
    return output


def main():
    JPS.apply()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for n_h2o in (2.0, 1.0):
        model = ALLDHIsotherm(
            q_max=Q_MAX,
            log_K_i=np.log(12.0),
            n_h2o=n_h2o,
        )
        suffix = "" if n_h2o == 2.0 else "_n_h2o_1"
        outputs.extend(
            (
                _plot(
                    model,
                    n_h2o,
                    np.linspace(0.0, 1.0, 401),
                    f"multi_ion_isotherms_linear{suffix}.svg",
                    "Multi-ion ALLDH isotherms: full LiCl sweep",
                ),
                _plot(
                    model,
                    n_h2o,
                    np.geomspace(1e-4, 0.1, 401),
                    f"multi_ion_isotherms_trace{suffix}.svg",
                    "Multi-ion ALLDH isotherms: trace-LiCl window",
                    log_x=True,
                ),
            )
        )
    outputs.extend(
        (
            _plot_mg_surface(),
            _plot_matched_background(),
            _plot_aqueous_activity(),
            _plot_reaction_activity_linear(),
            _plot_reference_brines(),
        )
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
