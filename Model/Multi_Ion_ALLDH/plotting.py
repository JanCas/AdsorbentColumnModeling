"""Matplotlib plots for the Jiang (2020) multi-ion ALLDH validation.

Kept separate from `jiang_validation.py` so the simulation module stays free of
matplotlib and `JansPlottingStuff`. Styling comes entirely from `JPS.apply()`;
nothing here overrides fonts, colours, grid, or line weights.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

import JansPlottingStuff as JPS

_log = logging.getLogger(__name__)

# The style's default figsize is a single-panel size and cannot carry a 2x2
# grid; this is the double-column width the same page layout expects. Layout,
# not style.
_FIGSIZE = (7.1, 5.6)

_PANEL_LABELS = ("a", "b", "c", "d")


def apply_style():
    """Apply Jan's matplotlib style. Call once before building a figure."""
    _log.debug("applying JansPlottingStuff style")
    JPS.apply()


def _cycle_colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _label_panels(axes):
    for axis, label in zip(axes, _PANEL_LABELS, strict=False):
        axis.set_title(
            label, loc="left", fontweight="bold", x=-0.16, y=1.02
        )


def _paired_legend(axis, handles, labels, **kwargs):
    """One entry per condition, each drawn as its model line plus its markers."""
    axis.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        **kwargs,
    )


def _plot_equilibrium(axis, study, fit):
    concentration = np.asarray(fit["curve_concentration_mg_L"])
    axis.plot(
        study["Isotherm"]["C_e"],
        study["Isotherm"]["q_e"],
        linestyle="none",
        marker="o",
        color="black",
        zorder=3,
        label="Jiang data",
    )
    axis.plot(
        concentration, fit["curve_langmuir_mg_g"], label="Table 2 Langmuir"
    )
    axis.plot(
        concentration,
        fit["curve_activity_model_mg_g"],
        linestyle="--",
        label="activity model",
    )
    axis.plot(
        concentration,
        fit["curve_activity_model_mg_zero_mg_g"],
        linestyle=":",
        label=r"activity model, Mg$^{2+}=0$",
    )
    axis.set(
        xlabel=r"Li [mg L$^{-1}$]",
        ylabel=r"uptake [mg g$^{-1}$]",
        xlim=(0.0, None),
        ylim=(0.0, None),
    )
    axis.legend(loc="lower right")


def _plot_breakthrough(axis, series, xlabel=r"time [min]"):
    """Draw model lines and their paired data markers for one comparison."""
    handles, labels = [], []
    for color, item in zip(_cycle_colors(), series, strict=False):
        (line,) = axis.plot(
            item["model_x"], item["model_y"], color=color, zorder=3
        )
        (points,) = axis.plot(
            item["data_x"],
            item["data_y"],
            linestyle="none",
            marker="o",
            color=color,
            zorder=2,
        )
        handles.append((line, points))
        labels.append(item["label"])
    axis.set(
        xlabel=xlabel,
        ylabel=r"$C_\mathrm{out}/C_\mathrm{in}$",
        xlim=(0.0, None),
        ylim=(0.0, 1.03),
    )
    _paired_legend(axis, handles, labels, loc="lower right")


def _condition_series(study, condition_rows, comparison, group_name, value_name, units):
    curves = study["ColumnExperiments"][group_name]["Curves"]
    series = []
    for curve_index, curve in enumerate(curves):
        rows = [
            row
            for row in condition_rows
            if row["comparison"] == comparison
            and row["curve_index"] == curve_index
        ]
        value = float(curve[value_name])
        value = value * 100.0 if comparison == "height" else value
        series.append(
            {
                "label": f"{value:g} {units}",
                "model_x": [row["time_min"] for row in rows],
                "model_y": [row["outlet_C_over_C0"] for row in rows],
                "data_x": curve["time"],
                "data_y": curve["C_out/C_in"],
            }
        )
    return series


def _flow_series(study, trajectory_rows):
    series = []
    for curve in study["ColumnExperiments"]["BreakthroughCurves"]["Curves"]:
        flow_bv_h = float(curve["flowrate"])
        rows = [
            row
            for row in trajectory_rows
            if row["source_flow_BV_h"] == flow_bv_h
        ]
        series.append(
            {
                "label": f"{rows[0]['source_flow_mL_min']:g} mL min$^{{-1}}$",
                "model_x": [row["time_min"] for row in rows],
                "model_y": [row["outlet_C_over_C0"] for row in rows],
                "data_x": np.asarray(curve["BV"]) * 60.0 / flow_bv_h,
                "data_y": curve["C_out/C_in"],
            }
        )
    return series


def plot_jiang_validation(study, fit, trajectory_rows, condition_rows, output):
    """Write the four-panel equilibrium and breakthrough validation figure."""
    apply_style()
    _log.info("building Jiang validation figure -> %s", output)
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE, layout="constrained")
    axes = axes.ravel()

    _plot_equilibrium(axes[0], study, fit)
    axes[0].set_title("Equilibrium")

    _plot_breakthrough(
        axes[1],
        _condition_series(
            study, condition_rows, "height", "HeightCurves", "height", "cm"
        ),
    )
    axes[1].set_title("Bed height")

    _plot_breakthrough(
        axes[2],
        _condition_series(
            study,
            condition_rows,
            "influent_concentration",
            "InfluentConcentrationCurves",
            "influent_concentration",
            r"mg L$^{-1}$",
        ),
    )
    axes[2].set_title("Influent lithium")

    _plot_breakthrough(axes[3], _flow_series(study, trajectory_rows))
    axes[3].set_title("Flow rate")

    _label_panels(axes)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"))
    plt.close(fig)
    return output
