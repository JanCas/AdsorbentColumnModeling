"""Calibrate the activity isotherm and run the Jiang column cases.

Jiang's fixed MgCl2 matrix identifies a conditional activity isotherm. It
cannot independently identify the water stoichiometry, so ``n_h2o`` remains
an explicit calibration input (one by default).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import numpy as np

from Model.Multi_Ion_ALLDH import (
    ALLDHIsotherm,
    AqueousConcentration,
    ColumnParams,
    initial_state,
    simulate_column,
)
from Model.Multi_Ion_ALLDH.plotting import plot_jiang_validation
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "LiteratureReview" / "isotherm_kinetics.json"
RESULT_DIR = ROOT / "Results" / "Multi_Ion_ALLDH"
STUDY_KEY = "jiangApplicationConcentrationDependentHSDM2020"

LI_MW = 6.94
MG_MW = 24.305
LICL_MW = 42.394
MGCL2_MW = 95.211

# Jiang, Separation and Purification Technology 241 (2020) 116682.
BRINE_DENSITY_KG_L = 1.253
MG_G_L = 100.0
PARTICLE_DENSITY_KG_M3 = 1378.7
# Pooled least-squares fit across Jiang's three flow-rate curves.
K_PSO_KG_MOL_S = 1.00148162948e-4


def load_jiang(path: Path = DATA_PATH):
    with path.open() as stream:
        return json.load(stream)[STUDY_KEY]


def concentration_from_elemental_li(li_mg_l):
    """Convert Jiang elemental-ion concentrations to salt molarities.

    Scalars and arrays are both accepted; an array input returns array fields
    that ``jax.vmap`` can map over.
    """
    li_mg_l = np.asarray(li_mg_l, dtype=float)
    n_li_l = li_mg_l / (1000.0 * LI_MW)
    n_mg_l = np.full_like(n_li_l, MG_G_L / MG_MW)
    water_kg_l = BRINE_DENSITY_KG_L - (
        n_li_l * LICL_MW + n_mg_l * MGCL2_MW
    ) / 1000.0
    if np.any(water_kg_l <= 0.0):
        raise ValueError("calculated Jiang brine water inventory is nonpositive")
    return (
        AqueousConcentration(
            n_li_l * 1000.0, np.zeros_like(n_li_l), n_mg_l * 1000.0
        ),
        water_kg_l * 1000.0,
    )


def jiang_langmuir_mg_g(study, li_mg_l):
    """Evaluate Jiang's Table 2 Langmuir fit in its native units."""
    fit = study["Isotherm"]["FitParameters"]
    li_mg_l = np.asarray(li_mg_l, dtype=float)
    affinity = fit["K_s"] * li_mg_l
    return fit["q_max"] * affinity / (1.0 + affinity)


def _isotherm_grid(c_max, n_points=400):
    """Return a dense Li grid [mg/L] that resolves the low-concentration knee.

    A purely linear grid under-samples the steep rise below ~100 mg/L, which is
    what makes a seven-point plot look piecewise linear.
    """
    return np.unique(
        np.concatenate(
            (
                np.linspace(0.0, c_max, n_points),
                np.geomspace(c_max * 1.0e-4, c_max, n_points),
            )
        )
    )


def _isotherm_curves(study, model, n_points=400):
    """Evaluate both isotherms on a dense grid for smooth plotting."""
    grid = _isotherm_grid(
        float(np.max(np.asarray(study["Isotherm"]["C_e"], dtype=float))),
        n_points,
    )
    concentration, water = concentration_from_elemental_li(grid)
    brine = np.asarray(jax.vmap(model)(concentration.molality(water))) * LI_MW
    mg_zero = (
        np.asarray(
            jax.vmap(model)(
                AqueousConcentration(
                    grid / LI_MW, np.zeros_like(grid), np.zeros_like(grid)
                ).molality(1000.0)
            )
        )
        * LI_MW
    )
    return {
        "curve_concentration_mg_L": grid,
        "curve_langmuir_mg_g": jiang_langmuir_mg_g(study, grid),
        "curve_activity_model_mg_g": brine,
        "curve_activity_model_mg_zero_mg_g": mg_zero,
    }


def calibrate_isotherm(study, n_h2o=1.0, reference_li_mg_l=350.0):
    """Map Jiang's Langmuir fit to activities at the column feed."""
    fit = study["Isotherm"]["FitParameters"]
    if fit["m"] != 1.0 or fit.get("K_L_units") != "L/mg":
        raise ValueError("Jiang isotherm must be the Table 2 Langmuir fit")
    q_max_mg_g = float(fit["q_max"])
    reference_c, reference_water = concentration_from_elemental_li(
        reference_li_mg_l
    )
    reference = reference_c.molality(reference_water)
    log_a_licl, log_a_water = map(float, reference.log_activities())
    log_activity_reference = log_a_licl + n_h2o * log_a_water
    log_k = np.log(fit["K_s"] * reference_li_mg_l) - log_activity_reference
    model = ALLDHIsotherm(q_max_mg_g / LI_MW, log_k, n_h2o)

    concentration = np.asarray(study["Isotherm"]["C_e"], dtype=float)
    target = jiang_langmuir_mg_g(study, concentration)
    predicted = np.asarray(
        [
            float(model(c.molality(water)) * LI_MW)
            for c, water in map(concentration_from_elemental_li, concentration)
        ]
    )
    mg_zero = np.asarray(
        [
            float(
                model(
                    AqueousConcentration(li_mg_l / LI_MW, 0.0, 0.0).molality(
                        1000.0
                    )
                )
                * LI_MW
            )
            for li_mg_l in concentration
        ]
    )
    residual = predicted - target
    sse = float(np.sum(residual**2))
    r2 = 1.0 - sse / float(np.sum((target - target.mean()) ** 2))
    projection = {
        "q_max_mg_g": q_max_mg_g,
        "q_max_mol_kg": q_max_mg_g / LI_MW,
        "log_K_i": float(log_k),
        "K_i": float(np.exp(log_k)),
        "n_h2o": float(n_h2o),
        "reference_li_mg_L": float(reference_li_mg_l),
        "projection_sse_mg2_g2": sse,
        "projection_r2": r2,
        "concentration_mg_L": concentration,
        "langmuir_mg_g": target,
        "activity_model_mg_g": predicted,
        "activity_model_mg_zero_mg_g": mg_zero,
    }
    projection.update(_isotherm_curves(study, model))
    return model, projection


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _simulate_case(params, times):
    y0 = initial_state(
        params,
        AqueousConcentration(0.0, 0.0, params.inlet.c_MgCl2),
        q0=0.0,
    )
    solution = simulate_column(params, float(times[-1]), y0=y0, save_ts=times)
    outlet_ratio = np.asarray(solution.ys.c_LiCl[:, -1]) / float(
        params.inlet.c_LiCl
    )
    outlet_composition = AqueousConcentration(
        solution.ys.c_LiCl[:, -1],
        solution.ys.c_NaCl[:, -1],
        solution.ys.c_MgCl2[:, -1],
    ).molality(params.water_mass_concentration)
    outlet_activity = np.asarray(
        jax.vmap(lambda comp: comp.li_cl_activity())(outlet_composition)
    )
    mean_q = np.asarray(solution.ys.q).mean(axis=1) * LI_MW
    return solution, outlet_ratio, outlet_activity, mean_q


def _run_columns(study, model, output_dir, n_cells, n_save):
    properties = study["ColumnExperiments"]["Properties"]
    length = float(properties["Length"])
    diameter = float(properties["Diameter"])
    epsilon = float(properties["porosity"])
    area = np.pi * (diameter / 2.0) ** 2
    bed_volume = area * length
    feed, water_mass_concentration = concentration_from_elemental_li(350.0)
    data_rows = []
    trajectory_rows = []
    metrics = []

    curves = study["ColumnExperiments"]["BreakthroughCurves"]["Curves"]
    for curve in curves:
        flow_bv_h = float(curve["flowrate"])
        flow_m3_s = flow_bv_h * bed_volume / 3600.0
        source_flow = flow_m3_s * 60.0e6
        superficial_velocity = flow_m3_s / area
        max_bv = max(0.0, max(curve["BV"]))
        bv = np.linspace(0.0, max_bv, n_save)
        times = bv * bed_volume / flow_m3_s
        params = ColumnParams(
            L=length,
            eps=epsilon,
            rho_p=PARTICLE_DENSITY_KG_M3,
            water_mass_concentration=water_mass_concentration,
            u_s=superficial_velocity,
            k2=K_PSO_KG_MOL_S,
            inlet=feed,
            isotherm=model,
            N=n_cells,
        )
        solution, outlet_ratio, outlet_activity, mean_q = _simulate_case(
            params, times
        )

        observed_bv = np.maximum(np.asarray(curve["BV"], dtype=float), 0.0)
        observed = np.asarray(curve["C_out/C_in"], dtype=float)
        predicted = np.interp(observed_bv, bv, outlet_ratio)
        error = predicted - observed
        metrics.append(
            {
                "source_flow_mL_min": source_flow,
                "source_flow_BV_h": flow_bv_h,
                "rmse_C_over_C0": float(np.sqrt(np.mean(error**2))),
                "mae_C_over_C0": float(np.mean(np.abs(error))),
            }
        )
        data_rows.extend(
            {
                "source_flow_mL_min": source_flow,
                "source_flow_BV_h": flow_bv_h,
                "BV": raw_bv,
                "C_out_over_C_in": ratio,
            }
            for raw_bv, ratio in zip(curve["BV"], observed, strict=True)
        )
        trajectory_rows.extend(
            {
                "source_flow_mL_min": source_flow,
                "source_flow_BV_h": flow_bv_h,
                "BV": value_bv,
                "time_min": time_s / 60.0,
                "outlet_c_LiCl_mol_m3": outlet,
                "outlet_C_over_C0": ratio,
                "outlet_a_LiCl": activity,
                "mean_q_mg_g": loading,
            }
            for value_bv, time_s, outlet, ratio, activity, loading in zip(
                bv,
                solution.ts,
                solution.ys.c_LiCl[:, -1],
                outlet_ratio,
                outlet_activity,
                mean_q,
                strict=True,
            )
        )
        np.savez_compressed(
            output_dir / f"jiang_pso_flow_{source_flow:g}_mL_min.npz",
            time_s=np.asarray(solution.ts),
            BV=bv,
            c_LiCl_mol_m3=np.asarray(solution.ys.c_LiCl),
            c_NaCl_mol_m3=np.asarray(solution.ys.c_NaCl),
            c_MgCl2_mol_m3=np.asarray(solution.ys.c_MgCl2),
            q_mol_kg_sorbent=np.asarray(solution.ys.q),
            cumulative_Li_out_mol_m2=np.asarray(solution.ys.cumulative_Li_out),
        )

    _write_csv(output_dir / "jiang_pso_flow_data.csv", data_rows)
    _write_csv(output_dir / "jiang_pso_flow_trajectories.csv", trajectory_rows)
    return (
        metrics,
        trajectory_rows,
        feed,
        water_mass_concentration,
    )


def _run_condition_columns(study, model, output_dir, n_cells, n_save):
    columns = study["ColumnExperiments"]
    diameter = float(columns["Properties"]["Diameter"])
    epsilon = float(columns["Properties"]["porosity"])
    area = np.pi * (diameter / 2.0) ** 2
    data_rows = []
    trajectory_rows = []
    metrics = []

    for comparison, group_name in (
        ("height", "HeightCurves"),
        ("influent_concentration", "InfluentConcentrationCurves"),
    ):
        for curve_index, curve in enumerate(columns[group_name]["Curves"]):
            length = float(curve["height"])
            flow_ml_min = float(curve["flowrate"])
            influent_mg_l = float(curve["influent_concentration"])
            flow_m3_s = flow_ml_min * 1.0e-6 / 60.0
            feed, water_mass_concentration = concentration_from_elemental_li(
                influent_mg_l
            )
            time_min = np.linspace(0.0, max(curve["time"]), n_save)
            times = time_min * 60.0
            params = ColumnParams(
                L=length,
                eps=epsilon,
                rho_p=PARTICLE_DENSITY_KG_M3,
                water_mass_concentration=water_mass_concentration,
                u_s=flow_m3_s / area,
                k2=K_PSO_KG_MOL_S,
                inlet=feed,
                isotherm=model,
                N=n_cells,
            )
            solution, outlet_ratio, outlet_activity, mean_q = _simulate_case(
                params, times
            )
            observed_time = np.asarray(curve["time"], dtype=float)
            observed = np.asarray(curve["C_out/C_in"], dtype=float)
            error = np.interp(observed_time, time_min, outlet_ratio) - observed
            metrics.append(
                {
                    "comparison": comparison,
                    "height_m": length,
                    "source_flow_mL_min": flow_ml_min,
                    "influent_Li_mg_L": influent_mg_l,
                    "rmse_C_over_C0": float(np.sqrt(np.mean(error**2))),
                    "mae_C_over_C0": float(np.mean(np.abs(error))),
                }
            )
            data_rows.extend(
                {
                    "comparison": comparison,
                    "curve_index": curve_index,
                    "height_m": length,
                    "source_flow_mL_min": flow_ml_min,
                    "influent_Li_mg_L": influent_mg_l,
                    "time_min": time,
                    "C_out_over_C_in": ratio,
                }
                for time, ratio in zip(observed_time, observed, strict=True)
            )
            trajectory_rows.extend(
                {
                    "comparison": comparison,
                    "curve_index": curve_index,
                    "height_m": length,
                    "source_flow_mL_min": flow_ml_min,
                    "influent_Li_mg_L": influent_mg_l,
                    "time_min": time,
                    "outlet_c_LiCl_mol_m3": outlet,
                    "outlet_C_over_C0": ratio,
                    "outlet_a_LiCl": activity,
                    "mean_q_mg_g": loading,
                }
                for time, outlet, ratio, activity, loading in zip(
                    time_min,
                    solution.ys.c_LiCl[:, -1],
                    outlet_ratio,
                    outlet_activity,
                    mean_q,
                    strict=True,
                )
            )
            value = length * 100.0 if comparison == "height" else influent_mg_l
            units = "cm" if comparison == "height" else "mg_L"
            np.savez_compressed(
                output_dir / f"jiang_pso_{comparison}_{value:g}_{units}.npz",
                time_s=np.asarray(solution.ts),
                c_LiCl_mol_m3=np.asarray(solution.ys.c_LiCl),
                c_NaCl_mol_m3=np.asarray(solution.ys.c_NaCl),
                c_MgCl2_mol_m3=np.asarray(solution.ys.c_MgCl2),
                q_mol_kg_sorbent=np.asarray(solution.ys.q),
                cumulative_Li_out_mol_m2=np.asarray(
                    solution.ys.cumulative_Li_out
                ),
            )

    _write_csv(output_dir / "jiang_pso_condition_data.csv", data_rows)
    _write_csv(output_dir / "jiang_pso_condition_trajectories.csv", trajectory_rows)
    return metrics, trajectory_rows


def run(
    output_dir=RESULT_DIR,
    n_h2o=1.0,
    n_cells=40,
    n_save=401,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study = load_jiang()
    model, fit = calibrate_isotherm(study, n_h2o)
    metrics, trajectories, feed, water_mass_concentration = _run_columns(
        study, model, output_dir, n_cells, n_save
    )
    condition_metrics, condition_trajectories = _run_condition_columns(
        study, model, output_dir, n_cells, n_save
    )
    feed_molality = feed.molality(water_mass_concentration)
    feed_a_licl, feed_a_water = map(float, feed_molality.activities())
    summary = {
        key: value
        for key, value in fit.items()
        if not isinstance(value, np.ndarray)
    }
    summary.update(
        {
            "jiang_temperature_K": study["Isotherm"]["T"],
            "pitzer_temperature_K": 298.15,
            "feed_c_LiCl_mol_m3": float(feed.c_LiCl),
            "feed_c_MgCl2_mol_m3": float(feed.c_MgCl2),
            "feed_a_LiCl": feed_a_licl,
            "feed_a_water": feed_a_water,
            "water_mass_concentration_kg_m3": water_mass_concentration,
            "q_eq_feed_mg_g": float(model(feed_molality) * LI_MW),
            "k2_kg_sorbent_mol_Li_s": K_PSO_KG_MOL_S,
            "rate_law": "dq/dt = k2*(q_eq-q)*abs(q_eq-q)",
            "fit_basis": "One global k2 fitted to pooled residuals from the three Figure 4C flow-rate curves.",
            "column_metrics": metrics,
            "height_metrics": [
                item
                for item in condition_metrics
                if item["comparison"] == "height"
            ],
            "influent_concentration_metrics": [
                item
                for item in condition_metrics
                if item["comparison"] == "influent_concentration"
            ],
            "notes": [
                "n_h2o is conditional because Jiang used one fixed MgCl2 matrix.",
                "The Mg=0 curve reuses the brine-calibrated K_i and assumes 1000 kg water/m3; it is a counterfactual, not a separate fit.",
                "Pitzer activities are evaluated at 298.15 K; Jiang ran at 303 K.",
                "Figure 4A-C curves were digitized from the embedded source figure.",
                "The same global PSO rate is used for every flow, height, and inlet-concentration case.",
            ],
        }
    )
    with (output_dir / "jiang_pso_metrics.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    _write_csv(
        output_dir / "jiang_pso_isotherm_curve.csv",
        (
            {
                "Li_mg_L": concentration,
                "langmuir_q_mg_g": langmuir,
                "activity_model_q_mg_g": activity,
                "activity_model_mg_zero_q_mg_g": mg_zero,
            }
            for concentration, langmuir, activity, mg_zero in zip(
                fit["curve_concentration_mg_L"],
                fit["curve_langmuir_mg_g"],
                fit["curve_activity_model_mg_g"],
                fit["curve_activity_model_mg_zero_mg_g"],
                strict=True,
            )
        ),
    )
    _write_csv(
        output_dir / "jiang_pso_isotherm_projection.csv",
        (
            {
                "Li_mg_L": concentration,
                "digitized_q_mg_g": observed,
                "langmuir_q_mg_g": langmuir,
                "activity_model_q_mg_g": activity,
                "activity_model_mg_zero_q_mg_g": mg_zero,
            }
            for concentration, observed, langmuir, activity, mg_zero in zip(
                fit["concentration_mg_L"],
                study["Isotherm"]["q_e"],
                fit["langmuir_mg_g"],
                fit["activity_model_mg_g"],
                fit["activity_model_mg_zero_mg_g"],
                strict=True,
            )
        ),
    )
    plot_jiang_validation(
        study,
        fit,
        trajectories,
        condition_trajectories,
        output_dir / "jiang_pso_validation.svg",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--n-h2o", type=float, default=1.0)
    parser.add_argument("--cells", type=int, default=40)
    parser.add_argument("--save-points", type=int, default=401)
    args = parser.parse_args()
    result = run(
        args.output_dir,
        args.n_h2o,
        args.cells,
        args.save_points,
    )
    print(json.dumps(result, indent=2))
