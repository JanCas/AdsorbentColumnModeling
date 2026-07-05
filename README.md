# Adsorbent Column Modeling

1‑D packed‑bed column models for selective ion capture (application: **lithium recovery**),
driven through **global sensitivity analysis** and **multi‑objective optimization**.
Everything is written in [JAX](https://github.com/jax-ml/jax) and integrated with the
[diffrax](https://github.com/patrick-kidger/diffrax) ODE solver, so the forward model is
fast, vectorizable, and differentiable.

> **New here?** Read [`PROJECT_GUIDE.html`](PROJECT_GUIDE.html) (open in a browser) for the full
> physics write‑up with equations and diagrams. This README is the practical entry point.

---

## What it models

A packed bed of sorbent particles through which a feed solution flows. The target species is
captured during an **adsorption (loading)** phase and recovered during a **desorption (elution)**
phase. Two physically distinct capture mechanisms are modeled:

| Model | Mechanism | Distinguishing feature |
|-------|-----------|------------------------|
| **AlLDH** | Li/Al layered‑double‑hydroxide adsorbent | Langmuir isotherm + pseudo‑second‑order (PSO) uptake kinetics |
| **IX** | Ion‑exchange resin | Capturing one Li⁺ releases one H⁺, so the model carries explicit acid/base (pH) chemistry |

Each model exists in **two forms**:

- **Dimensional** (SI units) — used for engineering optimization.
- **Non‑dimensional** — used for sensitivity analysis, so results are mechanism‑agnostic and
  comparable across the two models. Loading is normalized by `q_max`; the key dimensionless
  groups are **Λ** (bed capacity ÷ fluid delivery), **Da** (Damköhler = reaction ÷ advection),
  and **θ** (Langmuir feed favorability).

```
        AlLDH (Langmuir + PSO)          IX (ion exchange + pH)
                 │                               │
        ┌────────┴────────┐             ┌────────┴────────┐
    dimensional      non-dimensional  dimensional     non-dimensional
    (optimizer)        (Sobol)        (verify/demo)      (Sobol)
        │                │                                 │
        ▼                └──────────────┬──────────────────┘
   NSGA-II Pareto front                 ▼
   (pymoo)                    Sobol indices (SALib)
                                        │
                                        ▼
                                    Results/  (CSVs + plots)
```

*The dimensional form feeds the optimizer; the non‑dimensional form feeds the Sobol sweeps.*

## The shared modeling framework

Both models share the same skeleton: a 1‑D advection equation for the mobile (liquid) phase
coupled to a lumped linear‑driving‑force (LDF) kinetic law for the stationary (solid) phase,
with an equilibrium isotherm setting the driving force. Axial dispersion is neglected —
transport is pure plug‑flow advection plus mass transfer to the solid.

- **Advection** — first‑order upwind (backward) finite difference on `N = 100` nodes.
- **Inlet BC** — one Dirichlet condition applied through a ghost cell (feed during loading,
  strip fluid during desorption).
- **Outlet BC** — none; a free convective outflow is the correct, complete treatment for pure
  advection.
- **Cycle** — adsorption loads an (almost) empty bed until outlet breakthrough; desorption
  inherits that loaded profile, switches the inlet off/to strip fluid, and runs until the outlet
  falls back below threshold. Integrated with diffrax's `Tsit5` explicit solver and event‑based
  stop conditions.

## Repository layout

```
Model/
  AlLDH/
    diffrax_non_dim_qmax.py    ★ AlLDH non-dim model (Sobol)
    diffrax_column_model.py      AlLDH dimensional column (the optimizer's forward model)
    multi_optim.py             ★ NSGA-II optimization driver
  IX/
    NonDim/ix_nondim_qmax.py   ★ IX non-dim model (Sobol)
    NonDim/verification_qmax.py  mass-balance / convergence checks
    dim/ix_model.py              IX dimensional column (three fields: cation A, proton-excess T, loading n)
    dim/{demo,plotting,verification}.py
Sensitivity/
  sobol_driver.py              ★ generic Sobol engine (SALib)
  run_joint_qmax.py            ★ entry point: runs AlLDH + IX sweeps
  adapters/alldh_nondim_qmax.py  maps a Sobol sample row → AlLDH model call → QoIs
  adapters/ix_nondim_qmax.py     maps a Sobol sample row → IX model call → QoIs
utils/
  Dataclasses.py               Study / ColumnParameters (dimensional params, isotherm fits)
  ResultsDataclasses.py        output containers
  logging_setup.py             stdlib logging config for the orchestration layer
Results/                       generated CSVs + plots (sweeps, optimizations) and their plotting scripts
LiteratureReview/              fitted isotherm/kinetics data (isotherm_kinetics.json)
PROJECT_GUIDE.html             full physics write-up (start here)
```

★ marks the files most worth reading first. The two non‑dimensional model files are heavily
commented — start there, then read the matching adapter to see exactly which dimensionless groups
are swept and over what ranges.

## Installation

```bash
# Python 3.13+
pip install -r requirements.txt
```

Key dependencies: `jax`, `diffrax`, `equinox`, `SALib`, `pymoo`, `pandas`, `matplotlib`, `tqdm`.
JAX is forced onto **CPU** and **float64** inside the runners (the IX model's `√(T² + 4·Kw)`
proton closure would lose the `Kw` term in float32).

## Usage

### Sensitivity sweep — both models (joint runner)

```bash
python -m Sensitivity.run_joint_qmax --output-dir Results/Sensitivity/my_run \
       --N 2048 --ix-scope full --bounds ix
```

- `--second-order` — also compute S2 (second‑order) Sobol indices.
- `--data-plots` — emit scatter PNG/SVGs.
- `--log10-da-range` / `--log10-lambda-range` … — trim the sampling box.

The driver also exposes `run_sobol(model, N=…)` directly if you build a single adapter via
`get_model()` / `get_full_model()` and want to sweep one model only.

### Optimization — dimensional AlLDH

```bash
python Model/AlLDH/multi_optim.py --brine_concentration 50 \
       --loss-fraction 0.01 --output_dir Results/Optimization/my_opt
```

Runs NSGA‑II over the 5‑variable design vector to trade off **specific energy consumption (SEC,
minimize)** against **productivity (maximize)**, and writes `pareto_front_<loss>.csv` plus a
Pareto plot.

### Logging

```bash
LOG_LEVEL=DEBUG python -m Sensitivity.run_joint_qmax ...
```

Logging uses stdlib `logging` at the orchestration layer only. Never put `print`/`logging`
inside a JAX‑jitted (`@eqx.filter_jit`) body — use `jax.debug.print` there instead.

## Quantities of interest

Both non‑dimensional models return the same metric set (IX adds two pH diagnostics), computed
over one load→desorb cycle, so the two mechanisms can be compared directly:

| Metric | Meaning |
|--------|---------|
| `tau_ads` / `tau_des` | dimensionless time to breakthrough / elution cutoff |
| `U_b` | bed utilization at end of loading — `∫ n* dζ`, the fraction of `q_max` occupied |
| `R_outlet_des` | amount recovered at the outlet during desorption, `∫ C*(ζ=1) dτ` |
| `productivity` | `R_outlet_des ÷ (tau_ads + tau_des)` — recovery per unit cycle time |
| `R_release` / `R_wash` | mass released from the solid vs. void‑flush wash (should sum to ≈ `R_outlet_des`) |
| `R_wash_over_R_release` | diagnostic: genuine elution vs. just flushing the void |
| `H_out`, `H_out_over_H_in` *(IX)* | outlet proton level at end of desorption (acid‑breakthrough diagnostic) |

## Notes

- The repo was trimmed to the **`q_max`‑normalized** non‑dimensional models only. An older
  feed‑equilibrium‑normalized variant and a standalone Sobol script were removed;
  `Sensitivity.run_joint_qmax` is the single sensitivity entry point. Some comments still name the
  deleted `diffrax_non_dim.py` / `ix_nondim.py` — those are historical breadcrumbs, not live imports.
