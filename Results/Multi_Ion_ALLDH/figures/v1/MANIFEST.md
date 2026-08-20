# v1 — Jiang 2020: activity isotherm + PSO fixed-bed validation, nine columns

Generated: 2026-08-20  
Git commit: `13011ba1` (`Model/Multi_Ion_ALLDH/` is untracked at this commit)  
Data and metrics: [`../../`](../../) — `jiang_pso_metrics.json`,
`jiang_pso_isotherm_curve.csv`, `jiang_pso_isotherm_projection.csv`,
`jiang_pso_flow_*.csv`, `jiang_pso_condition_*.csv`, `jiang_pso_*.npz`

Nine breakthrough curves against Jiang's Figure 4A–C, plus the equilibrium
panel showing the activity isotherm reproducing the Table 2 Langmuir fit.
**One global PSO rate constant** is used for all nine; nothing is refitted per
case.

## Configuration

| Setting | Value |
|---|---|
| Cases | 3 flow (6, 9, 15 mL/min) · 3 height (0.3, 0.6, 1.0 m) · 3 inlet Li (300, 350, 400 mg/L) |
| Isotherm | activity-based, `q = q_max·σ(ln K_i + ln a_LiCl + n_h2o·ln a_w)` |
| `q_max` | 5.9522 mg/g = 0.85767 mol/kg — Jiang Table 2, taken as-is, not refitted |
| `log K_i` | −1.39740 (`K_i` = 0.24724), anchored so the activity isotherm matches Table 2 Langmuir at the 350 mg/L feed |
| `n_h2o` | 1.0 — **assumed, not identified**; Jiang used one fixed MgCl₂ matrix |
| `k2` | 1.00148162948e-4 kg sorbent/(mol Li·s), one global value pooled over the three Figure 4C flow curves |
| Rate law | `dq/dt = k2·(q_eq − q)·|q_eq − q|` (sign-preserving PSO) |
| Activity model | Pitzer LiCl/NaCl/MgCl₂ at **298.15 K**; Jiang ran at 303 K |
| Background | Mg²⁺ = 100 g/L held fixed, brine density 1.253 kg/L → 859.13 kg water/m³ solution |
| Feed (350 mg/L Li) | `a_LiCl` = 105.54, `a_w` = 0.4613, `q_eq` = 5.4956 mg/g |
| Particle density | 1378.7 kg/m³ |
| Bed porosity | 0.355 |
| Grid / solver | N = 40 cells, upwind advection, Tsit5, PID rtol 1e-5 / atol 1e-8, 401 save points |
| Isotherm panel grid | 800-point union of linear and log Li grids over 0–1523.66 mg/L |

## Produced by

```bash
/opt/miniconda3/envs/ads_col/bin/python -m Model.Multi_Ion_ALLDH.jiang_validation
```

## Figures

- `jiang_pso_validation.png`
- `jiang_pso_validation.svg`

## Result

Isotherm projection onto activities: SSE = 0.00831 mg²/g², R² = 0.99930 against
the Table 2 Langmuir curve at the seven digitized `C_e` points.

| Panel | Case | RMSE `C/C₀` | MAE `C/C₀` |
|---|---|---:|---:|
| Flow | 6 mL/min | 0.0574 | 0.0438 |
| Flow | 9 mL/min | 0.0458 | 0.0366 |
| Flow | 15 mL/min | 0.0760 | 0.0378 |
| Height | 0.3 m | 0.0760 | 0.0386 |
| Height | 0.6 m | 0.0760 | 0.0378 |
| Height | 1.0 m | 0.0502 | 0.0417 |
| Inlet Li | 300 mg/L | 0.0584 | 0.0410 |
| Inlet Li | 350 mg/L | 0.0760 | 0.0378 |
| Inlet Li | 400 mg/L | 0.0747 | 0.0446 |

The 0.6 m / 15 mL/min / 350 mg/L case is the same physical run in all three
panels, which is why its numbers repeat.

## Note

> **Superseded by [v2](../v2/)**, which is the same run restyled through
> `JansPlottingStuff`. Kept because it is what `REPORT.md` linked before that
> restyle. The numbers on this page remain valid — v2 changed only the drawing.

First versioned figure for this directory. It supersedes nothing — the loose
`multi_ion_*.svg` files alongside it are earlier, unrelated isotherm and
sensitivity studies that were never versioned, not previous drafts of this
figure.

The equilibrium panel in this version plots both isotherms on a dense grid. The
unversioned 2026-08-19 figure drew those curves through only the seven digitized
`C_e` points, which rendered the sharp low-concentration knee as straight
segments. The underlying fit is unchanged — `jiang_pso_isotherm_projection.csv`
is byte-identical across the two runs — so this is a plotting fix, not a
configuration change.
