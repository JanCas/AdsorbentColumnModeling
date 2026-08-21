# v2 — v1 restyled through JansPlottingStuff for publication

Generated: 2026-08-20  
Git commit: `13011ba1` (`Model/Multi_Ion_ALLDH/` is untracked at this commit)  
Data and metrics: [`../../`](../../) — `jiang_pso_metrics.json`,
`jiang_pso_isotherm_curve.csv`, `jiang_pso_isotherm_projection.csv`,
`jiang_pso_flow_*.csv`, `jiang_pso_condition_*.csv`, `jiang_pso_*.npz`

> **The model configuration is identical to [v1](../v1/).** Same isotherm, same
> `k2`, same nine cases, same grid and solver; `jiang_pso_metrics.json` is
> unchanged. This version differs only in how the figure is drawn.

## Configuration

Unchanged from v1 — see [`../v1/MANIFEST.md`](../v1/MANIFEST.md) for the full
table. The summary rows that matter for reading the plot:

| Setting | Value |
|---|---|
| Cases | 3 flow (6, 9, 15 mL/min) · 3 height (0.3, 0.6, 1.0 m) · 3 inlet Li (300, 350, 400 mg/L) |
| `q_max` | 5.9522 mg/g, Jiang Table 2, not refitted |
| `log K_i` | −1.39740, anchored to Table 2 Langmuir at the 350 mg/L feed |
| `n_h2o` | 1.0 — assumed, not identified |
| `k2` | 1.00148162948e-4 kg sorbent/(mol Li·s), one global value |
| Activity model | Pitzer LiCl/NaCl/MgCl₂ at 298.15 K; Jiang ran at 303 K |
| Grid / solver | N = 40 cells, upwind, Tsit5, PID rtol 1e-5 / atol 1e-8, 401 save points |

## What changed in the drawing

| Aspect | v1 | v2 |
|---|---|---|
| Style source | matplotlib defaults | `JansPlottingStuff.apply()` — `natcomm_paper.mplstyle` |
| Where the code lives | `_plot` inside `jiang_validation.py` | [`Model/Multi_Ion_ALLDH/plotting.py`](../../../../Model/Multi_Ion_ALLDH/plotting.py) |
| Figure size | 11.0 × 8.5 in | 7.1 × 5.6 in (double-column) |
| Output | 180 dpi PNG, opaque | 300 dpi, transparent, `bbox=tight` |
| Fonts | sans default | serif (Times), style-controlled sizes |
| Grid | `alpha=0.25` overlay | none — the style sets `axes.grid: False` |
| Colours | matplotlib `tab10` | batlowS categorical cycle from the style |
| Legends | 6 entries per panel, `fontsize=7.5` | 3 paired line+marker entries, style font size |
| Panel labels | none | **a**–**d** |
| Data markers | `scatter(s=10, alpha=0.55)` | `plot(ls="none", marker="o")` so `lines.markersize` applies |

No style keyword is hardcoded in `plotting.py`; line width, marker size, fonts,
tick direction, colour cycle and background all come from `JPS.apply()`. The
only non-style override is `figsize`, because the style's default is a
single-panel size that cannot carry a 2 × 2 grid.

The middle series in panels b–d is `#faccfa`, the second batlowS colour, which
is pale on white. This was raised and the unmodified cycle was chosen
deliberately over sampling the batlow ramp.

## Produced by

```bash
/opt/miniconda3/envs/ads_col/bin/python -m Model.Multi_Ion_ALLDH.jiang_validation
```

## Figures

- `jiang_pso_validation.png`
- `jiang_pso_validation.svg`

## Result

Identical to v1 — reproduced here so the figure and its numbers travel together.
Isotherm projection: SSE `0.00831` mg²/g², R² `0.99930`.

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

## Note

Supersedes v1 as the figure to cite. v1 is kept because it is what
[`../../REPORT.md`](../../REPORT.md) linked before this restyle.
