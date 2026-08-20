# Multi-ion ALLDH figure versions

Graphs produced from the multi-ion ALLDH model live here, in a numbered version
folder.

**A version is one set of graphs generated under one configuration.** Version
numbers are chronological and mean nothing beyond creation order — `v5` is not
"better" than `v3`, it is only later. When you generate graphs under a
configuration that differs from the newest version, create the next `vN` rather
than overwriting.

The convention, including what counts as a configuration change, is in the
`figure-versioning` skill.

> The loose `multi_ion_*.svg` files in the parent directory predate this
> convention and are **not** covered by it. They come from
> `Model/Multi_Ion_ALLDH/plot_isotherms.py` (isotherm and brine surveys,
> 2026-08-07) and from the sensitivity sweep (2026-08-09), and are separate
> studies rather than earlier drafts of anything here.

| Version | Date | Scope | Isotherm | Style | Figures | Status |
|---|---|---|---|---|---:|---|
| [v1](v1/) | 2026-08-20 | Jiang 2020, nine columns + equilibrium | activity, `n_h2o` = 1 | matplotlib defaults | 2 | superseded by v2 |
| [v2](v2/) | 2026-08-20 | Jiang 2020, nine columns + equilibrium | activity, `n_h2o` = 1 | `JPS.apply()`, natcomm paper | 2 | **current** |

## What changed between versions

- **v1** is the first version in this directory.
- **v1 → v2** is a **restyle only**. Identical model configuration, identical
  solve, identical `jiang_pso_metrics.json` — every RMSE is unchanged. The
  figure moved to `JansPlottingStuff.apply()` (serif, batlowS cycle, no grid,
  transparent, 300 dpi, double-column), the plotting code moved out of
  `jiang_validation.py` into `Model/Multi_Ion_ALLDH/plotting.py`, legends went
  from six entries to three paired line+marker entries, and panels gained
  **a**–**d** labels. Bumped rather than overwritten because the figure a reader
  sees is materially different.

## Reading a version

Each folder carries a `MANIFEST.md` giving the configuration that produced it,
the command that generated it, the git commit, and a link back to the data that
accompanies those figures. Data files stay with their study directory; only
graphs are versioned here.
