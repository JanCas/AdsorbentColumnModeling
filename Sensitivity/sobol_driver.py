"""Model-agnostic Sobol sensitivity driver.

To add a new model, build a `SobolModel` adapter (see Sensitivity/adapters/)
and pass it to `run_sobol`. The driver owns the SALib sampling, evaluation
loop, index analysis, CSV/plot output. The adapter owns everything
model-specific: parameter names, bounds, the model call, and the output dir.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SobolModel:
    """One Sobol study, fully described.

    Attributes
    ----------
    tag : short slug used in filenames and log lines (e.g. "alldh_nondim").
    names : raw input parameter names (must match the order of `bounds`).
    bounds : (lo, hi) per parameter.
    dists : SALib distribution per parameter, e.g. ["unif", "unif", ...].
    pretty_names : LaTeX-rendered axis labels matching `names`.
    outputs : output_key -> human title; output_key is used in filenames.
    output_dir : where CSVs + PNG/SVG land. Created if missing.
    run : (np.ndarray of shape (num_vars,)) -> {output_key: float, ...}.
          Must return a value for every key in `outputs`.
    postfix : optional, builds a tqdm postfix dict from (sample_vec, result).
    """
    tag: str
    names: list[str]
    bounds: list[tuple[float, float]]
    dists: list[str]
    pretty_names: list[str]
    outputs: dict[str, str]
    output_dir: Path
    run: Callable[[np.ndarray], dict[str, float]]
    postfix: Callable[[np.ndarray, dict[str, float]], dict[str, str]] | None = None


# ---------------------------------------------------------------------------
# Plotting helpers (lifted from Model/AlLDH/simbol_diffrax_non_dim.py:44-78)
# ---------------------------------------------------------------------------

def _build_sobol_df(Si, names: list[str]) -> pd.DataFrame:
    """Tidy the SALib analysis result `Si` into a per-parameter DataFrame.

    One row per input parameter, carrying the first-order (S1) and total-order
    (ST) Sobol indices plus their bootstrap confidence half-widths (`*_conf`).
    """
    return pd.DataFrame({
        "param":   names,
        "S1":      Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST":      Si["ST"],
        "ST_conf": Si["ST_conf"],
    })


def _plot_sobol_indices(df: pd.DataFrame, filename: Path, title: str) -> None:
    """Draw the standard two-panel Sobol bar chart for one output metric.

    Left panel = first-order S1 (variance explained by each input alone),
    right panel = total-order ST (input plus all its interactions). Error bars
    are the SALib confidence half-widths. Saved as both PNG and SVG.
    """
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].bar(x, df["S1"], yerr=df["S1_conf"], width=0.6, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["param"], fontsize=9)
    axes[0].set_ylabel("S1")
    axes[0].set_title("First-order")

    axes[1].bar(x, df["ST"], yerr=df["ST_conf"], width=0.6, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["param"], fontsize=9)
    axes[1].set_ylabel("ST")
    axes[1].set_title("Total-order")

    ymax = max(1.0, 1.1 * df[["S1", "ST"]].to_numpy().max())
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.savefig(filename, dpi=300)
    fig.savefig(str(filename).replace(".png", ".svg"))
    plt.close(fig)


def _build_s2_df(Si, names: list[str]) -> pd.DataFrame:
    """Long-form second-order Sobol table: one row per (i, j) pair, i < j.

    SALib returns S2 as a (k, k) array with NaN on the diagonal and the
    lower triangle — only S2[i, j] for i < j is meaningful. We flatten that
    upper triangle into a tidy CSV so it slots in alongside the existing
    `sobol_<metric>.csv` (S1/ST) output.
    """
    s2 = np.asarray(Si["S2"], dtype=float)
    s2_conf = np.asarray(Si["S2_conf"], dtype=float)
    k = len(names)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            rows.append({
                "param_i": names[i],
                "param_j": names[j],
                "S2":      s2[i, j],
                "S2_conf": s2_conf[i, j],
            })
    return pd.DataFrame(rows)


def _plot_s2_diagnostic(
    Si,
    names: list[str],
    df_s1st: pd.DataFrame,
    filename: Path,
    title: str,
) -> None:
    """Three-panel S2 diagnostic — replaces the bare symmetrised heatmap.

    Panel A — upper-triangle heatmap. Only the cells SALib actually fills
    are rendered (the symmetric mirror is redundant), so the eye gets half
    the visual load. Each cell is annotated with both $S_2$ and its
    confidence so noisy pairs are obvious.

    Panel B — pair ranking. Horizontal bar chart of every $(i, j)$ pair in
    the upper triangle, sorted by $|S_2|$, with $S_2\\_conf$ error bars.
    This is the direct answer to "which pair carries the interaction
    load?" — much easier to read than scanning a $k \\times k$ grid.

    Panel C — interaction-budget check. Per axis $i$, compares the total
    interaction load $S_T(i) - S_1(i)$ against $\\sum_{j \\ne i} S_2[i, j]$
    (what the pairwise indices actually capture). A large positive gap
    means $\\ge 3$-way interactions matter on that axis; near-equality
    means $S_2$ tells the full interaction story. This is the same sanity
    check spelled out in `Results/Sensitivity/joint/1.24/review.html`
    appendix §"Sanity check before trusting the S2 numbers".

    No `cmap=` is passed — `JansPlottingStuff.apply()` owns the project
    style. Text-on-heatmap colour flips at half the value range so the
    annotations stay readable against any cmap.
    """
    import matplotlib.gridspec as gridspec

    s2 = np.asarray(Si["S2"], dtype=float)
    s2_conf = np.asarray(Si["S2_conf"], dtype=float)
    k = len(names)
    triu = np.triu_indices(k, k=1)
    finite_mask = np.isfinite(s2[triu])
    if not finite_mask.any():
        _log.warning("S2 diagnostic %s: no finite interactions, skipping.",
                     filename.name)
        return

    # Pair list (upper triangle only), sorted by |S2| descending.
    pairs: list[tuple[int, int, float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            v = s2[i, j]
            if np.isfinite(v):
                pairs.append((i, j, float(v), float(s2_conf[i, j])))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)

    # Per-axis interaction budget: ST - S1 vs sum over j of S2[i, j].
    s1 = df_s1st["S1"].to_numpy(dtype=float)
    st = df_s1st["ST"].to_numpy(dtype=float)
    interaction_load = st - s1
    s2_filled = np.where(np.isnan(s2), 0.0, s2)
    s2_sym = s2_filled + s2_filled.T  # symmetrise so row-sum captures both i<j and j<i
    np.fill_diagonal(s2_sym, 0.0)
    s2_axis_share = s2_sym.sum(axis=1)

    # Figure layout.
    n_pairs = len(pairs)
    fig_h = max(7.5, 0.32 * max(n_pairs, k) + 4.5)
    fig = plt.figure(figsize=(13.5, fig_h), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[1.4, 1.0],
                           width_ratios=[1.0, 1.05])

    # --- Panel A: upper-triangle heatmap ---
    ax_h = fig.add_subplot(gs[0, 0])
    display = np.full((k, k), np.nan)
    display[triu] = s2[triu]
    vmax_pos = float(np.nanmax(display[triu])) if np.isfinite(display[triu]).any() else 0.0
    vmax = max(vmax_pos, 1e-12)
    im = ax_h.imshow(display, aspect="equal", vmin=0.0, vmax=vmax)
    ax_h.set_xticks(np.arange(k))
    ax_h.set_yticks(np.arange(k))
    ax_h.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_h.set_yticklabels(names, fontsize=9)
    threshold = 0.5 * vmax
    for i in range(k):
        for j in range(i + 1, k):
            v = display[i, j]
            if not np.isfinite(v):
                continue
            c = s2_conf[i, j]
            color = "white" if v > threshold else "black"
            ax_h.text(j, i, f"{v:.2f}\n±{c:.2f}", ha="center", va="center",
                      fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax_h, shrink=0.85)
    cbar.set_label(r"$S_2$")
    ax_h.set_title("A) Upper-triangle $S_2$ heatmap")

    # --- Panel B: ranked pair bar ---
    ax_b = fig.add_subplot(gs[0, 1])
    labels = [f"{names[i]}, {names[j]}" for i, j, _, _ in pairs]
    vals = np.array([v for _, _, v, _ in pairs])
    confs = np.array([c for _, _, _, c in pairs])
    y = np.arange(len(pairs))
    ax_b.barh(y, vals, xerr=confs, height=0.7, capsize=3)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.invert_yaxis()
    ax_b.axvline(0.0, color="k", lw=0.6, alpha=0.5)
    ax_b.set_xlabel(r"$S_2 \pm S_{2,\mathrm{conf}}$")
    ax_b.grid(axis="x", alpha=0.3)
    ax_b.set_title("B) Pairs ranked by $|S_2|$")

    # --- Panel C: per-axis interaction-budget bar ---
    ax_c = fig.add_subplot(gs[1, :])
    x = np.arange(k)
    w = 0.38
    ax_c.bar(x - w / 2, interaction_load, width=w,
             label=r"$S_T - S_1$ (total interaction load)")
    ax_c.bar(x + w / 2, s2_axis_share, width=w,
             label=r"$\sum_{j\neq i} S_2[i, j]$ (pairwise share)")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_c.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax_c.set_ylabel("Sobol mass")
    ax_c.legend(loc="upper right", fontsize=9)
    ax_c.grid(axis="y", alpha=0.3)
    ax_c.set_title(r"C) Interaction-budget check: $S_T - S_1$ vs $\sum_{j} S_2[i, j]$")

    fig.suptitle(f"{title} — Second-order diagnostic")
    fig.savefig(filename, dpi=300)
    fig.savefig(str(filename).replace(".png", ".svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Setup hooks (only fire when run_sobol is called, not at import)
# ---------------------------------------------------------------------------

def _apply_runtime_setup() -> None:
    """Force CPU JAX + apply Jan's plotting style. Match what the original
    simbol_diffrax_non_dim.py did at the top of __main__."""
    try:
        import jax
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        _log.info("JAX devices: %s", jax.devices())
    except ImportError:
        _log.debug("jax not available; skipping CPU forcing")
    try:
        import JansPlottingStuff as JPS
        JPS.apply()
        _log.debug("applied JansPlottingStuff style")
    except ImportError:
        _log.debug("JansPlottingStuff not available; default matplotlib style")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sobol(
    model: SobolModel,
    N: int = 2 ** 11,
    calc_second_order: bool = False,
    output_dir: Path | str | None = None,
    data_plots: bool | str | None = False,
    also_linear: bool = False,
    by_metric: bool = False,
) -> Path:
    """Sample, evaluate, analyse, write outputs. Returns the output_dir.

    Per-sample exceptions are caught and logged at WARNING with the offending
    parameter vector so a single bad evaluation doesn't kill the whole sweep.
    Failed samples leave NaN in every output column; the analysis step then
    drops them via SALib's standard handling (variance == 0 keys are skipped).

    If `output_dir` is provided it overrides `model.output_dir` for this run
    (the underlying SobolModel is left untouched).

    `data_plots` controls whether and how scatter plots are emitted:
      * False or None: no plots.
      * True:          PNG + SVG into <output_dir>/DataPlots/{png,svg}/.
      * str:           comma-separated extension list (e.g. "png" or
                       "png,svg,pdf"); each ext lands in its own subdir.
    Plot failures are logged but don't fail the sweep.

    `also_linear=True` emits a second full set of plots with input axes
    exponentiated (log10_X → X) into <output_dir>/DataPlots_linear/.

    `by_metric=True` groups the scatter outputs into per-metric
    subfolders (<DataPlots>/<ext>/<metric>/<basename>.<ext>) instead of
    a single flat ext-folder. Applies to the linear sibling too.

    These two flags have no effect when data_plots is off.
    """
    _apply_runtime_setup()

    if output_dir is not None:
        model = dataclasses.replace(model, output_dir=Path(output_dir))
    out_dir = Path(model.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log.info("[%s] output dir = %s", model.tag, out_dir)

    # SALib "problem" definition: the parameter space to sweep. Names, bounds
    # (lo, hi) and per-parameter distributions come straight from the adapter,
    # so the driver never hard-codes anything model-specific.
    problem = {
        "num_vars": len(model.names),
        "names":    list(model.names),
        "bounds":   [list(b) for b in model.bounds],
        "dists":    list(model.dists),
    }
    _log.info("[%s] problem: %d vars %s; bounds=%s",
              model.tag, problem["num_vars"], problem["names"], problem["bounds"])

    # Draw the Saltelli/Sobol sample matrix. Shape is (n_evals, num_vars):
    # one row per model evaluation, one column per input. The row count is
    # N*(2*num_vars+2) with second-order indices, else N*(num_vars+2).
    _log.info("[%s] generating Sobol samples (N=%d, calc_second_order=%s) ...",
              model.tag, N, calc_second_order)
    param_values = sobol_sample.sample(
        problem, N, calc_second_order=calc_second_order,
    )
    n_evals = param_values.shape[0]
    _log.info("[%s] total evaluations: %d", model.tag, n_evals)

    # One output vector per QoI, pre-filled with NaN. A NaN survives wherever a
    # sample fails, and SALib/our variance guard skip those downstream.
    output_arrays = {key: np.full(n_evals, np.nan) for key in model.outputs}
    failures = 0
    t_loop_start = time.perf_counter()

    # Serial evaluation loop: run the forward model once per sample row.
    # (No vmap here — each adapter.run() drives a diffrax integration whose
    # step count depends on the sample, so runs are evaluated one at a time.)
    with tqdm.tqdm(total=n_evals, desc=f"[{model.tag}] eval", unit="eval") as pbar:
        for i, vec in enumerate(param_values):
            try:
                # Adapter maps this sample row -> {output_key: value} QoI dict.
                result = model.run(vec)
            except Exception:
                # A single bad evaluation (solver blow-up, NaN, etc.) is logged
                # with its parameter vector and skipped, leaving NaN in row i.
                failures += 1
                _log.exception(
                    "[%s] sample %d/%d failed; params=%s",
                    model.tag, i + 1, n_evals,
                    dict(zip(model.names, [float(v) for v in vec])),
                )
                pbar.update(1)
                continue
            for key in model.outputs:
                output_arrays[key][i] = float(result[key])
            if model.postfix is not None:
                pbar.set_postfix(model.postfix(vec, result))
            pbar.update(1)

    elapsed = time.perf_counter() - t_loop_start
    _log.info("[%s] eval loop done in %.2fs (%d failures / %d evals)",
              model.tag, elapsed, failures, n_evals)
    if failures > 0:
        _log.warning("[%s] %d of %d samples failed (%.1f%%); affected outputs "
                     "contain NaN", model.tag, failures, n_evals,
                     100.0 * failures / n_evals)

    # Analyse each QoI independently: SALib decomposes its variance over the
    # sampled input box into per-input first-order (S1) and total-order (ST)
    # contributions (plus pairwise S2 if requested).
    for key, title in model.outputs.items():
        values = output_arrays[key]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            _log.error("[%s] %s: all values non-finite, skipping.", model.tag, key)
            continue
        # A constant output has zero variance to decompose — Sobol is undefined,
        # so skip it rather than feed SALib a degenerate column.
        if np.ptp(finite) < 1e-15:
            _log.warning("[%s] %s: no variance, skipping Sobol analysis.",
                         model.tag, key)
            continue
        _log.info("[%s] Sobol analysis: %s  (n_finite=%d, min=%.3g, max=%.3g)",
                  model.tag, key, finite.size, float(finite.min()), float(finite.max()))
        Si = sobol_analyze.analyze(
            problem, values,
            calc_second_order=calc_second_order, print_to_console=False,
        )
        df = _build_sobol_df(Si, model.pretty_names)
        df.to_csv(out_dir / f"sobol_{key}.csv", index=False)
        _plot_sobol_indices(df, out_dir / f"sobol_{key}.png", title)
        _log.debug("[%s] %s S1=%s ST=%s", model.tag, key,
                   list(np.round(df["S1"], 4)), list(np.round(df["ST"], 4)))

        if calc_second_order and "S2" in Si:
            # S2 outputs (CSV + diagnostic plot) live in their own subfolder
            # so they don't visually crowd the first-order S1/ST sweep, and so
            # downstream review tools can iterate the S2/ dir directly.
            s2_dir = out_dir / "S2"
            s2_dir.mkdir(parents=True, exist_ok=True)
            s2_df = _build_s2_df(Si, model.pretty_names)
            s2_df.to_csv(s2_dir / f"sobol_{key}_S2.csv", index=False)
            try:
                _plot_s2_diagnostic(Si, model.pretty_names, df,
                                    s2_dir / f"sobol_{key}_S2.png", title)
            except Exception:
                _log.exception("[%s] S2 diagnostic failed for %s",
                               model.tag, key)

    # Persist the raw sample -> QoI table (one row per evaluation): input
    # columns first, then every output column. This is the source data the
    # scatter "DataPlots" are drawn from.
    _log.info("[%s] saving raw samples to %s",
              model.tag, out_dir / "sobol_cycle_samples.csv")
    sample_columns = {
        name: param_values[:, i] for i, name in enumerate(model.names)
    }
    sample_columns.update(output_arrays)
    samples_csv = out_dir / "sobol_cycle_samples.csv"
    pd.DataFrame(sample_columns).to_csv(samples_csv, index=False)

    if data_plots:
        if data_plots is True:
            exts = ["png", "svg"]
        elif isinstance(data_plots, str):
            exts = [e.strip().lower().lstrip(".")
                    for e in data_plots.split(",") if e.strip()]
        else:
            exts = []
        if exts:
            _emit_data_plots(model, samples_csv, out_dir / "DataPlots", exts,
                             also_linear=also_linear,
                             by_metric=by_metric)

    _log.info("[%s] done. outputs -> %s", model.tag, out_dir)
    return out_dir


def _emit_data_plots(
    model: SobolModel,
    samples_csv: Path,
    out_dir: Path,
    exts: list[str],
    *,
    also_linear: bool = False,
    by_metric: bool = False,
) -> None:
    """Render scatter plots for every (input, output) pair into `out_dir`,
    in each extension listed in `exts`. Failures are logged but don't
    propagate; the sweep result on disk is already complete by the time
    this runs."""
    import sys
    _RESULTS_DIR = Path(__file__).resolve().parents[1] / "Results"
    if str(_RESULTS_DIR) not in sys.path:
        sys.path.insert(0, str(_RESULTS_DIR))
    try:
        from plotting import generate_data_plots
    except ImportError:
        _log.exception("[%s] could not import Results.plotting; skipping data plots",
                       model.tag)
        return

    _log.info("[%s] generating data plots (exts=%s) -> %s",
              model.tag, exts, out_dir)
    try:
        n = generate_data_plots(
            csv_path=samples_csv,
            out_dir=out_dir,
            inputs=list(model.names),
            outputs=list(model.outputs.keys()),
            exts=exts,
            also_linear=also_linear,
            by_metric=by_metric,
            log_fn=lambda msg: _log.info("[%s] %s", model.tag, msg),
        )
    except Exception:
        _log.exception("[%s] data-plot generation failed", model.tag)
        return
    _log.info("[%s] wrote %d data plots × %d ext = %d files",
              model.tag, n, len(exts), n * len(exts))
