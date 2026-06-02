"""Model-agnostic Sobol sensitivity driver.

To add a new model, build a `SobolModel` adapter (see Sensitivity/adapters/)
and pass it to `run_sobol`. The driver owns the SALib sampling, evaluation
loop, index analysis, CSV/plot output. The adapter owns everything
model-specific: parameter names, bounds, the model call, and the output dir.
"""

from __future__ import annotations

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
    return pd.DataFrame({
        "param":   names,
        "S1":      Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST":      Si["ST"],
        "ST_conf": Si["ST_conf"],
    })


def _plot_sobol_indices(df: pd.DataFrame, filename: Path, title: str) -> None:
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
) -> Path:
    """Sample, evaluate, analyse, write outputs. Returns the output_dir.

    Per-sample exceptions are caught and logged at WARNING with the offending
    parameter vector so a single bad evaluation doesn't kill the whole sweep.
    Failed samples leave NaN in every output column; the analysis step then
    drops them via SALib's standard handling (variance == 0 keys are skipped).
    """
    _apply_runtime_setup()

    out_dir = Path(model.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log.info("[%s] output dir = %s", model.tag, out_dir)

    problem = {
        "num_vars": len(model.names),
        "names":    list(model.names),
        "bounds":   [list(b) for b in model.bounds],
        "dists":    list(model.dists),
    }
    _log.info("[%s] problem: %d vars %s; bounds=%s",
              model.tag, problem["num_vars"], problem["names"], problem["bounds"])

    _log.info("[%s] generating Sobol samples (N=%d, calc_second_order=%s) ...",
              model.tag, N, calc_second_order)
    param_values = sobol_sample.sample(
        problem, N, calc_second_order=calc_second_order,
    )
    n_evals = param_values.shape[0]
    _log.info("[%s] total evaluations: %d", model.tag, n_evals)

    output_arrays = {key: np.full(n_evals, np.nan) for key in model.outputs}
    failures = 0
    t_loop_start = time.perf_counter()

    with tqdm.tqdm(total=n_evals, desc=f"[{model.tag}] eval", unit="eval") as pbar:
        for i, vec in enumerate(param_values):
            try:
                result = model.run(vec)
            except Exception:
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

    for key, title in model.outputs.items():
        values = output_arrays[key]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            _log.error("[%s] %s: all values non-finite, skipping.", model.tag, key)
            continue
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

    _log.info("[%s] saving raw samples to %s",
              model.tag, out_dir / "sobol_cycle_samples.csv")
    sample_columns = {
        name: param_values[:, i] for i, name in enumerate(model.names)
    }
    sample_columns.update(output_arrays)
    pd.DataFrame(sample_columns).to_csv(
        out_dir / "sobol_cycle_samples.csv", index=False,
    )

    _log.info("[%s] done. outputs -> %s", model.tag, out_dir)
    return out_dir
