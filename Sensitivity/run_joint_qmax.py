"""Joint Sobol sensitivity run — q_max-normalised AlLDH + IX legs.

Sibling of Sensitivity/run_joint.py. Wires the qmax adapters
(Sensitivity/adapters/alldh_nondim_qmax.py, ix_nondim_qmax.py) so the joint
sweep runs against the q_max-normalised models
(Model/AlLDH/diffrax_non_dim_qmax.py, Model/IX/NonDim/ix_nondim_qmax.py).

Outputs land under:

    Results/Sensitivity/<run-name>/
    ├── joint_config.json
    ├── alldh/   (full standalone Sobol output, tagged *_qmax internally)
    └── ix/      (full standalone Sobol output, tagged *_qmax internally)

Bounds presets, axes, and CLI flags are identical to run_joint.py. The
numeric Sobol box on shared axes (Λ, Da, θ, C_thresh_ads, C_thresh_des) is
the same; only the forward model differs (n* = q/q_max on this side).

Usage:
    python -m Sensitivity.run_joint_qmax -o my_qmax_run
    python -m Sensitivity.run_joint_qmax -o smoke --N 64 --data-plots
    python -m Sensitivity.run_joint_qmax -o alldhbox_run --bounds alldh
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path

# Put the two non-dim model folders (and the repo root) on sys.path so the
# adapters can `import diffrax_non_dim_qmax` / `import ix_nondim_qmax` directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "Model" / "AlLDH"))
sys.path.insert(0, str(_REPO_ROOT / "Model" / "IX" / "NonDim"))
sys.path.insert(0, str(_REPO_ROOT))

from Sensitivity.adapters.alldh_nondim_qmax import get_model as get_alldh_model
from Sensitivity.adapters.ix_nondim_qmax import get_full_model, get_screen_model
from Sensitivity.sobol_driver import run_sobol
from utils.logging_setup import configure_logging

_log = logging.getLogger(__name__)


# Joint bounds preset -> (AlLDH adapter bounds-mode, IX adapter bounds-mode).
# One CLI flag picks matched sampling boxes for both legs.
_BOUNDS_MAP: dict[str, tuple[str, str]] = {
    "ix":    ("ix",        "ix"),
    "alldh": ("alldh_log", "alldh"),
}


def _override_axis(model, axis_name: str, lo: float, hi: float):
    """Return a copy of `model` with one named axis' bounds replaced by (lo, hi).

    Raises SystemExit if the axis isn't present in this model's `names` (e.g.
    an IX-only axis passed to the AlLDH leg).
    """
    try:
        idx = model.names.index(axis_name)
    except ValueError:
        raise SystemExit(
            f"axis {axis_name!r} not present in model {model.tag!r} "
            f"(names: {model.names}). Cannot apply override."
        )
    new_bounds = list(model.bounds)
    new_bounds[idx] = (lo, hi)
    return dataclasses.replace(model, bounds=new_bounds)


def _apply_overrides(model, c_thresh_des_range, log10_lambda_range,
                     log10_da_range):
    """Apply the CLI bounds overrides that exist on BOTH legs (shared axes)."""
    if c_thresh_des_range is not None:
        model = _override_axis(model, "C_thresh_des", *c_thresh_des_range)
    if log10_lambda_range is not None:
        model = _override_axis(model, "log10_Lambda", *log10_lambda_range)
    if log10_da_range is not None:
        model = _override_axis(model, "log10_Da", *log10_da_range)
    return model


def _apply_ix_overrides(model, log10_k_star_range, log10_buffer_range,
                        log10_h_in_des_range):
    """IX-leg-only axis overrides. K*, buffer (ω/H_in_ads), H_in_des have no
    AlLDH analog, so these are never applied to the AlLDH leg. buffer/H_in_des
    exist only in the IX *full* scope (nominal in screen), so overriding them
    under --ix-scope screen errors out via _override_axis."""
    if log10_k_star_range is not None:
        model = _override_axis(model, "log10_K_star", *log10_k_star_range)
    if log10_buffer_range is not None:
        model = _override_axis(model, "log10_buffer", *log10_buffer_range)
    if log10_h_in_des_range is not None:
        model = _override_axis(model, "log10_H_in_des", *log10_h_in_des_range)
    return model


def _run_alldh_leg(
    *,
    alldh_bounds: str,
    output_dir,
    N: int,
    second_order: bool,
    data_plots,
    also_linear: bool,
    by_metric: bool,
    c_thresh_des_range,
    log10_lambda_range,
    log10_da_range,
) -> None:
    """Build the AlLDH adapter, apply overrides, and run its Sobol sweep.

    Self-contained (re-imports + re-inits logging and sys.path) so it can be
    the target of a fresh `spawn`ed process under --parallel.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parents[1]
    for p in (_ROOT / "Model" / "AlLDH", _ROOT):
        sp = str(p)
        if sp not in _sys.path:
            _sys.path.insert(0, sp)

    from Sensitivity.adapters.alldh_nondim_qmax import get_model
    from Sensitivity.sobol_driver import run_sobol
    from utils.logging_setup import configure_logging
    configure_logging()

    model = get_model(alldh_bounds)
    model = _apply_overrides(model, c_thresh_des_range, log10_lambda_range,
                             log10_da_range)
    run_sobol(model, N=N, calc_second_order=second_order,
              output_dir=output_dir, data_plots=data_plots,
              also_linear=also_linear, by_metric=by_metric)


def _run_ix_leg(
    *,
    ix_bounds: str,
    ix_scope: str,
    output_dir,
    N: int,
    second_order: bool,
    data_plots,
    also_linear: bool,
    by_metric: bool,
    t_max_mult: float,
    c_thresh_des_range,
    log10_lambda_range,
    log10_da_range,
    log10_k_star_range,
    log10_buffer_range,
    log10_h_in_des_range,
) -> None:
    """Build the IX adapter (screen/full), apply overrides, run its Sobol sweep.

    Self-contained like `_run_alldh_leg` so it can run in a spawned process.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parents[1]
    for p in (_ROOT / "Model" / "IX" / "NonDim", _ROOT):
        sp = str(p)
        if sp not in _sys.path:
            _sys.path.insert(0, sp)

    from Sensitivity.adapters.ix_nondim_qmax import (
        get_full_model, get_screen_model,
    )
    from Sensitivity.sobol_driver import run_sobol
    from utils.logging_setup import configure_logging
    configure_logging()

    factory = get_screen_model if ix_scope == "screen" else get_full_model
    model = factory(ix_bounds, t_max_mult)
    # Shared-axis overrides first, then the IX-only ones (K*, buffer, H_in_des).
    model = _apply_overrides(model, c_thresh_des_range, log10_lambda_range,
                             log10_da_range)
    model = _apply_ix_overrides(model, log10_k_star_range,
                                log10_buffer_range, log10_h_in_des_range)
    run_sobol(model, N=N, calc_second_order=second_order,
              output_dir=output_dir, data_plots=data_plots,
              also_linear=also_linear, by_metric=by_metric)


# Axes that are intentionally model-specific and need NOT line up between the
# two legs. The IX isotherm reparameterization (θ → bare K*) means AlLDH's
# `log10_theta` (Langmuir steepness) and IX's `log10_K_star` (bare mass-action
# constant) are no longer the same physical axis — so cross-model Sobol
# comparison on the isotherm axis is no longer valid. They are tolerated here
# as model-specific rather than aborting the joint run.
_MODEL_SPECIFIC_AXES = {"log10_theta", "log10_K_star"}


def _assert_distribution_match(alldh_model, ix_model) -> None:
    """Sanity-check that the two legs agree on their shared axes before running.

    Every AlLDH axis must have an IX counterpart (else the joint comparison is
    broken) — except the isotherm axis, which legitimately diverged (θ vs K*)
    and is only warned about. IX-only axes (pH groups) are expected and logged.
    """
    alldh_names = set(alldh_model.names)
    ix_names    = set(ix_model.names)
    shared = alldh_names & ix_names
    # Only the genuinely shared axes must line up; the isotherm axis diverged
    # when IX moved θ → K*.
    missing = (alldh_names - ix_names) - _MODEL_SPECIFIC_AXES
    if missing:
        raise RuntimeError(
            f"joint bounds distribution mismatch: AlLDH axes {sorted(missing)} "
            f"have no matching IX axes. AlLDH names: {sorted(alldh_names)}; "
            f"IX names: {sorted(ix_names)}. Check "
            f"Sensitivity/run_joint_qmax.py:_BOUNDS_MAP and the per-model "
            f"bounds presets."
        )
    iso_mismatch = (alldh_names ^ ix_names) & _MODEL_SPECIFIC_AXES
    if iso_mismatch:
        _log.warning("isotherm axis is model-specific (%s) — AlLDH θ and IX "
                     "K* are NOT cross-comparable in this joint run",
                     sorted(iso_mismatch))
    _log.info("distribution check: %d shared axes (%s); "
              "%d IX-only axes (%s)",
              len(shared), sorted(shared),
              len(ix_names - shared), sorted(ix_names - shared))


def main() -> None:
    """CLI entry point: parse args, write config, run the AlLDH + IX legs.

    Resolves the run directory under Results/Sensitivity/, dumps a
    joint_config.json record of every setting, validates axis alignment, then
    runs the two Sobol sweeps either serially or in parallel spawned processes.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", "-o", required=True,
                   help="Run directory name (resolved under "
                        "Results/Sensitivity/). Absolute paths also work.")
    p.add_argument("--bounds", choices=list(_BOUNDS_MAP.keys()), default="ix",
                   help="Joint bounds preset (default: ix).")
    p.add_argument("--ix-scope", choices=["screen", "full"], default="full",
                   help="IX scope (default: full).")
    p.add_argument("--N", type=int, default=2 ** 11,
                   help="SALib base sample count (default: 2**11 = 2048).")
    p.add_argument("--second-order", action="store_true",
                   help="Compute second-order Sobol indices in both legs.")
    p.add_argument("--data-plots", nargs="?", const="png,svg", default=None,
                   metavar="EXTS",
                   help="Render scatter plots into each leg's "
                        "<leg>/DataPlots/<ext>/ subdir.")
    p.add_argument("--linear-plots", action="store_true",
                   help="Also emit a full set of scatter plots with every "
                        "log10_X input column exponentiated to its linear "
                        "value (Λ, Da, θ, …), into each leg's "
                        "DataPlots_linear/<ext>/ sibling folder. No effect "
                        "unless --data-plots is set.")
    p.add_argument("--plots-by-metric", action="store_true",
                   help="Group scatter plots into per-metric subfolders, "
                        "e.g. <leg>/DataPlots/<ext>/tau_ads/..., "
                        "<leg>/DataPlots/<ext>/U_b/..., etc. Applies to "
                        "the linear sibling too. No effect unless "
                        "--data-plots is set.")
    p.add_argument("--t-max-mult", type=float, default=1.0,
                   help="IX t_max multiplier (default 1.0). AlLDH-only "
                        "leg ignores this.")
    p.add_argument("--c-thresh-des-range", nargs=2, type=float,
                   metavar=("LO", "HI"), default=None,
                   help="Override C_thresh_des bounds on both legs.")
    p.add_argument("--log10-lambda-range", nargs=2, type=float,
                   metavar=("LO", "HI"), default=None,
                   help="Override log10_Lambda bounds on both legs.")
    p.add_argument("--log10-da-range", nargs=2, type=float,
                   metavar=("LO", "HI"), default=None,
                   help="Override log10_Da bounds on both legs.")
    p.add_argument("--log10-k-star-range", nargs=2, type=float,
                   metavar=("LO", "HI"), default=None,
                   help="Override log10_K_star bounds on the IX leg only "
                        "(K* has no AlLDH analog). log10-space endpoints.")
    p.add_argument("--log10-buffer-range", nargs=2, type=float,
                   metavar=("LO", "HI"), default=None,
                   help="Override log10_buffer bounds on the IX leg only. "
                        "buffer B = ω/H_in_ads = [OH⁻]_in/A_in (feed-basicity "
                        "buffer); bed loads iff B > 1−C_thresh_ads. Default "
                        "[-2, 0.5] spans no-load↔load. log10-space endpoints. "
                        "Requires --ix-scope full (axis is nominal in screen).")
    p.add_argument("--log10-h-in-des-range", nargs=2, type=float,
                   metavar=("LO", "HI"), default=None,
                   help="Override log10_H_in_des bounds on the IX leg only "
                        "(strip inlet proton level). log10-space endpoints. "
                        "Requires --ix-scope full (axis is nominal in screen).")
    p.add_argument("--parallel", action="store_true",
                   help="Run the two legs concurrently in separate processes.")
    args = p.parse_args()

    configure_logging()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / "Results" / "Sensitivity" / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    alldh_bounds, ix_bounds = _BOUNDS_MAP[args.bounds]

    config = {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "command":          sys.argv,
        "variant":          "qmax",
        "bounds":           args.bounds,
        "alldh_bounds":     alldh_bounds,
        "ix_bounds":        ix_bounds,
        "ix_scope":         args.ix_scope,
        "N":                args.N,
        "second_order":     args.second_order,
        "data_plots":       args.data_plots,
        "linear_plots":     args.linear_plots,
        "plots_by_metric":  args.plots_by_metric,
        "t_max_mult":       args.t_max_mult,
        "c_thresh_des_range_override": args.c_thresh_des_range,
        "log10_lambda_range_override": args.log10_lambda_range,
        "log10_da_range_override":     args.log10_da_range,
        "log10_k_star_range_override":   args.log10_k_star_range,
        "log10_buffer_range_override":   args.log10_buffer_range,
        "log10_h_in_des_range_override": args.log10_h_in_des_range,
        "parallel":                    args.parallel,
    }
    (output_root / "joint_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    _log.info("joint config -> %s", output_root / "joint_config.json")

    # Build both adapters once, up front, purely to validate axis alignment
    # (the actual sweeps rebuild them inside each leg). Apply the same CLI
    # overrides here so the check sees exactly what the legs will run.
    alldh_model = get_alldh_model(alldh_bounds)
    ix_factory = (
        get_screen_model if args.ix_scope == "screen" else get_full_model
    )
    ix_model = ix_factory(ix_bounds, args.t_max_mult)

    if args.c_thresh_des_range is not None:
        lo, hi = args.c_thresh_des_range
        _log.info("override: C_thresh_des bounds → [%.4f, %.4f] on both legs",
                  lo, hi)
        alldh_model = _override_axis(alldh_model, "C_thresh_des", lo, hi)
        ix_model    = _override_axis(ix_model, "C_thresh_des", lo, hi)
    if args.log10_lambda_range is not None:
        lo, hi = args.log10_lambda_range
        _log.info("override: log10_Lambda bounds → [%.3f, %.3f] on both legs",
                  lo, hi)
        alldh_model = _override_axis(alldh_model, "log10_Lambda", lo, hi)
        ix_model    = _override_axis(ix_model, "log10_Lambda", lo, hi)
    if args.log10_da_range is not None:
        lo, hi = args.log10_da_range
        _log.info("override: log10_Da bounds → [%.3f, %.3f] on both legs",
                  lo, hi)
        alldh_model = _override_axis(alldh_model, "log10_Da", lo, hi)
        ix_model    = _override_axis(ix_model, "log10_Da", lo, hi)
    # IX-only overrides (K*, H_in_ads, H_in_des have no AlLDH analog).
    if (args.log10_k_star_range is not None
            or args.log10_buffer_range is not None
            or args.log10_h_in_des_range is not None):
        _log.info("override (IX leg only): K*=%s buffer=%s H_in_des=%s",
                  args.log10_k_star_range, args.log10_buffer_range,
                  args.log10_h_in_des_range)
        ix_model = _apply_ix_overrides(
            ix_model, args.log10_k_star_range, args.log10_buffer_range,
            args.log10_h_in_des_range,
        )

    _assert_distribution_match(alldh_model, ix_model)
    del alldh_model, ix_model  # validation-only; legs rebuild their own models

    # Per-leg kwargs. The leg runners rebuild the adapter and re-apply the same
    # overrides, so both serial calls and spawned processes see identical config.
    alldh_kwargs = dict(
        alldh_bounds=alldh_bounds,
        output_dir=output_root / "alldh",
        N=args.N,
        second_order=args.second_order,
        data_plots=args.data_plots,
        also_linear=args.linear_plots,
        by_metric=args.plots_by_metric,
        c_thresh_des_range=args.c_thresh_des_range,
        log10_lambda_range=args.log10_lambda_range,
        log10_da_range=args.log10_da_range,
    )
    ix_kwargs = dict(
        ix_bounds=ix_bounds,
        ix_scope=args.ix_scope,
        output_dir=output_root / "ix",
        N=args.N,
        second_order=args.second_order,
        data_plots=args.data_plots,
        also_linear=args.linear_plots,
        by_metric=args.plots_by_metric,
        t_max_mult=args.t_max_mult,
        c_thresh_des_range=args.c_thresh_des_range,
        log10_lambda_range=args.log10_lambda_range,
        log10_da_range=args.log10_da_range,
        log10_k_star_range=args.log10_k_star_range,
        log10_buffer_range=args.log10_buffer_range,
        log10_h_in_des_range=args.log10_h_in_des_range,
    )

    if args.parallel:
        # Each leg runs in its own spawned process (fresh JAX/interpreter state,
        # avoids CPU-thread contention between the two independent sweeps).
        _log.info("==== joint q_max run: AlLDH + IX legs in parallel (spawn) ====")
        ctx = mp.get_context("spawn")
        p_alldh = ctx.Process(target=_run_alldh_leg, kwargs=alldh_kwargs,
                              name="alldh-qmax-leg")
        p_ix    = ctx.Process(target=_run_ix_leg,    kwargs=ix_kwargs,
                              name="ix-qmax-leg")
        p_alldh.start()
        p_ix.start()
        p_alldh.join()
        p_ix.join()
        fail = False
        if p_alldh.exitcode != 0:
            _log.error("AlLDH leg failed with exit code %d", p_alldh.exitcode)
            fail = True
        if p_ix.exitcode != 0:
            _log.error("IX leg failed with exit code %d", p_ix.exitcode)
            fail = True
        if fail:
            sys.exit(1)
    else:
        _log.info("==== joint q_max run: AlLDH leg (bounds=%s) ====", alldh_bounds)
        _run_alldh_leg(**alldh_kwargs)
        _log.info("==== joint q_max run: IX leg (bounds=%s, scope=%s, "
                  "t_max_mult=%g) ====",
                  ix_bounds, args.ix_scope, args.t_max_mult)
        _run_ix_leg(**ix_kwargs)

    _log.info("joint q_max run complete; outputs at %s", output_root)


if __name__ == "__main__":
    main()
