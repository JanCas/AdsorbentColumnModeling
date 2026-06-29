"""Generic scatter-plot generator for Sobol / sensitivity output CSVs.

Usage:
    python Results/plotting.py PATH/TO/sobol_cycle_samples.csv
    python Results/plotting.py ... -o OUT_DIR --ext png
    python Results/plotting.py ... --ext png,svg        # both in one run -> <out>/png/, <out>/svg/
    python Results/plotting.py ... --inputs log10_Lambda,log10_Da --outputs tau_ads
    python Results/plotting.py ... --no-color           # skip 3rd-input color overlay
    python Results/plotting.py ... --schema labels.json # override pretty names

Auto-detects which columns are inputs vs outputs from their names. Outputs go
to <csv-dir>/plots/ by default; when --ext lists multiple extensions, each
ext gets its own subfolder so the SVGs and PNGs don't intermix.

Pretty labels are looked up in (priority):
    1. --schema FILE                (explicit override)
    2. <csv-dir>/schema.json        (sibling sidecar)
    3. PRETTY_DEFAULTS              (built-in fallback covering IX + AlLDH)
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

# --- pretty-name fallback registry ---------------------------------------
# Extend when new sweep axes / QoIs appear. The schema.json sidecar mechanism
# below is the preferred long-term path; this dict is the survival floor.
PRETTY_DEFAULTS: dict[str, str] = {
    # IX non-dim Sobol sample axes (log-space + linear thresholds)
    "log10_Lambda":    r"$\log_{10}\Lambda$",
    "log10_Da":        r"$\log_{10}Da$",
    "log10_K_star":    r"$\log_{10}K^*$",
    "log10_H_in_des":  r"$\log_{10}H^*_{in,des}$",
    "log10_H_in_ads":  r"$\log_{10}H^*_{in,ads}$",
    "log10_buffer":    r"$\log_{10}(\omega/H^*_{in,ads})$",  # feed-basicity buffer [OH-]/A_in
    "log10_omega":     r"$\log_{10}\omega$",
    "C_thresh_ads":    r"$C^*_{th,ads}$",
    "C_thresh_des":    r"$C^*_{th,des}$",
    # Legacy IX CSVs from before the K*/H_in renames
    "log10_theta":     r"$\log_{10}\Theta$",
    "log10_beta":      r"$\log_{10}\beta$",
    "log10_Pi_el":     r"$\log_{10}\Pi_{el}$",
    "log10_H_star_in": r"$\log_{10}H^*_{in}$",
    # AlLDH non-dim Sobol axes (linear)
    "Lambda":          r"$\Lambda$",
    "Da":              r"$Da$",
    "theta":           r"$\theta$",
    "epsilon":         r"$\varepsilon$",
    # Legacy 2nd-order naming (older Results/Sensitivity/* folders)
    "Da2_star":        r"$Da_2^*$",
    "Theta":           r"$\Theta^*$",
    "tau_star_break":  r"$\tau_b$",
    "bed_utilization": r"$U_b$",
    # Shared QoIs
    "tau_ads":         r"$\tau_{ads}$",
    "tau_des":         r"$\tau_{des}$",
    "U_b":             r"$U_b$",
    "R_outlet_des":    r"$R_{outlet,des}$",
    "R_release":       r"$R_{release}$",
    "R_wash":          r"$R_{wash}$",
    "R_wash_over_R_release": r"$R_{wash}/R_{release}$",
    "productivity":    r"Productivity",
}

# Outputs that should render on a log-scaled y-axis. Heavy-tailed metrics
# (where the floor clamp can drive values across many decades) are
# unreadable on a linear axis — log scale spreads the bulk while preserving
# the floor-hit tail. Linear y stays the default for everything else.
_LOG_Y_OUTPUTS = {
    "R_wash_over_R_release",
}

# Auto-detection rules. Inputs match a regex on the column name; outputs are
# an explicit allow-list so adding a new QoI is a one-line change here.
_INPUT_PATTERNS = [
    re.compile(r"^log10_"),
    re.compile(r"^C_thresh_"),
]
_OUTPUT_NAMES = {
    "tau_ads", "tau_des", "U_b", "R_outlet_des", "productivity",
    "tau_star_break", "bed_utilization",
}


def load_pretty_names(csv_path: Path, override: Path | None) -> dict[str, str]:
    """Return label dict. Schema-file overrides shadow PRETTY_DEFAULTS."""
    if override is not None:
        return {**PRETTY_DEFAULTS, **json.loads(override.read_text())}
    sidecar = csv_path.parent / "schema.json"
    if sidecar.exists():
        return {**PRETTY_DEFAULTS, **json.loads(sidecar.read_text())}
    return dict(PRETTY_DEFAULTS)


def classify_columns(cols: list[str]) -> tuple[list[str], list[str]]:
    """Split columns into (inputs, outputs) using prefix/allow-list rules."""
    inputs = [c for c in cols if any(p.search(c) for p in _INPUT_PATTERNS)]
    outputs = [c for c in cols if c in _OUTPUT_NAMES]
    return inputs, outputs


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    c: str | None,
    labels: dict[str, str],
    out_paths: list[Path],
) -> None:
    """Single scatter rendered once; written to every path in `out_paths`.

    `c` colours points by a third column when provided. Passing a list of
    paths lets one render hit multiple file formats (e.g. PNG + SVG) without
    redrawing. Outputs listed in `_LOG_Y_OUTPUTS` get a log y-axis.
    """
    fig, ax = plt.subplots()
    if c is None:
        ax.scatter(df[x], df[y], s=8)
    else:
        sc = ax.scatter(df[x], df[y], c=df[c], s=8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(labels.get(c, c))
    ax.set_xlabel(labels.get(x, x))
    ax.set_ylabel(labels.get(y, y))
    if y in _LOG_Y_OUTPUTS:
        ax.set_yscale("log")
    for p in out_paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def _split_csv_arg(s: str | None) -> list[str] | None:
    return [c.strip() for c in s.split(",")] if s else None


_EXT_CHOICES = ("svg", "png", "pdf")


def _parse_exts(s: str) -> list[str]:
    """Comma-separated extension list with dedup + validation."""
    exts: list[str] = []
    for e in s.split(","):
        e = e.strip().lower().lstrip(".")
        if e not in _EXT_CHOICES:
            raise argparse.ArgumentTypeError(
                f"unknown extension {e!r}; choose from {_EXT_CHOICES}")
        if e not in exts:
            exts.append(e)
    if not exts:
        raise argparse.ArgumentTypeError("--ext must list at least one extension")
    return exts


def generate_data_plots(
    csv_path: Path,
    out_dir: Path,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    exts: list[str] = ["svg"],
    exp_log: bool = False,
    no_color: bool = False,
    schema_path: Path | None = None,
    apply_style: bool = True,
    also_linear: bool = False,
    by_metric: bool = False,
    log_fn=print,
) -> int:
    """Render input-vs-output scatter plots from a Sobol samples CSV.

    Returns the number of distinct plots (file count = returned × len(exts)).
    When multiple `exts` are requested, each gets its own subfolder under
    `out_dir` (e.g. `out_dir/png/`, `out_dir/svg/`); otherwise files land
    directly in `out_dir`.

    `inputs` / `outputs` may be None to fall back to auto-detection by
    column-name (`^log10_` / `^C_thresh_` for inputs; `_OUTPUT_NAMES` for
    outputs). Pass them explicitly for AlLDH legacy CSVs where inputs are
    bare names like `Lambda`, `Da`, ....

    `exp_log=True` swaps `log10_X` columns for their exponentiated counter-
    parts and replaces axis references; equivalent to plotting on linear
    axes.

    `also_linear=True` emits a SECOND full set of plots with every
    `log10_X` input column exponentiated to `X` — equivalent to running
    again with `exp_log=True`. The linear-axis copies land in a SIBLING
    folder of `out_dir` named `<out_dir.name>_linear` (e.g. `DataPlots/`
    → `DataPlots_linear/`), with the same ext-subfolder structure. Y-axis
    treatment of heavy-tailed outputs (`_LOG_Y_OUTPUTS`) is unchanged.

    `by_metric=True` groups output files into per-metric subfolders
    (`<out_dir>/<ext>/<y>/<basename>.<ext>`) so plots are easy to scan
    one output at a time. Applies to the linear sibling as well.
    """
    if apply_style:
        try:
            import JansPlottingStuff as JPS
            JPS.apply()
        except ImportError:
            pass

    df = pd.read_csv(csv_path)
    labels = load_pretty_names(csv_path, schema_path)

    if inputs is None or outputs is None:
        auto_in, auto_out = classify_columns(list(df.columns))
        if inputs is None:
            inputs = auto_in
        if outputs is None:
            outputs = auto_out

    # Capture originals before the exp_log rename so the also_linear recursion
    # below sees the same log10-named columns the caller passed in.
    orig_inputs  = list(inputs)
    orig_outputs = list(outputs)

    if exp_log:
        rename = {c: c[len("log10_"):] for c in df.columns
                  if c.startswith("log10_")}
        for src, dst in rename.items():
            if dst not in df.columns:
                df[dst] = 10.0 ** df[src]
        inputs  = [rename.get(c, c) for c in inputs]
        outputs = [rename.get(c, c) for c in outputs]

    if not inputs or not outputs:
        raise ValueError(
            f"no plotable columns: inputs={inputs}, outputs={outputs}. "
            f"available columns: {list(df.columns)}"
        )

    multi_ext = len(exts) > 1
    if multi_ext:
        ext_dirs = {ext: out_dir / ext for ext in exts}
    else:
        ext_dirs = {exts[0]: out_dir}
    for d in ext_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def paths_for(basename: str, y: str) -> list[Path]:
        # by_metric=True nests each output's plots under a `<y>/` subfolder
        # of its ext-dir so the flat ~hundreds of files become a small
        # collection of per-metric subdirectories.
        if by_metric:
            out_paths = []
            for ext in exts:
                metric_dir = ext_dirs[ext] / y
                metric_dir.mkdir(parents=True, exist_ok=True)
                out_paths.append(metric_dir / f"{basename}.{ext}")
            return out_paths
        return [ext_dirs[ext] / f"{basename}.{ext}" for ext in exts]

    log_fn(f"loaded {len(df)} rows from {csv_path}")
    log_fn(f"inputs:  {inputs}")
    log_fn(f"outputs: {outputs}")
    log_fn(f"writing {len(exts)} format(s) {exts} to {out_dir}"
           + (" (per-ext subfolders)" if multi_ext else ""))

    tasks: list[tuple[str, str, str | None, str]] = []
    for x, y in product(inputs, outputs):
        if no_color:
            tasks.append((x, y, None, f"{x}__{y}"))
        else:
            for c in inputs:
                if c == x:
                    continue
                tasks.append((x, y, c, f"{x}__{y}__c-{c}"))

    with tqdm.tqdm(tasks, desc="plotting", unit="plot") as pbar:
        for x, y, c, basename in pbar:
            pbar.set_postfix(x=x, y=y, c=c or "-")
            scatter_plot(df, x, y, c, labels, paths_for(basename, y))

    log_fn(f"wrote {len(tasks)} render(s) × {len(exts)} ext = "
           f"{len(tasks) * len(exts)} file(s).")
    total_renders = len(tasks)

    # Linear-axis companion pass: render again with exp_log=True (every
    # log10_X column exponentiated to X) into a sibling folder so the
    # caller can compare log-axis and linear-axis views side by side.
    # `not exp_log` guards against infinite recursion when the caller is
    # itself already in exp_log mode.
    if also_linear and not exp_log:
        linear_dir = out_dir.parent / f"{out_dir.name}_linear"
        log_fn(f"linear-axis companion pass -> {linear_dir}")
        total_renders += generate_data_plots(
            csv_path=csv_path,
            out_dir=linear_dir,
            inputs=orig_inputs,
            outputs=orig_outputs,
            exts=exts,
            exp_log=True,
            no_color=no_color,
            schema_path=schema_path,
            apply_style=False,        # JPS.apply() already done above
            also_linear=False,        # don't recurse
            by_metric=by_metric,
            log_fn=log_fn,
        )

    return total_renders


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="path to a samples CSV")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output dir (default: <csv-dir>/plots/)")
    p.add_argument("--schema", type=Path, default=None,
                   help="JSON file overriding pretty-name labels")
    p.add_argument("--inputs",  type=str, default=None,
                   help="comma-separated input columns (default: auto-detect)")
    p.add_argument("--outputs", type=str, default=None,
                   help="comma-separated output columns (default: auto-detect)")
    p.add_argument("--ext", type=_parse_exts, default=["svg"],
                   help="output file extension(s); comma-separated to emit "
                        "multiple in one run (e.g. png,svg). When multiple, "
                        "each ext lands in <out>/<ext>/. Choices: "
                        f"{', '.join(_EXT_CHOICES)}.")
    p.add_argument("--no-color", action="store_true",
                   help="emit one plot per (input, output) only; "
                        "skip the third-input colour-overlay sweep")
    p.add_argument("--exp-log", action="store_true",
                   help="exponentiate every log10_X column to X and plot with "
                        "linear axes (axis labels switch from log10 labels to "
                        "the bare-symbol labels in PRETTY_DEFAULTS).")
    p.add_argument("--linear-plots", action="store_true",
                   help="also emit a full second set of plots with every "
                        "log10_X input column exponentiated to X (linear "
                        "axes), into a sibling folder "
                        "<out>_linear/. Equivalent to running again with "
                        "--exp-log into that sibling.")
    p.add_argument("--by-metric", action="store_true",
                   help="group output files into per-metric subfolders "
                        "(<out_dir>/<ext>/<metric>/<basename>.<ext>) "
                        "instead of one flat directory per ext.")
    args = p.parse_args()

    out_dir = args.out or (args.csv.parent / "plots")
    try:
        generate_data_plots(
            csv_path=args.csv,
            out_dir=out_dir,
            inputs=_split_csv_arg(args.inputs),
            outputs=_split_csv_arg(args.outputs),
            exts=args.ext,
            exp_log=args.exp_log,
            no_color=args.no_color,
            schema_path=args.schema,
            also_linear=args.linear_plots,
            by_metric=args.by_metric,
        )
    except ValueError as e:
        raise SystemExit(f"{e}\npass --inputs / --outputs explicitly.") from e


if __name__ == "__main__":
    main()
