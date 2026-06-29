"""Reorganize existing DataPlots/ folders into per-metric subfolders.

Moves flat `<x>__<y>__c-<c>.<ext>` files into `<y>/<x>__<y>__c-<c>.<ext>`,
mirroring the layout produced by `--plots-by-metric` on new runs.

Idempotent: files already sitting inside a per-metric subfolder are skipped.
Both `DataPlots/` and `DataPlots_linear/` are handled.

Usage:
    python -m Sensitivity.reorganize_plots PATH [PATH ...]
    python -m Sensitivity.reorganize_plots PATH --dry-run

PATH can be:
  * a `DataPlots/` (or `DataPlots_linear/`) folder directly,
  * a leg directory containing `DataPlots/` and/or `DataPlots_linear/`,
  * a joint run directory with `alldh/` and `ix/` subfolders,
  * any ancestor — the script walks down to find every plot folder.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

_EXTS = {".png", ".svg", ".pdf"}
_EXT_SUBDIR_NAMES = {"png", "svg", "pdf"}
_TARGET_DIR_NAMES = {"DataPlots", "DataPlots_linear"}


def _split_basename(stem: str) -> tuple[str, str, str | None] | None:
    """Parse a `<x>__<y>` or `<x>__<y>__c-<c>` basename. Returns
    (x, y, color_or_None) or None if it doesn't match the convention.

    Convention from Results/plotting.py: `__` (double underscore) is the
    field separator; the color field is prefixed with `c-`. Single
    underscores inside any name (e.g. `log10_Lambda`, `R_wash_over_R_release`)
    are part of that name and are NOT treated as separators.
    """
    parts = stem.split("__")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3 and parts[2].startswith("c-"):
        return parts[0], parts[1], parts[2][2:]
    return None


def _reorganize_folder(folder: Path, dry_run: bool) -> tuple[int, int]:
    """Move plot files at the top level of `folder` into per-metric subdirs.

    Returns (moved, skipped). Files inside an existing subfolder (i.e. a
    previous reorganization) are not touched because the loop only iterates
    top-level entries of `folder`.
    """
    moved = 0
    skipped = 0
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _EXTS:
            continue
        parsed = _split_basename(entry.stem)
        if parsed is None:
            _log.warning("can't parse %s, skipping", entry.name)
            skipped += 1
            continue
        _x, y, _c = parsed
        dest = folder / y / entry.name
        if dest.exists():
            skipped += 1
            continue
        if dry_run:
            print(f"  WOULD MOVE: {entry.name}  ->  {y}/")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            entry.rename(dest)
        moved += 1
    return moved, skipped


def _find_plot_folders(root: Path) -> list[Path]:
    """Find every DataPlots[_linear] folder at or under `root`."""
    if root.is_file():
        return []
    if root.name in _TARGET_DIR_NAMES:
        return [root]
    found: list[Path] = []
    for name in _TARGET_DIR_NAMES:
        found.extend(d for d in root.rglob(name) if d.is_dir())
    return sorted(set(found))


def reorganize(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Reorganize one path (recursively finds all plot folders inside)."""
    folders = _find_plot_folders(path)
    if not folders:
        print(f"{path}: no DataPlots/ folders found")
        return 0, 0
    total_moved = 0
    total_skipped = 0
    for folder in folders:
        # If `folder` contains ext-subfolders (png/, svg/, pdf/), files live
        # inside those rather than at folder's top level. Reorganize each
        # ext-subfolder; otherwise reorganize folder itself.
        ext_children = [
            c for c in folder.iterdir()
            if c.is_dir() and c.name in _EXT_SUBDIR_NAMES
        ]
        if ext_children:
            for ext_dir in sorted(ext_children):
                moved, skipped = _reorganize_folder(ext_dir, dry_run)
                print(f"  {folder.relative_to(path) if folder != path else folder.name}"
                      f"/{ext_dir.name}: moved {moved}, skipped {skipped}")
                total_moved += moved
                total_skipped += skipped
        else:
            moved, skipped = _reorganize_folder(folder, dry_run)
            print(f"  {folder.relative_to(path) if folder != path else folder.name}"
                  f": moved {moved}, skipped {skipped}")
            total_moved += moved
            total_skipped += skipped
    return total_moved, total_skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="+", type=Path,
                   help="One or more paths to reorganize. Each is searched "
                        "recursively for DataPlots[_linear]/ folders.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print intended moves without touching the filesystem.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    grand_moved = 0
    grand_skipped = 0
    for path in args.paths:
        path = path.resolve()
        print(f"\n== {path} ==")
        m, s = reorganize(path, dry_run=args.dry_run)
        grand_moved += m
        grand_skipped += s

    verb = "would move" if args.dry_run else "moved"
    print(f"\nTOTAL: {verb} {grand_moved} files; skipped {grand_skipped}.")


if __name__ == "__main__":
    main()
