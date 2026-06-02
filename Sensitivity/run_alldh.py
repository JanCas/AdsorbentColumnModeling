"""Sobol sensitivity run for the AlLDH non-dim column.

Usage (from repo root):
    python -m Sensitivity.run_alldh
    python -m Sensitivity.run_alldh --N 64        # quick sanity run

The default N=2**11 reproduces the cadence of the legacy
Model/AlLDH/simbol_diffrax_non_dim.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Sibling-import path so `from diffrax_non_dim import run_cycle, NonDimNumbers`
# inside the adapter resolves. Mirrors Model/AlLDH/multi_optim.py:12 style.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "Model" / "AlLDH"))
sys.path.insert(0, str(_REPO_ROOT))

from Sensitivity.adapters.alldh_nondim import model
from Sensitivity.sobol_driver import run_sobol
from utils.logging_setup import configure_logging

_log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=2 ** 11,
                   help="SALib base sample count (default: 2**11 = 2048)")
    p.add_argument("--second-order", action="store_true",
                   help="Compute second-order Sobol indices.")
    args = p.parse_args()
    configure_logging()
    _log.info("starting AlLDH Sobol run: N=%d second_order=%s",
              args.N, args.second_order)
    run_sobol(model, N=args.N, calc_second_order=args.second_order)


if __name__ == "__main__":
    main()
