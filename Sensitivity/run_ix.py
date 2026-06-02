"""Sobol sensitivity run for the non-dim IX column.

PLACEHOLDER: this entry point will work once the non-dim IX model exists
and `Sensitivity/adapters/ix_nondim.py` is filled in.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the (future) non-dim IX module sibling-importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "Model" / "IX"))
sys.path.insert(0, str(_REPO_ROOT))

from Sensitivity.adapters.ix_nondim import get_model
from Sensitivity.sobol_driver import run_sobol
from utils.logging_setup import configure_logging

_log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=2 ** 11)
    p.add_argument("--second-order", action="store_true")
    args = p.parse_args()
    configure_logging()
    _log.info("starting IX Sobol run: N=%d second_order=%s",
              args.N, args.second_order)
    model = get_model()        # raises NotImplementedError until you fill in the adapter
    run_sobol(model, N=args.N, calc_second_order=args.second_order)


if __name__ == "__main__":
    main()
