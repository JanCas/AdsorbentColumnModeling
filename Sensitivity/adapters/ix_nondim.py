"""SobolModel adapter for the non-dim IX (ion-exchange) column. PLACEHOLDER.

The non-dim IX model does not exist yet. Once it lands (target file:
`Model/IX/<ix_nondim_module>.py` exposing a `run_cycle` returning scalar
metrics), fill in the four pieces marked below and the entry script
`Sensitivity/run_ix.py` will just work.

Reference adapter: `Sensitivity/adapters/alldh_nondim.py` — same shape,
already wired against the AlLDH non-dim model.
"""

from __future__ import annotations

from pathlib import Path

from Sensitivity.sobol_driver import SobolModel


def _build() -> SobolModel:
    raise NotImplementedError(
        "Non-dim IX model not yet implemented. "
        "Mirror Sensitivity/adapters/alldh_nondim.py once "
        "Model/IX/<nondim_module>.py exposes a scalar-returning run_cycle: "
        "fill in `names`, `bounds`, `pretty_names`, `outputs`, and the "
        "`run(vec) -> dict` closure that calls it."
    )


# Build lazily so `import Sensitivity.adapters.ix_nondim` doesn't raise.
# The entry script (Sensitivity/run_ix.py) calls get_model() to trigger _build.
def get_model() -> SobolModel:
    return _build()
