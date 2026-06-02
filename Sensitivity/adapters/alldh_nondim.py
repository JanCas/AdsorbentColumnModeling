"""SobolModel adapter for the AlLDH non-dim column (`Model/AlLDH/diffrax_non_dim.py`).

Bounds and pretty names are copied verbatim from the original
`Model/AlLDH/simbol_diffrax_non_dim.py:18-35`.  The model call is synced to
the *current* `run_cycle` signature (3 args, 5 returns) — `sec_star`,
`eta_p`, `Psi` from the legacy script are dropped because they are no
longer part of `run_cycle`.

Importing this module does not start a JAX session — the model import is
deferred into the `run` closure so the adapter is cheap to inspect.
"""

from __future__ import annotations

import logging
from pathlib import Path

from Sensitivity.sobol_driver import SobolModel

_log = logging.getLogger(__name__)


# Module-level (sibling-import) wiring is mirrored from
# Model/AlLDH/multi_optim.py:12 which does `from diffrax_column_model import run_model`.
# The entry script (Sensitivity/run_alldh.py) prepends Model/AlLDH to sys.path
# before triggering this import, so we can keep the AlLDH file layout untouched.


def _build() -> SobolModel:
    _log.debug("building AlLDH non-dim Sobol adapter")

    def run(vec):
        import jax.numpy as jnp
        from diffrax_non_dim import NonDimNumbers, run_cycle  # AlLDH

        Lambda, Da, theta, c_th_ads, c_th_des, epsilon = vec
        tau_ads, tau_des, U_b, R_outlet_des, productivity = run_cycle(
            non_dim=NonDimNumbers(
                Da=Da, Lambda=Lambda, theta=theta, epsilon=epsilon,
            ),
            c_thresh_ads=jnp.asarray(c_th_ads),
            c_thresh_des=jnp.asarray(c_th_des),
        )
        return {
            "tau_ads":      float(tau_ads),
            "tau_des":      float(tau_des),
            "U_b":          float(U_b),
            "R_outlet_des": float(R_outlet_des),
            "productivity": float(productivity),
        }

    def postfix(vec, result):
        Lambda, Da, theta, c_th_ads, c_th_des, epsilon = vec
        return {
            "Λ":     f"{Lambda:.2g}",
            "Da":    f"{Da:.2g}",
            "θ":     f"{theta:.2g}",
            "ε":     f"{epsilon:.2f}",
            "c_ads": f"{c_th_ads:.2f}",
            "c_des": f"{c_th_des:.2f}",
            "U_b":   f"{result['U_b']:.3g}",
        }

    repo_root = Path(__file__).resolve().parents[2]

    return SobolModel(
        tag="alldh_nondim",
        names=[
            "Lambda", "Da", "theta",
            "C_thresh_ads", "C_thresh_des", "epsilon",
        ],
        bounds=[
            (0.01,  10.0),    # Lambda  — sorbent/fluid capacity ratio
            (0.005, 10.0),    # Da      — Damkoehler number
            (0.5,   10.0),    # theta   — isotherm steepness
            (0.10,   0.8),    # C_thresh_ads — adsorption outlet cutoff
            (0.05,   0.18),   # C_thresh_des — desorption eluate cutoff
            (0.35,   0.5),    # epsilon — bed porosity
        ],
        dists=["unif"] * 6,
        pretty_names=[
            r"$\Lambda$", r"$Da$", r"$\Theta$",
            r"$C^*_{th,ads}$", r"$C^*_{th,des}$", r"$\varepsilon$",
        ],
        outputs={
            "tau_ads":      "Sobol — Adsorption time",
            "tau_des":      "Sobol — Desorption time",
            "U_b":          "Sobol — Bed utilisation",
            "R_outlet_des": "Sobol — Li recovered (desorption)",
            "productivity": "Sobol — Productivity (R_outlet_des/τ_cycle)",
        },
        output_dir=repo_root / "Results/Sensitivity/alldh_nondim/v1",
        run=run,
        postfix=postfix,
    )


model = _build()
