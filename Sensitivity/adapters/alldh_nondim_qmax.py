"""SobolModel adapter for the q_max-normalised AlLDH non-dim column
(`Model/AlLDH/diffrax_non_dim_qmax.py`).

Direct clone of Sensitivity/adapters/alldh_nondim.py with two changes:
  1. Imports `NonDimNumbers`/`run_cycle` from `diffrax_non_dim_qmax`, so
     `n* = q/q_max` and the isotherm is θC*/(1+θC*).
  2. Tags and output subdirs are suffixed `_qmax` so results don't collide
     with the feed-eq sibling adapter.

Bounds on the shared axes (Λ, Da, θ, C_thresh_ads, C_thresh_des) are kept
identical to the feed-eq adapter so the same Sobol sample maps to the same
NUMERIC parameter values in both adapters. The two adapters do NOT describe
the same physical bed: Λ_qmax = Λ_feed · (1+θ)/θ, so a given (log10_Λ,
log10_θ) draw represents different (ρ_p, q_max, c_feed) combinations under
the two bases. Interpret accordingly.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

from Sensitivity.sobol_driver import SobolModel

_log = logging.getLogger(__name__)

# Tiny floor applied to R_release in the denominator of the wash/release
# ratio so the metric stays finite when the bed never loaded (R_release → 0).
# Matches the existing 1e-12 clamp on R_outlet_des inside the model.
_R_RELEASE_FLOOR = 1.0e-12


# ---------------------------------------------------------------------------
# Bounds presets — same as alldh_nondim.py
# ---------------------------------------------------------------------------

_NAMES_LEGACY = [
    "Lambda", "Da", "theta",
    "C_thresh_ads", "C_thresh_des",
]
_BOUNDS_LEGACY = [
    (0.01,  10.0),
    (0.005, 10.0),
    (0.5,   10.0),
    (0.50,   0.99),    # C_thresh_ads: realistic operating cutoff before full BT
    (0.05,   0.18),
]
_PRETTY_LEGACY = [
    r"$\Lambda$", r"$Da$", r"$\Theta$",
    r"$C^*_{th,ads}$", r"$C^*_{th,des}$",
]

_NAMES_IX = [
    "log10_Lambda", "log10_Da", "log10_theta",
    "C_thresh_ads", "C_thresh_des",
]
_BOUNDS_IX = [
    (1.0,  2.5),
    (-1.1, 0.3),
    (-2.0, 0.3),     # log10 θ
    (0.50, 0.99),    # C_thresh_ads
    (0.05, 0.18),
]
_PRETTY_IX = [
    r"$\log_{10}\Lambda$", r"$\log_{10}Da$", r"$\log_{10}\Theta$",
    r"$C^*_{th,ads}$", r"$C^*_{th,des}$",
]

_NAMES_IX_LINEAR = _NAMES_LEGACY
_BOUNDS_IX_LINEAR = [
    (10.0 ** 1.0,  10.0 ** 2.5),
    (10.0 ** -1.1, 10.0 ** 0.3),
    (10.0 ** -2.0, 10.0 ** 0.3),     # θ ∈ [0.01, ~2.0]
    (0.50, 0.99),    # C_thresh_ads
    (0.05, 0.18),
]
_PRETTY_IX_LINEAR = _PRETTY_LEGACY

_NAMES_ALLDH_LOG = [
    "log10_Lambda", "log10_Da", "log10_theta",
    "C_thresh_ads", "C_thresh_des",
]
_BOUNDS_ALLDH_LOG = [
    (math.log10(0.01),  math.log10(10.0)),
    (math.log10(0.005), math.log10(10.0)),
    (math.log10(0.5),   0.3),    # log10 θ ∈ [-0.30, 0.3] (θ ∈ [0.5, ~2.0])
    (0.50, 0.99),    # C_thresh_ads
    (0.05, 0.18),
]
_PRETTY_ALLDH_LOG = [
    r"$\log_{10}\Lambda$", r"$\log_{10}Da$", r"$\log_{10}\Theta$",
    r"$C^*_{th,ads}$", r"$C^*_{th,des}$",
]


_BoundsMode = Literal["legacy", "alldh_log", "ix", "ix_linear"]


def _build(bounds_mode: _BoundsMode = "legacy") -> SobolModel:
    _log.debug("building AlLDH non-dim Sobol adapter (q_max, bounds=%s)",
               bounds_mode)

    if bounds_mode == "legacy":
        names, bounds, pretty = _NAMES_LEGACY, _BOUNDS_LEGACY, _PRETTY_LEGACY
        tag = "alldh_nondim_qmax"
        out_subdir = "alldh_nondim_qmax/v1"
    elif bounds_mode == "alldh_log":
        names, bounds, pretty = (
            _NAMES_ALLDH_LOG, _BOUNDS_ALLDH_LOG, _PRETTY_ALLDH_LOG,
        )
        tag = "alldh_nondim_qmax_alldhlog"
        out_subdir = "alldh_nondim_qmax_alldhlog/v1"
    elif bounds_mode == "ix":
        names, bounds, pretty = _NAMES_IX, _BOUNDS_IX, _PRETTY_IX
        tag = "alldh_nondim_qmax_ixbounds"
        out_subdir = "alldh_nondim_qmax_ixbounds/v1"
    elif bounds_mode == "ix_linear":
        names, bounds, pretty = (
            _NAMES_IX_LINEAR, _BOUNDS_IX_LINEAR, _PRETTY_IX_LINEAR,
        )
        tag = "alldh_nondim_qmax_ixbounds_linear"
        out_subdir = "alldh_nondim_qmax_ixbounds_linear/v1"
    else:
        raise ValueError(
            f"bounds_mode must be 'legacy', 'alldh_log', 'ix', or "
            f"'ix_linear', got {bounds_mode!r}"
        )

    def run(vec):
        import jax.numpy as jnp
        from diffrax_non_dim_qmax import NonDimNumbers, run_cycle  # q_max basis

        if bounds_mode in ("ix", "alldh_log"):
            log10_Lambda, log10_Da, log10_theta, c_th_ads, c_th_des = vec
            Lambda = 10.0 ** log10_Lambda
            Da     = 10.0 ** log10_Da
            theta  = 10.0 ** log10_theta
        else:
            Lambda, Da, theta, c_th_ads, c_th_des = vec

        c_th_des = min(c_th_des, 0.99 * c_th_ads)

        (tau_ads, tau_des, U_b, R_outlet_des, productivity,
         R_release, R_wash) = run_cycle(
            non_dim=NonDimNumbers(
                Da=Da, Lambda=Lambda, theta=theta,
            ),
            c_thresh_ads=jnp.asarray(c_th_ads),
            c_thresh_des=jnp.asarray(c_th_des),
        )
        r_release_f = float(R_release)
        r_wash_f    = float(R_wash)
        return {
            "tau_ads":      float(tau_ads),
            "tau_des":      float(tau_des),
            "U_b":          float(U_b),
            "R_outlet_des": float(R_outlet_des),
            "R_release":    r_release_f,
            "R_wash":       r_wash_f,
            "R_wash_over_R_release": r_wash_f / max(r_release_f, _R_RELEASE_FLOOR),
            "productivity": float(productivity),
        }

    def postfix(vec, result):
        if bounds_mode in ("ix", "alldh_log"):
            log10_Lambda, log10_Da, log10_theta, c_th_ads, c_th_des = vec
            Lambda = 10.0 ** log10_Lambda
            Da     = 10.0 ** log10_Da
            theta  = 10.0 ** log10_theta
        else:
            Lambda, Da, theta, c_th_ads, c_th_des = vec
        return {
            "Λ":     f"{Lambda:.2g}",
            "Da":    f"{Da:.2g}",
            "θ":     f"{theta:.2g}",
            "c_ads": f"{c_th_ads:.2f}",
            "c_des": f"{c_th_des:.2f}",
            "U_b":   f"{result['U_b']:.3g}",
        }

    repo_root = Path(__file__).resolve().parents[2]

    return SobolModel(
        tag=tag,
        names=names,
        bounds=bounds,
        dists=["unif"] * len(names),
        pretty_names=pretty,
        outputs={
            "tau_ads":      "Sobol — Adsorption time",
            "tau_des":      "Sobol — Desorption time",
            "U_b":          "Sobol — Bed utilisation (q_max basis)",
            "R_outlet_des": "Sobol — Li recovered (desorption)",
            "R_release":    "Sobol — Li released from adsorbed phase",
            "R_wash":       "Sobol — Solution-phase wash mass",
            "R_wash_over_R_release": "Sobol — Wash/Release ratio",
            "productivity": "Sobol — Productivity (R_outlet_des/τ_cycle)",
        },
        output_dir=repo_root / "Results" / "Sensitivity" / out_subdir,
        run=run,
        postfix=postfix,
    )


def get_model(bounds: _BoundsMode = "legacy") -> SobolModel:
    """Build the q_max-normalised AlLDH non-dim Sobol adapter.

    bounds : same presets as the feed-eq adapter; numeric ranges unchanged.
             Only the loading normalisation inside the forward model differs.
    """
    return _build(bounds)


model = _build("legacy")
