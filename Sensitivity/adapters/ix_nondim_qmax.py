"""SobolModel adapters for the q_max-normalised non-dim IX column
(`Model/IX/NonDim/ix_nondim_qmax.py`).

Direct clone of Sensitivity/adapters/ix_nondim.py with three changes:
  1. Imports `NonDimParams`/`default_params`/`qois` (and the t_max overrides)
     from `ix_nondim_qmax`, so the loading axis is n* = q/q_max and the
     isotherm is the bare mass-action form n*_eq = K*A*/(H* + K*A*).
  2. Tags and output subdirs are suffixed `_qmax` so results don't collide
     with the feed-eq sibling adapter.
  3. Proton chemistry uses a SINGLE buffer axis `log10_buffer` instead of ω
     and H_in_ads independently. B = ω/H_in_ads = [OH⁻]_in/A_in = |T*_in,load|
     is the feed-basicity buffer; the bed loads iff B > 1 − C_thresh_ads, and
     the QoIs depend only on this ratio (absolute ω adds ≈0 — verified on runs
     2.x/2.1/2.2). ω is fixed at K_w/A_IN_REF² (water-floor term in h_star,
     negligible direct effect) and H_in_ads = ω/B is derived per-sample.
     See Model/IX/NonDim/diagnostic_ph_gradient.py for the diagnostic.

Bounds presets ("ix", "alldh", "alldh_lowda") are otherwise identical to
the feed-eq adapter on shared axes — the same Sobol draw yields the same
NUMERIC parameter values in both adapters. Λ_qmax = Λ_feed · (H_in_ads+K*)/K*,
so the same (log10_Λ, log10_K*) draw represents a different physical bed
under the two bases.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

from Sensitivity.sobol_driver import SobolModel

_log = logging.getLogger(__name__)


# Proton chemistry is parameterized by a SINGLE buffer axis instead of ω and
# H_in_ads independently (they are coupled through A_in, and an analysis of
# runs 2.x/2.1/2.2 showed the QoIs depend only on their ratio — absolute ω
# adds ≈0 explanatory power once the ratio is known). The sampled axis is
#     B = ω / H_in_ads = [OH⁻]_in / A_in = |T*_in,load|   (feed-basicity buffer)
# The bed loads iff B > 1 − C_thresh_ads, so B is the load-relevant coordinate.
#
# ω itself is only needed for the water-floor term in h_star, where its
# absolute value is immaterial; it is FIXED at a reference (K_w / A_in_ref²),
# and H_in_ads is then DERIVED per-sample as H_in_ads = ω_ref / B.
_K_W       = 1.0e-14          # (mol/L)², water self-ionization at 25 °C
_A_IN_REF  = 0.05            # mol/L (≈ 347 mg/L Li); water-floor reference only
_OMEGA_REF = _K_W / _A_IN_REF ** 2   # = 4e-12, fixed (negligible direct effect)
# Screen-scope nominal buffer: mid of the transition-spanning band [-2, 0.5].
_BUFFER_NOMINAL = 10.0 ** (-0.75)    # ≈ 0.18

_NAMES_SCREEN = ["log10_Lambda", "log10_Da", "log10_K_star", "C_thresh_ads"]
_NAMES_FULL   = _NAMES_SCREEN + [
    "log10_H_in_des", "log10_buffer", "C_thresh_des",
]

# Bounds (the IX-tuned preset). See the feed-eq adapter for the rationale;
# numeric ranges match it axis-for-axis.
# Λ [50,1000]; Da [0.1,3]; K* [5e-10,2e-9]; H_in_des [2,10];
# H_in_ads [2e-11,2e-10]; C_thresh_ads [0.5,0.95]; C_thresh_des [0.01,0.05].
_BOUNDS_SCREEN = [
    (1.7,  3.0),     # log10 Λ        ([50, 1000])
    (-1.0, 0.5),     # log10 Da       ([0.1, 3])
    (-9.3, -8.7),    # log10 K*       ([5e-10, 2e-9])
    (0.50, 0.95),    # C_thresh_ads   ([0.5, 0.95])
]
_BOUNDS_FULL = _BOUNDS_SCREEN + [
    (0.3,  1.0),     # log10 H_in_des  ([2, 10])
    (-2.0, 0.5),     # log10 buffer    (B=ω/H_in_ads=[OH⁻]/A_in; spans no-load↔load)
    (0.01, 0.05),    # C_thresh_des    ([0.01, 0.05])
]

# AlLDH-matched bounds: AlLDH endpoints on shared axes (Λ, Da, C_thresh_ads,
# C_thresh_des); IX-only axes (K*, H_in_des, H_in_ads) keep the IX band.
_BOUNDS_SCREEN_ALLDH = [
    (math.log10(0.01),  math.log10(10.0)),
    (math.log10(0.005), math.log10(10.0)),
    (-9.3, -8.7),    # log10 K* (IX band — no AlLDH analog)
    (0.50, 0.99),    # C_thresh_ads
]
_BOUNDS_FULL_ALLDH = _BOUNDS_SCREEN_ALLDH + [
    (0.3,  1.0),     # log10 H_in_des  (IX default)
    (-2.0, 0.5),     # log10 buffer    (IX default)
    (0.05, 0.18),    # C_thresh_des    (AlLDH range)
]

_DA_LOWDA_UPPER = -0.58
_BOUNDS_SCREEN_ALLDH_LOWDA = [
    (math.log10(0.01),  math.log10(10.0)),
    (math.log10(0.005), _DA_LOWDA_UPPER),
    (-9.3, -8.7),    # log10 K* (IX band — no AlLDH analog)
    (0.50, 0.99),    # C_thresh_ads
]
_BOUNDS_FULL_ALLDH_LOWDA = _BOUNDS_SCREEN_ALLDH_LOWDA + [
    (0.3,  1.0),     # log10 H_in_des
    (-2.0, 0.5),     # log10 buffer
    (0.05, 0.18),    # C_thresh_des
]

_PRETTY_SCREEN = [
    r"$\log_{10}\Lambda$", r"$\log_{10}Da$", r"$\log_{10}K^*$",
    r"$C^*_{th,ads}$",
]
_PRETTY_FULL = _PRETTY_SCREEN + [
    r"$\log_{10}H^*_{in,des}$",
    r"$\log_{10}(\omega/H^*_{in,ads})$",
    r"$C^*_{th,des}$",
]

_H_IN_DES_NOMINAL     = 10.0 ** 0.65   # ≈ 4.5, mid of [2, 10]
_C_THRESH_DES_NOMINAL = 0.03

# Tiny floor applied to R_release in the denominator of the wash/release
# ratio so the metric stays finite when the bed never loaded (R_release → 0).
# Matches the existing 1e-12 clamp on R_outlet_des inside the model.
_R_RELEASE_FLOOR = 1.0e-12

_OUTPUTS = {
    "tau_ads":      "Sobol — Adsorption time",
    "tau_des":      "Sobol — Desorption time",
    "U_b":          "Sobol — Bed utilisation (q_max basis)",
    "R_outlet_des": "Sobol — A recovered (desorption)",
    "R_release":    "Sobol — A released from adsorbed phase",
    "R_wash":       "Sobol — Solution-phase wash mass",
    "R_wash_over_R_release": "Sobol — Wash/Release ratio",
    "productivity": "Sobol — Productivity (R_outlet_des/τ_cycle)",
    "H_out":           "Sobol — Outlet proton H* at end of desorption",
    "H_out_over_H_in": "Sobol — Outlet proton ratio H*_out / H*_in",
}


def _build(
    scope: Literal["screen", "full"],
    bounds_mode: Literal["ix", "alldh", "alldh_lowda"] = "ix",
    t_max_mult: float = 1.0,
) -> SobolModel:
    """Assemble the IX SobolModel for one scope + bounds preset.

    `scope` selects the sampled axes: "screen" sweeps the 4 core groups
    (Λ, Da, K*, C_thresh_ads) with the pH/threshold axes held at nominal
    values; "full" adds the 3 desorption/pH axes (H_in_des, buffer B,
    C_thresh_des). `bounds_mode` selects the sampling box (native IX band vs.
    AlLDH-matched endpoints). `t_max_mult` optionally stretches the per-phase
    integration horizons.
    """
    _log.debug("building IX non-dim Sobol adapter (q_max, scope=%s, bounds=%s, "
               "t_max_mult=%g)", scope, bounds_mode, t_max_mult)

    if bounds_mode == "ix":
        bounds_screen, bounds_full = _BOUNDS_SCREEN, _BOUNDS_FULL
        tag_suffix = ""
    elif bounds_mode == "alldh":
        bounds_screen, bounds_full = _BOUNDS_SCREEN_ALLDH, _BOUNDS_FULL_ALLDH
        tag_suffix = "_alldhbounds"
    elif bounds_mode == "alldh_lowda":
        bounds_screen, bounds_full = (
            _BOUNDS_SCREEN_ALLDH_LOWDA, _BOUNDS_FULL_ALLDH_LOWDA,
        )
        tag_suffix = "_alldhbounds_lowda"
    else:
        raise ValueError(
            f"bounds_mode must be 'ix', 'alldh', or 'alldh_lowda', "
            f"got {bounds_mode!r}"
        )
    if t_max_mult != 1.0:
        if t_max_mult <= 0:
            raise ValueError(f"t_max_mult must be > 0, got {t_max_mult!r}")
        tag_suffix = f"{tag_suffix}_tmult{t_max_mult:g}".replace(".", "p")

    if scope == "screen":
        names, bounds, pretty = _NAMES_SCREEN, bounds_screen, _PRETTY_SCREEN
    elif scope == "full":
        names, bounds, pretty = _NAMES_FULL, bounds_full, _PRETTY_FULL
    else:
        raise ValueError(f"scope must be 'screen' or 'full', got {scope!r}")

    _phase_cache: list = []

    def run(vec):
        """Map one Sobol sample row -> IX non-dim cycle -> QoI dict.

        Unpacks the sample into the dimensionless groups (Λ, Da, K*, plus the
        pH/threshold axes in `full` scope), builds `NonDimParams`, runs one
        load->desorb cycle via `qois`, and returns the metrics in `outputs`.
        """
        import jax.numpy as jnp
        from ix_nondim_qmax import NonDimParams, default_params, qois

        # Named lookup (order-independent) into the sample row. log-space axes
        # are exponentiated back to physical groups.
        values = dict(zip(names, vec))
        Lambda       = 10.0 ** values["log10_Lambda"]
        Da           = 10.0 ** values["log10_Da"]
        K_star       = 10.0 ** values["log10_K_star"]
        C_thresh_ads = values["C_thresh_ads"]
        # ω is fixed (water floor); the sampled buffer B = ω/H_in_ads sets the
        # feed basicity, and H_in_ads is derived from it.
        omega        = _OMEGA_REF
        # In "screen" scope the desorption/pH axes aren't sampled — hold them at
        # nominal so the 4 core axes carry all the variance.
        if scope == "full":
            H_in_des     = 10.0 ** values["log10_H_in_des"]
            buffer       = 10.0 ** values["log10_buffer"]
            C_thresh_des = values["C_thresh_des"]
        else:
            H_in_des     = _H_IN_DES_NOMINAL
            buffer       = _BUFFER_NOMINAL
            C_thresh_des = _C_THRESH_DES_NOMINAL
        # Derive the feed inlet proton level from the sampled buffer ratio.
        H_in_ads = omega / buffer      # B = ω/H_in_ads  ->  H_in_ads = ω/B
        # Keep the desorption cutoff strictly below the adsorption cutoff.
        C_thresh_des = min(C_thresh_des, 0.99 * C_thresh_ads)

        # Pack the groups into the model's parameter container.
        nd = default_params(
            Lambda=Lambda, Da=Da, K_star=K_star,
            omega=omega, H_in_des=H_in_des, H_in_ads=H_in_ads,
            C_thresh_ads=C_thresh_ads, C_thresh_des=C_thresh_des,
        )
        # Non-default integration horizons: build (and cache) stretched load/des
        # PhaseConfigs once, then reuse them for every sample in this sweep.
        if t_max_mult != 1.0:
            if not _phase_cache:
                from ix_nondim_qmax import (
                    _default_load_phase, _default_des_phase,
                    _T_MAX_LOAD_DEFAULT, _T_MAX_DES_DEFAULT,
                )
                _phase_cache.append(_default_load_phase(
                    t_max=_T_MAX_LOAD_DEFAULT * t_max_mult,
                ))
                _phase_cache.append(_default_des_phase(
                    t_max=_T_MAX_DES_DEFAULT * t_max_mult,
                ))
            metrics = qois(nd, load_phase=_phase_cache[0],
                           des_phase=_phase_cache[1])
        else:
            metrics = qois(nd)
        # Copy through every QoI the model returns (as Python floats).
        result = {key: float(metrics[key]) for key in _OUTPUTS if key in metrics}
        # Derived wash/release ratio (not in metrics dict). Floor denominator
        # so the metric stays finite when the bed never loaded.
        r_release = float(metrics["R_release"])
        r_wash    = float(metrics["R_wash"])
        result["R_wash_over_R_release"] = r_wash / max(r_release, _R_RELEASE_FLOOR)
        return result

    def postfix(vec, result):
        """Build the live tqdm postfix (current groups + productivity)."""
        values = dict(zip(names, vec))
        fields = {
            "Λ":     f"{10**values['log10_Lambda']:.2g}",
            "Da":    f"{10**values['log10_Da']:.2g}",
            "K*":    f"{10**values['log10_K_star']:.2g}",
            "c_ads": f"{values['C_thresh_ads']:.2f}",
        }
        if scope == "full":
            fields["H_des"] = f"{10**values['log10_H_in_des']:.2g}"
            fields["buf"]   = f"{10**values['log10_buffer']:.2g}"
            fields["c_des"] = f"{values['C_thresh_des']:.2f}"
        fields["prod"] = f"{result['productivity']:.3g}"
        return fields

    repo_root = Path(__file__).resolve().parents[2]
    tag = f"ix_nondim_qmax_{scope}{tag_suffix}"

    return SobolModel(
        tag=tag,
        names=names,
        bounds=bounds,
        dists=["unif"] * len(names),
        pretty_names=pretty,
        outputs=_OUTPUTS,
        output_dir=repo_root / "Results" / "Sensitivity" / tag / "v1",
        run=run,
        postfix=postfix,
    )


def get_screen_model(
    bounds: Literal["ix", "alldh", "alldh_lowda"] = "ix",
    t_max_mult: float = 1.0,
) -> SobolModel:
    """Screen-scope IX adapter: 4 core axes (Λ, Da, K*, C_thresh_ads)."""
    return _build("screen", bounds, t_max_mult)


def get_full_model(
    bounds: Literal["ix", "alldh", "alldh_lowda"] = "ix",
    t_max_mult: float = 1.0,
) -> SobolModel:
    """Full-scope IX adapter: core axes plus H_in_des, buffer B, C_thresh_des."""
    return _build("full", bounds, t_max_mult)


def get_model() -> SobolModel:
    """Default IX adapter used by the joint runner (full scope, IX bounds)."""
    return get_full_model()
