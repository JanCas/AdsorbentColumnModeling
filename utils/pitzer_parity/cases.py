"""Input grid for the pitzer_mix parity test. Spec: pitzer-parity-test-cases.md sec.2.

This module is the single source of truth for the inputs. The spec is explicit that the
triples are generated once in Python and the same values fed to both sides — generating
them independently in each language reintroduces exactly the transcription risk the suite
exists to catch.

The subgroup structure is diagnostic, not decorative. Single-salt cases zero out the
mixing terms entirely, binaries activate exactly one theta/psi pair, ternaries turn
everything on. A failure confined to one subgroup names the broken term before you read
any code.

Pure numpy — no JAX, no RNG, no I/O beyond the explicit writer.
"""

import numpy as np

# --- the intermediates to compare, in dependency order (spec sec.1) ---
#
# Ordering is load-bearing: the suite asserts down this list and stops at the first
# mismatch, so the first failure is the cause and everything after it is downstream
# contamination.
#
# This is a deliberate copy of utils.pitzer.TERM_NAMES (the PitzerTerms field order). It
# lives here because oracle.py needs the column order and must not import JAX — oct2py
# forks a pty, which Python flags as a deadlock hazard against JAX's thread pool.
# test_parity.py::test_term_names_match asserts the copy has not drifted from the original.
# Every name is also a local of the MATLAB, verbatim, which is how the driver harvests them.
TERM_NAMES: tuple[str, ...] = (
    # composition and Debye-Huckel
    "I_tot", "Z_sum", "m_Cl", "m_tot", "A_phi",
    # unsymmetric mixing
    "thetaE_LiMg", "dthetaE_LiMg", "Phi_LiMg", "Phiphi_LiMg",
    # binary interaction, activity form
    "g_chi", "dg_chi", "B_LiCl", "dB_LiCl", "C_LiCl", "BC_LiCl",
    # osmotic assembly
    "phi_DH", "bin_phi", "tert_phi", "F_mix", "osmo_w",
    # outputs
    "lngamma_LiCl", "lngamma_NaCl", "lngamma_MgCl2",
)

# Tier 1 of the tolerance table (spec sec.3): closed-form algebra, identical IEEE
# operations on both sides with nothing lost in transfer. Everything else depends on the
# J quadrature and gets an ionic-strength-dependent tier.
CLOSED_FORM: frozenset[str] = frozenset({
    "I_tot", "Z_sum", "m_Cl", "m_tot", "A_phi",
    "g_chi", "dg_chi", "B_LiCl", "dB_LiCl", "C_LiCl", "BC_LiCl",
    "phi_DH", "bin_phi",
})

# --- J0/J1 standalone sweep (spec sec.1) ---
# Compared before anything else: J0 and J1 are pure one-argument functions and everything
# else depends on them, so if either differs the rest of the suite tells you nothing.
X_SWEEP: tuple[float, ...] = (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def _single_salt() -> list[tuple[float, float, float]]:
    """One salt at a time; all mixing terms are identically zero."""
    out = []
    for m in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        out += [(m, 0.0, 0.0), (0.0, m, 0.0), (0.0, 0.0, m)]
    return out


def _binary() -> list[tuple[float, float, float]]:
    """All three pairs on a 3x3 grid; exactly one theta/psi pair active at a time."""
    lv = (0.1, 1.0, 3.0)
    out = []
    for a in lv:
        for b in lv:
            out += [(a, b, 0.0), (a, 0.0, b), (0.0, a, b)]
    return out


def _ternary() -> list[tuple[float, float, float]]:
    """Full 3-factor grid; the only subgroup where the cross terms are all active."""
    lv = (0.05, 0.5, 2.0, 4.0)
    return [(a, b, c) for a in lv for b in lv for c in lv]


# The regime that actually matters for the ALLDH brine.
_TARGET_BRINE = [(0.1, 2.0, 1.0), (0.05, 3.0, 2.0), (0.02, 4.0, 1.5)]

# One species near zero while the others are high.
_TRACE = [(1e-4, 3.0, 2.0), (3.0, 1e-4, 1e-4)]

# Ill-conditioned regime: J0 is a cancellation of two O(1) terms here (spec sec.5).
_LOW_I = [(1e-3, 0.0, 0.0), (1e-4, 1e-4, 1e-4)]

# Upper end of the parameterisation.
_HIGH_I = [(0.0, 4.0, 2.0), (1.0, 3.0, 3.0)]

# NaN behaviour: I_tot = 0 makes the J prefactor 1/X singular, m_tot = 0 makes 2/m_tot
# singular. Both sides must produce NaN by the same route.
_DEGENERATE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


_SUBGROUPS: dict[str, list[tuple[float, float, float]]] = {
    "single": _single_salt(),
    "binary": _binary(),
    "ternary": _ternary(),
    "brine": _TARGET_BRINE,
    "trace": _TRACE,
    "low_I": _LOW_I,
    "high_I": _HIGH_I,
    "degenerate": _DEGENERATE,
}


def _build() -> list[tuple[str, float, float, float]]:
    """Flatten the subgroups, dropping duplicates but keeping first-seen order.

    Order is fixed and deterministic: the oracle CSV is indexed by row, so a reordering
    here silently misaligns the comparison.
    """
    seen: set[tuple[float, float, float]] = set()
    out: list[tuple[str, float, float, float]] = []
    for group, triples in _SUBGROUPS.items():
        for t in triples:
            if t in seen:
                continue
            seen.add(t)
            out.append((group, *t))
    return out


CASES: tuple[tuple[str, float, float, float], ...] = tuple(_build())


def triples() -> np.ndarray:
    """(n_cases, 3) float64 array of (m_LiCl, m_NaCl, m_MgCl2)."""
    return np.array([c[1:] for c in CASES], dtype=np.float64)


def subgroups() -> tuple[str, ...]:
    """Subgroup label per row of `triples()`."""
    return tuple(c[0] for c in CASES)


def case_id(idx: int) -> str:
    """Stable pytest parameter id, e.g. `ternary-0.5_2_4`."""
    group, a, b, c = CASES[idx]
    return f"{group}-{a:g}_{b:g}_{c:g}"


def write_inputs(path, header: bool = True) -> None:
    """Write the triples as %.17g CSV, which round-trips float64 exactly.

    `header=False` produces the headerless form the Octave/MATLAB driver reads with
    fscanf, which is the one CSV reader that behaves identically in both.
    """
    arr = triples()
    with open(path, "w") as fh:
        if header:
            fh.write("m_LiCl,m_NaCl,m_MgCl2\n")
        for row in arr:
            fh.write(",".join(f"{v:.17g}" for v in row) + "\n")


if __name__ == "__main__":
    import collections

    counts = collections.Counter(subgroups())
    print(f"{len(CASES)} cases")
    for k, v in counts.items():
        print(f"  {k:12s} {v:4d}")
    print(f"J sweep: {len(X_SWEEP)} values of X")
