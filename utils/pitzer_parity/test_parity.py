"""Differential test: the JAX port of `pitzer_mix` against the MATLAB original.

Spec: pitzer-parity-test-cases.md. Same inputs, same outputs. The MATLAB is the oracle and
no expected values appear anywhere in this file — every assertion compares the two
implementations to each other.

    python -m pytest utils/pitzer_parity/test_parity.py -v
    python -m pytest utils/pitzer_parity/test_parity.py --oracle-backend oct2py
    python -m pytest utils/pitzer_parity/test_parity.py --regenerate-oracle

`lngamma` is compared, never `gamma`: the logs are the actual output, and exponentiating
amplifies discrepancies non-uniformly.
"""

import numpy as np
import pytest

import oracle
import cases
from cases import CASES, CLOSED_FORM, TERM_NAMES, X_SWEEP


# --- tolerances (spec sec.3) ---------------------------------------------------------
#
# Tier 1 is closed-form algebra: identical IEEE operations on both sides with nothing lost
# in transfer, so agreement should be near-exact. Anything in that tier off by more than
# 1e-13 is a transcription error, not floating-point noise, and widening the tolerance to
# make it pass hides the bug this suite exists to find. Confirmed empirically: Octave and
# MATLAB agree on every tier-1 term to 0 or 1 ulp.
#
# The looser tiers exist only because the implementations use different quadrature for
# J0/J1. That is the sole legitimate source of disagreement in this function.

_CLOSED_FORM_TOL = (1e-13, 1e-15)
_QUAD_TOL = ((0.1, 1e-8, 1e-14), (0.01, 1e-6, 1e-13), (0.0, 1e-4, 1e-10))


def _tolerance(name: str, I_tot: float) -> tuple[float, float]:
    """(rtol, atol) for one intermediate at one ionic strength."""
    if name in CLOSED_FORM:
        return _CLOSED_FORM_TOL
    for floor, rtol, atol in _QUAD_TOL:
        if I_tot >= floor:
            return rtol, atol
    raise AssertionError(f"unreachable: I_tot={I_tot}")


def _j_tolerance(x: float) -> tuple[float, float]:
    """(rtol, atol) for a standalone J0/J1 evaluation.

    The spec tiers the quadrature-dependent quantities by ionic strength, but J0/J1 are
    functions of X alone. X = 6|z_i z_j| A_phi sqrt(I), so the like-charged Li-Li pair
    (|z_i z_j| = 1) inverts to I = (X / (6 A_phi))^2 — the lowest ionic strength that can
    produce this X, hence the most permissive tier that could legitimately apply.
    """
    from utils.pitzer import _A_PHI

    return _tolerance("J0", (x / (6.0 * _A_PHI)) ** 2)


def _matches(got: float, want: float, rtol: float, atol: float) -> bool:
    """Agreement test that treats NaN and Inf as values rather than as failures.

    The degenerate subgroup exists to assert the shared NaN behaviour: I_tot = 0 makes the
    1/X prefactor of the J integrals singular and m_tot = 0 makes 2/m_tot singular, and
    the port must reproduce that rather than paper over it.
    """
    if np.isnan(got) or np.isnan(want):
        return np.isnan(got) and np.isnan(want)
    if np.isinf(got) or np.isinf(want):
        return got == want
    return abs(got - want) <= atol + rtol * abs(want)


# Points where GNU Octave's adaptive quadrature, not the port, is the odd one out.
#
# Octave is a stand-in for the oracle the spec actually names, and it is a good one: it
# agrees with MATLAB to <= 1.4e-11 on every quadrature-dependent term at I >= 0.1 except
# the entries below. Attribution at (0.05, 0.05, 0.05), tert_phi:
#
#     MATLAB  2.9289950027115431e-05
#     JAX     2.9289950028380546e-05    rel 4.3e-11 vs MATLAB
#     Octave  2.9289949607121666e-05    rel 1.4e-08 vs MATLAB
#
# The port sits 330x closer to MATLAB than Octave does, so the tier is not the problem and
# must not be widened — the spec is explicit that differing quadrature is the sole
# legitimate source of disagreement, and this is exactly that. Recorded as an xfail rather
# than silently tolerated so it stays visible and stays attributed.
_OCTAVE_QUADRATURE_OUTLIERS: frozenset[tuple[str, str]] = frozenset({
    ("ternary-0.05_0.05_0.05", "tert_phi"),
})


def _report(name: str, got: float, want: float, rtol: float, atol: float) -> str:
    denom = abs(want) if want != 0 else 1.0
    return (
        f"{name}: jax={got!r} oracle={want!r} "
        f"absdiff={abs(got - want):.3e} reldiff={abs(got - want) / denom:.3e} "
        f"(rtol={rtol:g} atol={atol:g})"
    )


# --- oracle data ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def oracle_data(pytestconfig):
    """(terms, j_sweep) from the oracle: cached golden CSV, or a live re-run."""
    backend = pytestconfig.getoption("--oracle-backend")
    if pytestconfig.getoption("--regenerate-oracle"):
        # Subprocess, not oracle.generate(): JAX is already loaded in this process.
        return oracle.regenerate_via_subprocess(backend)
    return oracle.load_cache(backend)


@pytest.fixture(scope="session")
def jax_terms():
    """(n_cases, 23) float64 of every intermediate, from the JAX port.

    Scalar calls in a loop rather than a single vmap, so what is under test is exactly the
    call signature a user writes.
    """
    from utils.pitzer import pitzer_terms

    rows = []
    for _, a, b, c in CASES:
        t = pitzer_terms(a, b, c)
        rows.append([float(getattr(t, n)) for n in TERM_NAMES])
    return np.array(rows, dtype=np.float64)


# --- 1. J0 and J1 on their own (spec sec.1) ------------------------------------------
#
# Declared first, and run first, deliberately. These are pure one-argument functions and
# everything else depends on them; if either differs, the rest of the suite tells you
# nothing about where the problem is.


@pytest.mark.parametrize("i_x", range(len(X_SWEEP)), ids=[f"X={x:g}" for x in X_SWEEP])
def test_j_integrals(i_x, oracle_data):
    from utils.pitzer import j_integrals

    _, j_sweep = oracle_data
    x_oracle, j0_oracle, j1_oracle = j_sweep[i_x]

    x = X_SWEEP[i_x]
    assert x == pytest.approx(x_oracle, rel=0, abs=0), (
        "the oracle was run on a different X grid than this suite — regenerate it"
    )

    j0, j1 = j_integrals(x)
    rtol, atol = _j_tolerance(x)

    assert _matches(float(j0), j0_oracle, rtol, atol), _report("J0", float(j0), j0_oracle, rtol, atol)
    assert _matches(float(j1), j1_oracle, rtol, atol), _report("J1", float(j1), j1_oracle, rtol, atol)


# --- 2. the full input grid (spec sec.2) ---------------------------------------------


@pytest.mark.parametrize(
    "i_case", range(len(CASES)), ids=[cases.case_id(i) for i in range(len(CASES))]
)
def test_case(i_case, oracle_data, jax_terms, pytestconfig):
    """Compare all 23 intermediates in dependency order; report only the first mismatch.

    Endpoint-only comparison leaves you with one wrong number and twenty candidate causes.
    Asserting down the dependency chain means the first failure is the cause and
    everything after it is downstream contamination, so only the first is worth reporting.
    """
    terms_oracle, _ = oracle_data
    want_row = terms_oracle[i_case]
    got_row = jax_terms[i_case]

    I_tot = want_row[TERM_NAMES.index("I_tot")]
    if not np.isfinite(I_tot):
        I_tot = 0.0  # degenerate cases fall into the most permissive tier

    case = cases.case_id(i_case)
    backend = pytestconfig.getoption("--oracle-backend")

    for name, got, want in zip(TERM_NAMES, got_row, want_row):
        rtol, atol = _tolerance(name, I_tot)
        if _matches(got, want, rtol, atol):
            continue
        if backend == "oct2py" and (case, name) in _OCTAVE_QUADRATURE_OUTLIERS:
            pytest.xfail(f"known Octave quadrature outlier: {_report(name, got, want, rtol, atol)}")
        pytest.fail(
            f"case {case} (I_tot={I_tot:g})\n"
            f"  first mismatch in dependency order — everything after it is downstream\n"
            f"  {_report(name, got, want, rtol, atol)}"
        )


# --- 3. the port is usable as a JAX function -----------------------------------------


def test_term_names_match():
    """cases.TERM_NAMES must stay in lockstep with the PitzerTerms field order.

    cases.py keeps its own copy because oracle.py needs the column order and must not
    import JAX. If the two drift, every column of the oracle CSV silently shifts and the
    whole comparison comes apart while still reporting numbers.
    """
    from utils.pitzer import TERM_NAMES as core_names

    assert core_names == TERM_NAMES, (
        "PitzerTerms field order and cases.TERM_NAMES have diverged — oracle CSV columns "
        f"would be misaligned.\n  utils.pitzer: {core_names}\n  cases.py:    {TERM_NAMES}"
    )


def test_pitzer_mix_matches_terms():
    """The 4-output wrapper returns exactly the corresponding PitzerTerms fields."""
    from utils.pitzer import pitzer_mix, pitzer_terms

    m = (0.1, 2.0, 1.0)
    t = pitzer_terms(*m)
    got = pitzer_mix(*m)
    want = (t.lngamma_LiCl, t.lngamma_NaCl, t.lngamma_MgCl2, t.osmo_w)
    assert all(float(g) == float(w) for g, w in zip(got, want))


def test_vmap_matches_scalar():
    """Batching must not change the numbers — the model calls this under vmap."""
    import jax
    import jax.numpy as jnp
    from utils.pitzer import pitzer_mix

    triples = cases.triples()[:32]
    batched = jax.vmap(pitzer_mix)(
        jnp.asarray(triples[:, 0]), jnp.asarray(triples[:, 1]), jnp.asarray(triples[:, 2])
    )
    for k, (a, b, c) in enumerate(triples):
        for out_b, out_s in zip(batched, pitzer_mix(a, b, c)):
            assert _matches(float(out_b[k]), float(out_s), 1e-13, 1e-15)


def test_grad_is_finite():
    """Differentiability is the point of the port; fixed-node quadrature preserves it."""
    import jax
    from utils.pitzer import pitzer_mix

    for i in range(4):
        g = jax.grad(lambda a, b, c: pitzer_mix(a, b, c)[i], argnums=(0, 1, 2))(0.1, 2.0, 1.0)
        assert all(np.isfinite(float(v)) for v in g), f"non-finite gradient of output {i}"


def test_float64_is_enabled():
    """Tier-1 tolerance is 1e-13, which float32 cannot express."""
    import jax.numpy as jnp
    from utils.pitzer import pitzer_mix

    assert pitzer_mix(0.1, 2.0, 1.0)[0].dtype == jnp.float64
