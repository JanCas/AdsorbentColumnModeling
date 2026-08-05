"""Validation for the osmotic-coefficient -> water-activity conversion.

`water_activity` has no counterpart in `pitzer-kim functions.m`, so unlike everything in
test_parity.py it has no MATLAB oracle and this is not a differential test. It is validated
two other ways:

  1. Against an independent implementation, isolating the conversion from the model. pyEQL's
     own osmotic coefficient is pushed through our conversion and must reproduce pyEQL's own
     water activity. This tests the relation and nothing else — if our Pitzer parameters
     disagreed with pyEQL's, this test would still pass, which is exactly what makes it a
     test *of the conversion*.
  2. Against the saturated-salt relative-humidity standards (Greenspan 1977), which are
     primary reference data. This one does exercise the whole chain, model included.

    python -m pytest test_water_activity.py -v
"""

import numpy as np
import pytest

import cases
from pitzer import pitzer_terms, water_activity, water_activity_of

M_W = 0.01801528  # kg/mol

# Greenspan, J. Res. NBS 81A (1977): equilibrium RH over saturated salt at 25 C, with the
# saturation molality. These are the reference points humidity sensors are calibrated on.
HUMIDITY_STANDARDS = [
    ("NaCl", (0.0, 6.15, 0.0), 0.7529),
    ("MgCl2", (0.0, 0.0, 5.81), 0.3278),
]

# Single salts only — see test_mixtures_diverge_from_pyeql for why mixtures are excluded.
SINGLE_SALT = [(0, 1, 0), (0, 3, 0), (0, 6, 0), (0, 0, 1), (0, 0, 3), (0, 0, 5),
               (1, 0, 0), (6, 0, 0)]


def _pyeql_solution(a, b, c):
    Solution = pytest.importorskip("pyEQL").Solution
    m = {"Li+": a, "Na+": b, "Mg+2": c, "Cl-": a + b + 2 * c}
    return Solution({k: f"{v} mol/kg" for k, v in m.items() if v > 0}, temperature="298.15 K")


# --- 1. the conversion itself, isolated from the model -------------------------------


@pytest.mark.parametrize("m", SINGLE_SALT + [(0.1, 2, 1), (0.05, 3, 2), (1, 3, 3)],
                         ids=lambda m: f"{m[0]:g}_{m[1]:g}_{m[2]:g}")
def test_conversion_reproduces_independent_implementation(m):
    """pyEQL's own phi through our conversion must give pyEQL's own water activity.

    Deliberately uses pyEQL's phi rather than ours, so a disagreement in the Pitzer
    parameters cannot mask or cause a failure here. Only the relation is under test.
    """
    s = _pyeql_solution(*m)
    m_tot = float(pitzer_terms(*m).m_tot)

    got = float(water_activity(float(s.get_osmotic_coefficient()), m_tot))
    want = float(s.get_water_activity())

    assert got == pytest.approx(want, abs=1e-4), (
        f"conversion disagrees with pyEQL at {m}: {got:.6f} vs {want:.6f}"
    )


def test_pure_water_is_unity():
    """Zero solute means unit activity, exactly."""
    assert float(water_activity(1.0, 0.0)) == 1.0


def test_wrapper_matches_explicit_call():
    """water_activity_of is a wrapper, not a second implementation."""
    for m in [(0.1, 2.0, 1.0), (0.0, 3.0, 0.0), (1.0, 3.0, 3.0)]:
        t = pitzer_terms(*m)
        assert float(water_activity_of(*m)) == float(water_activity(t.osmo_w, t.m_tot))


# --- 2. the whole chain against primary reference data --------------------------------


@pytest.mark.parametrize("name,m,a_w_std", HUMIDITY_STANDARDS, ids=[s[0] for s in HUMIDITY_STANDARDS])
def test_saturated_salt_humidity_standards(name, m, a_w_std):
    """Model + conversion vs the Greenspan saturated-salt RH standards.

    Tolerance is 0.005 in activity (0.5 % RH). This exercises the Pitzer parameters too,
    so it is the weaker test of the conversion but the stronger test of the model.
    """
    got = float(water_activity_of(*m))
    assert got == pytest.approx(a_w_std, abs=5e-3), (
        f"{name} saturated: a_w = {got:.4f}, standard = {a_w_std:.4f}"
    )


@pytest.mark.parametrize("m", SINGLE_SALT, ids=lambda m: f"{m[0]:g}_{m[1]:g}_{m[2]:g}")
def test_single_salt_agrees_with_pyeql(m):
    """End to end against pyEQL for single salts, where the two models are consistent."""
    got = float(water_activity_of(*m))
    want = float(_pyeql_solution(*m).get_water_activity())
    assert got == pytest.approx(want, abs=2e-3), f"{m}: {got:.4f} vs pyEQL {want:.4f}"


# --- 3. structural properties ---------------------------------------------------------


def test_monotonically_decreasing():
    """Adding salt must lower the water activity, for each salt independently."""
    for axis in range(3):
        prev = 1.1
        for level in (0.1, 0.5, 1.0, 2.0, 3.0, 4.0):
            m = [0.0, 0.0, 0.0]
            m[axis] = level
            a_w = float(water_activity_of(*m))
            assert a_w < prev, f"a_w not decreasing on axis {axis} at m={level}"
            prev = a_w


def test_bounded_on_the_parity_grid():
    """0 < a_w <= 1 everywhere the model produces a finite osmotic coefficient."""
    for _, a, b, c in cases.CASES:
        t = pitzer_terms(a, b, c)
        if not np.isfinite(float(t.osmo_w)):
            continue  # degenerate cases are NaN by construction; covered in test_parity
        a_w = float(water_activity_of(a, b, c))
        assert 0.0 < a_w <= 1.0, f"a_w = {a_w} out of range at ({a}, {b}, {c})"


def test_jit_vmap_grad():
    """Same JAX requirements as the rest of the model."""
    import jax
    import jax.numpy as jnp

    triples = cases.triples()[:16]
    batched = jax.vmap(water_activity_of)(
        jnp.asarray(triples[:, 0]), jnp.asarray(triples[:, 1]), jnp.asarray(triples[:, 2])
    )
    assert batched.shape == (16,)

    g = jax.grad(water_activity_of, argnums=(0, 1, 2))(0.1, 2.0, 1.0)
    assert all(np.isfinite(float(v)) for v in g)
    # More salt, less available water: every partial derivative must be negative.
    assert all(float(v) < 0 for v in g), f"expected d(a_w)/dm < 0, got {g}"
