import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Model.Multi_Ion_ALLDH import (
    ALLDHIsotherm,
    equilibrium_loading,
    equilibrium_occupancy,
    log_reaction_activity,
)
from utils.StreamData import Composition
from utils.pitzer import pitzer_terms, water_activity


def test_reaction_activity_matches_document_equation():
    composition = Composition(0.1, 2.0, 1.0)
    n_h2o = 3.0
    terms = pitzer_terms(
        composition.m_LiCl, composition.m_NaCl, composition.m_MgCl2
    )
    a_water = float(water_activity(terms.osmo_w, terms.m_tot))

    expected = (
        2.0 * float(terms.lngamma_LiCl)
        + np.log(float(composition.m_LiCl))
        + np.log(float(terms.m_Cl))
        + n_h2o * np.log(a_water)
    )

    got = float(log_reaction_activity(composition, n_h2o))
    assert got == pytest.approx(expected, rel=1e-13, abs=1e-15)


def test_composition_activities_match_pitzer_and_handle_pure_water():
    composition = Composition(m_LiCl=0.1, m_NaCl=2.0, m_MgCl2=1.0)
    terms = pitzer_terms(
        composition.m_LiCl, composition.m_NaCl, composition.m_MgCl2
    )
    expected_li_cl = (
        np.exp(2.0 * float(terms.lngamma_LiCl))
        * float(composition.m_LiCl)
        * float(terms.m_Cl)
    )
    expected_water = float(water_activity(terms.osmo_w, terms.m_tot))

    a_li_cl, a_water = jax.jit(lambda comp: comp.activities())(composition)
    assert float(a_li_cl) == pytest.approx(
        expected_li_cl, rel=1e-13, abs=1e-15
    )
    assert float(a_water) == pytest.approx(
        expected_water, rel=1e-13, abs=1e-15
    )
    assert float(composition.li_cl_activity()) == float(a_li_cl)
    assert float(composition.water_activity()) == float(a_water)

    pure_water = Composition(m_LiCl=0.0, m_NaCl=0.0, m_MgCl2=0.0)
    log_a_li_cl, log_a_water = pure_water.log_activities()
    assert np.isneginf(float(log_a_li_cl))
    assert float(log_a_water) == 0.0

    a_li_cl, a_water = jax.jit(lambda comp: comp.activities())(pure_water)
    assert float(a_li_cl) == 0.0
    assert float(a_water) == 1.0


def test_loading_matches_closed_form():
    composition = Composition(0.1, 2.0, 1.0)
    q_max = 3.98
    log_K_i = np.log(12.0)
    n_h2o = 2.0
    log_A_aq = float(log_reaction_activity(composition, n_h2o))
    K_A = np.exp(log_K_i + log_A_aq)
    expected = q_max * K_A / (1.0 + K_A)

    got = float(
        equilibrium_loading(composition, q_max, log_K_i, n_h2o)
    )

    assert got == pytest.approx(expected, rel=1e-13, abs=1e-15)
    assert 0.0 < got < q_max


def test_zero_li_is_exact_for_every_background():
    model = ALLDHIsotherm(q_max=3.98, log_K_i=np.log(12.0), n_h2o=2.0)

    for background in ((0.0, 0.0), (2.0, 1.0)):
        composition = Composition(0.0, *background)
        log_A_aq = float(log_reaction_activity(composition, model.n_h2o))
        assert np.isneginf(log_A_aq)
        assert float(model(composition)) == 0.0

    batched = jax.jit(jax.vmap(model))(
        Composition(
            jnp.array([0.0, 0.0, 0.1]),
            jnp.array([0.0, 2.0, 2.0]),
            jnp.array([0.0, 1.0, 1.0]),
        )
    )
    assert np.all(np.isfinite(np.asarray(batched)))
    assert np.all(np.asarray(batched[:2]) == 0.0)
    assert float(batched[2]) > 0.0
    zero_gradient = jax.grad(lambda m: model(Composition(m, 2.0, 1.0)))(0.0)
    assert np.isfinite(float(zero_gradient))


def test_occupancy_has_correct_limits_and_is_monotone():
    log_A = jnp.array([-jnp.inf, -10.0, 0.0, 10.0, jnp.inf])
    theta = np.asarray(equilibrium_occupancy(log_A, log_K_i=0.0))

    assert theta[0] == 0.0
    assert theta[-1] == 1.0
    assert np.all(np.diff(theta) > 0.0)


def test_water_activity_term_can_be_disabled():
    composition = Composition(0.1, 2.0, 1.0)
    terms = pitzer_terms(
        composition.m_LiCl, composition.m_NaCl, composition.m_MgCl2
    )
    log_a_water = np.log(float(water_activity(terms.osmo_w, terms.m_tot)))
    with_water = float(log_reaction_activity(composition, n_h2o=3.0))
    without_water = float(log_reaction_activity(composition, n_h2o=0.0))

    assert with_water - without_water == pytest.approx(
        3.0 * log_a_water, rel=1e-13, abs=1e-15
    )


def test_isotherm_supports_jit_vmap_and_grad():
    model = ALLDHIsotherm(q_max=3.98, log_K_i=np.log(12.0), n_h2o=2.0)
    scalar = jax.jit(lambda composition: model(composition))(
        Composition(0.1, 2.0, 1.0)
    )
    batched = jax.vmap(model)(
        Composition(
            jnp.array([0.05, 0.1, 0.2]),
            jnp.array([2.0, 2.0, 2.0]),
            jnp.array([1.0, 1.0, 1.0]),
        )
    )
    gradient = jax.grad(lambda m: model(Composition(m, 2.0, 1.0)))(0.1)

    assert np.isfinite(float(scalar))
    assert batched.shape == (3,)
    assert np.all(np.isfinite(np.asarray(batched)))
    assert np.isfinite(float(gradient))
