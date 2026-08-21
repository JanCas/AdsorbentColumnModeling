import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from Model.Multi_Ion_ALLDH import (
    ALLDHIsotherm,
    AqueousConcentration,
    ColumnParams,
    ColumnState,
    column_rhs,
    initial_state,
    make_adsorption_event,
    make_desorption_event,
    simulate_column,
    simulate_cycle,
)
from utils.StreamData import Composition


def _params(N=6, k2=0.2):
    return ColumnParams(
        L=0.01,
        eps=0.4,
        rho_p=1000.0,
        water_mass_concentration=1000.0,
        u_s=1e-3,
        k2=k2,
        inlet=AqueousConcentration(100.0, 2000.0, 1000.0),
        isotherm=ALLDHIsotherm(3.98, jnp.log(12.0), 2.0),
        N=N,
    )


def test_rhs_conserves_li_and_only_li_sorbs():
    params = _params()
    state = ColumnState(
        c_LiCl=jnp.linspace(10.0, 80.0, params.N),
        c_NaCl=jnp.full(params.N, 2000.0),
        c_MgCl2=jnp.full(params.N, 1000.0),
        q=jnp.linspace(0.0, 0.5, params.N),
    )
    rate = column_rhs(0.0, state, params)

    assert np.allclose(rate.c_NaCl, 0.0)
    assert np.allclose(rate.c_MgCl2, 0.0)

    dx = float(params.dx)
    inventory_rate = (
        float(params.eps) * dx * np.sum(rate.c_LiCl)
        + float((1.0 - params.eps) * params.rho_p) * dx * np.sum(rate.q)
    )
    boundary_flux = float(
        params.u_s * (params.inlet.c_LiCl - state.c_LiCl[-1])
    )
    assert inventory_rate == pytest.approx(boundary_flux, rel=2e-13, abs=1e-15)


def test_rhs_uses_reversible_pseudo_second_order_kinetics():
    params = _params(N=4, k2=0.2)
    state = initial_state(
        params, AqueousConcentration(50.0, 2000.0, 1000.0), q0=0.1
    )
    rate = column_rhs(0.0, state, params)
    q_eq = float(params.isotherm(Composition(0.05, 2.0, 1.0)))

    driving_force = q_eq - 0.1
    assert np.asarray(rate.q) == pytest.approx(
        np.full(params.N, params.k2 * driving_force * abs(driving_force))
    )


def test_diffrax_simulation_is_finite_and_loads_the_bed():
    params = _params(N=4)
    y0 = initial_state(params, AqueousConcentration(0.0, 2000.0, 1000.0))
    sol = simulate_column(
        params,
        1.0,
        y0=y0,
        save_ts=jnp.linspace(0.0, 1.0, 5),
    )

    assert sol.ys.q.shape == (5, params.N)
    assert all(
        np.all(np.isfinite(np.asarray(x)))
        for x in (
            sol.ys.c_LiCl,
            sol.ys.c_NaCl,
            sol.ys.c_MgCl2,
            sol.ys.q,
        )
    )
    assert np.all(np.asarray(sol.ys.q[-1]) >= 0.0)
    assert float(jnp.sum(sol.ys.q[-1])) > 0.0


def test_adsorption_and_desorption_event_signs():
    params = _params(N=4)
    state = initial_state(
        params, AqueousConcentration(0.0, 2000.0, 1000.0), q0=1.0
    )
    ads_event = make_adsorption_event(0.5)
    des_event = make_desorption_event(0.2)

    assert float(ads_event(0.0, state, params)) < 0.0
    state = eqx.tree_at(
        lambda y: y.c_LiCl,
        state,
        state.c_LiCl.at[-1].set(60.0),
    )
    assert float(ads_event(0.0, state, params)) > 0.0
    assert float(des_event(0.0, state, params)) > 0.0
    state = eqx.tree_at(lambda y: y.q, state, jnp.full(params.N, 0.1))
    assert float(des_event(0.0, state, params)) < 0.0


def test_cycle_hands_adsorption_state_to_desorption_and_hits_events():
    params = ColumnParams(
        L=1e-3,
        eps=0.4,
        rho_p=1000.0,
        water_mass_concentration=1000.0,
        u_s=5e-3,
        k2=200.0,
        inlet=AqueousConcentration(100.0, 2000.0, 1000.0),
        isotherm=ALLDHIsotherm(0.01, jnp.log(12.0), 2.0),
        N=4,
    )
    result = simulate_cycle(
        params,
        AqueousConcentration(0.0, 0.0, 0.0),
        adsorption_t_max=5.0,
        desorption_t_max=5.0,
        adsorption_outlet_fraction=0.5,
        desorption_loading_fraction=0.1,
        n_save=101,
    )

    assert 2 < len(result.adsorption_ts) < 101
    assert 2 < len(result.desorption_ts) < 101
    assert float(result.adsorption_duration) < 5.0
    assert float(result.desorption_duration) < 5.0
    assert float(result.adsorption.cumulative_Li_out[0]) == 0.0
    for field in ("c_LiCl", "c_NaCl", "c_MgCl2", "q"):
        assert np.allclose(
            getattr(result.desorption, field)[0],
            getattr(result.adsorption, field)[-1],
        )
    assert float(result.desorption.cumulative_Li_out[0]) == 0.0
    assert float(result.adsorption.c_LiCl[-1, -1]) == pytest.approx(50.0, rel=2e-4)
    assert float(jnp.mean(result.desorption.q[-1])) == pytest.approx(0.001, rel=2e-4)
    assert float(result.desorption.c_LiCl[-1, -1]) > 0.0
    assert 0.0 < float(result.bed_utilization) <= 1.0
    assert float(result.li_recovered) > 0.0
    assert float(result.productivity) == pytest.approx(
        float(result.li_recovered)
        / float(result.adsorption_duration + result.desorption_duration)
    )

    dx = float(params.dx)
    initial_inventory = (
        float(params.eps) * dx * np.sum(result.desorption.c_LiCl[0])
        + float((1.0 - params.eps) * params.rho_p)
        * dx
        * np.sum(result.desorption.q[0])
    )
    final_inventory = (
        float(params.eps) * dx * np.sum(result.desorption.c_LiCl[-1])
        + float((1.0 - params.eps) * params.rho_p)
        * dx
        * np.sum(result.desorption.q[-1])
    )
    assert float(result.li_recovered) == pytest.approx(
        initial_inventory - final_inventory,
        rel=2e-5,
    )
