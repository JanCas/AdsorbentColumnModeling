"""Verification suite for the q_max-normalised non-dim IX column model.

Mirror of Model/IX/NonDim/verification.py with two changes:

  1. Imports from `ix_nondim_qmax`, not `ix_nondim`.
  2. §7.1 expected value is θ/(1+θ), not 1: under n* = q/q_max the Langmuir
     isotherm n*_eq(A*, h) = θ A*/(h + θ A*) yields n*_eq(1, 1) = θ/(1+θ).
     A complementary saturation check confirms n*_eq → 1 as A* → ∞.

Phase saves that needed an explicit grid (7.2 mass conservation, 7.3 grid
convergence, 7.4 event localization) are set via `eqx.tree_at` overriding
`save_ts` directly on the PhaseConfig — the legacy `save_n=` kwarg in the
sibling verification.py was a stale interface and is not reintroduced here.

No matplotlib — spec §6 forbids plots/reports from this module.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ix_nondim_qmax import (
    NonDimParams,
    State,
    _bed_mean_at,
    _default_des_phase,
    _default_load_phase,
    _last_finite_idx,
    adsorption_event,
    default_params,
    desorption_event,
    h_star,
    n_eq_star,
    qois,
    run_phase,
    vector_field,
)

_log = logging.getLogger(__name__)


def _with_save_ts(phase, t_max: float, n_saves: int):
    """Return a copy of `phase` with `save_ts` populated as an n_saves-point
    linspace over [0, t_max]. Workaround for the stale `save_n=` kwarg that
    no longer exists on `_default_load_phase` / `_default_des_phase`."""
    ts = jnp.linspace(0.0, t_max, n_saves, dtype=jnp.float64)
    return eqx.tree_at(lambda p: p.save_ts, phase, ts, is_leaf=lambda x: x is None)


# --- 7.1 isotherm at feed point --------------------------------------------

def verify_isotherm_at_feed() -> tuple[bool, str]:
    """q_max normalisation, bare mass-action form: at feed (A*=1, H*=H_in_ads)
    n*_eq = K*/(H_in_ads + K*) = f < 1, NOT 1.

    Sweep K*; expected value is K*/(H_in_ads+K*). Also confirm the saturation
    limit n*_eq → 1 as A* → ∞ at fixed H*.
    """
    nd = default_params()
    K_stars = jnp.logspace(-11, -7, 9)
    H_feed = jnp.asarray(float(nd.H_in_ads))
    A_feed = jnp.asarray(1.0)

    # Part (a): n*_eq(1, H_in_ads) = K*/(H_in_ads+K*)
    err_feed = 0.0
    f_max = 0.0
    for k in K_stars:
        nd_k = nd.replace(K_star=k)
        val = float(n_eq_star(A_feed, H_feed, nd_k))
        expected = float(k) / (float(nd.H_in_ads) + float(k))
        err_feed = max(err_feed, abs(val - expected))
        f_max = max(f_max, val)
    ok_a = err_feed < 1e-12 and f_max < 1.0

    # Part (b): n*_eq → 1 as A* → ∞ (fixed H*, K*=H_in_ads representative)
    nd_k1 = nd.replace(K_star=float(nd.H_in_ads))
    A_big = jnp.asarray(1e9)
    sat = float(n_eq_star(A_big, H_feed, nd_k1))
    err_sat = abs(sat - 1.0)
    ok_b = err_sat < 1e-8

    ok = ok_a and ok_b
    return ok, (
        f"max |n*_eq(1,H_in_ads) - K*/(H_in_ads+K*)| over K* ∈ [1e-11, 1e-7] "
        f"= {err_feed:.2e}; max f = {f_max:.3f}; |n*_eq(1e9, H) - 1| = {err_sat:.2e}"
    )


# --- 7.2 mass conservation -------------------------------------------------

def verify_mass_conservation() -> tuple[bool, str]:
    """Spec §2: ∂(A*+T*)/∂t* = -∂(A*+T*)/∂x* (advected, no source).

    Run a load phase with dense saves and integrate bed inventory of
    (A*+T*) vs cumulative net flux. Trapezoidal floor sets the tolerance.
    """
    nd = default_params(Lambda=100.0, Da=1.0, K_star=1e-9, omega=1e-10, N=80)
    load = _with_save_ts(
        _default_load_phase(t_max=10.0, max_steps=2_000_000),
        t_max=10.0, n_saves=2000,
    )
    T_in_load = float(nd.H_in_ads) - float(nd.omega) / float(nd.H_in_ads)
    load = eqx.tree_at(lambda p: p.T_in, load,
                       jnp.asarray(T_in_load, dtype=jnp.float64))
    y0 = State(A=jnp.zeros(int(nd.N)),
               T=jnp.full((int(nd.N),), T_in_load, dtype=jnp.float64),
               n=jnp.zeros(int(nd.N)),
               R_outlet=jnp.asarray(0.0))
    sol = run_phase(y0, load, nd)

    ts = np.asarray(sol.ts)
    A_ts = np.asarray(sol.ys.A)
    T_ts = np.asarray(sol.ys.T)
    finite = np.isfinite(ts) & np.isfinite(A_ts[:, -1])
    ts = ts[finite]; A_ts = A_ts[finite]; T_ts = T_ts[finite]
    if ts.size < 3:
        return False, f"insufficient finite saves ({ts.size})"

    N = int(nd.N)
    dx = 1.0 / N
    bed_inv = (A_ts.sum(axis=1) + T_ts.sum(axis=1)) * dx

    flux_in = 1.0 + T_in_load
    flux_out_A = A_ts[:, -1]
    flux_out_T = T_ts[:, -1]
    net_flux = flux_in - (flux_out_A + flux_out_T)
    cum_flux = np.concatenate(
        [[0.0], np.cumsum(0.5 * (net_flux[1:] + net_flux[:-1]) * np.diff(ts))]
    )
    delta_inv = bed_inv - bed_inv[0]
    denom = max(float(np.max(np.abs(cum_flux))), 1e-12)
    err = float(np.max(np.abs(delta_inv - cum_flux)) / denom)
    ok = err < 5e-2
    return ok, (f"A+T mass-balance rel err = {err:.2e}; "
                f"delta_bed_max = {float(np.max(np.abs(delta_inv))):.3e}")


# --- 7.3 grid convergence --------------------------------------------------

def verify_grid_convergence() -> tuple[bool, str]:
    """t*_bt converges monotonically as N grows (upwind diffusion shrinking).
    Relative spread should be < 10% across N ∈ {32, 64, 128}; deltas shrink."""
    t_break = []
    for N in (32, 64, 128):
        nd = default_params(Lambda=100.0, Da=10.0, K_star=1e-9, omega=1e-10, N=N)
        load = _with_save_ts(
            _default_load_phase(t_max=500.0),
            t_max=500.0, n_saves=400,
        )
        T_in_load = float(nd.H_in_ads) - float(nd.omega) / float(nd.H_in_ads)
        load = eqx.tree_at(lambda p: p.T_in, load,
                           jnp.asarray(T_in_load, dtype=jnp.float64))
        y0 = State(A=jnp.zeros(N),
                   T=jnp.full((N,), T_in_load, dtype=jnp.float64),
                   n=jnp.zeros(N),
                   R_outlet=jnp.asarray(0.0))
        sol = run_phase(y0, load, nd)
        idx = int(_last_finite_idx(sol))
        t_break.append(float(np.asarray(sol.ts)[idx]))
    deltas = [abs(t_break[i] - t_break[i - 1]) for i in range(1, len(t_break))]
    spread = (max(t_break) - min(t_break)) / max(abs(np.mean(t_break)), 1e-12)
    deltas_shrink = deltas[-1] <= deltas[0] * 1.1
    ok = deltas_shrink and spread < 1.0e-1
    return ok, (f"t*_bt(N=[32,64,128]) = {[f'{t:.3f}' for t in t_break]}  "
                f"deltas = {[f'{d:.3f}' for d in deltas]}  "
                f"rel spread = {spread:.2e}")


# --- 7.4 event localization ------------------------------------------------

def verify_event_localization() -> tuple[bool, str]:
    """Each event's residual at solve end ≤ O(rtol·|y|)."""
    nd = default_params(Lambda=100.0, Da=5.0, K_star=1e-8, omega=1e-6)
    load = _with_save_ts(
        _default_load_phase(t_max=500.0),
        t_max=500.0, n_saves=400,
    )
    T_in_load = float(nd.H_in_ads) - float(nd.omega) / float(nd.H_in_ads)
    load = eqx.tree_at(lambda p: p.T_in, load,
                       jnp.asarray(T_in_load, dtype=jnp.float64))
    N = int(nd.N)
    y0 = State(A=jnp.zeros(N),
               T=jnp.full((N,), T_in_load, dtype=jnp.float64),
               n=jnp.zeros(N),
               R_outlet=jnp.asarray(0.0))
    sol_load = run_phase(y0, load, nd)
    idx_end = int(_last_finite_idx(sol_load))
    y_end = jax.tree_util.tree_map(lambda a: a[idx_end], sol_load.ys)
    args_load = nd.replace(A_in=load.A_in, T_in=load.T_in)
    # Load event is a signed residual (C_thresh_ads - A[-1]); near-zero at firing.
    res_load = float(adsorption_event(0.0, y_end, args_load))

    des = _with_save_ts(
        _default_des_phase(t_max=500.0),
        t_max=500.0, n_saves=400,
    )
    T_in_des = float(nd.H_in_des) - float(nd.omega) / float(nd.H_in_des)
    des = eqx.tree_at(lambda p: p.T_in, des, jnp.asarray(T_in_des))
    sol_des = run_phase(y_end, des, nd)
    idx_d = int(_last_finite_idx(sol_des))
    y_end_d = jax.tree_util.tree_map(lambda a: a[idx_d], sol_des.ys)
    args_des = nd.replace(A_in=des.A_in, T_in=des.T_in)
    # Desorb event is boolean (A[-1] <= C_thresh_des); calling it returns
    # True/False, not a residual. Use the scalar form (A[-1] - C_thresh_des)
    # directly — at firing this is ≤ 0 and bounded in magnitude by O(rtol).
    res_des = float(y_end_d.A[-1] - args_des.C_thresh_des)

    tol = 5e-2
    ok = abs(res_load) < tol and abs(res_des) < tol
    return ok, (f"|load_event(end)| = {abs(res_load):.2e}, "
                f"|des_event(end)|  = {abs(res_des):.2e}")


# --- 7.5 physics limits ----------------------------------------------------

def verify_physics_limits() -> tuple[bool, str]:
    """Three sanity checks under q_max normalisation:

    (i)   LDF sign: dn/dt has the same sign as (n_eq - n).
    (ii)  Da -> 0: bed loads negligibly (U_b ≈ 0).
    (iii) ω invariance of the isotherm at fixed H*=H_in_ads, independent of ω.
    """
    nd = default_params()
    N = int(nd.N)
    args = nd.replace(A_in=1.0, T_in=0.0)
    A_test = jnp.full((N,), 0.5)
    T_test = jnp.zeros(N)
    H_local = h_star(T_test, args.omega)
    n_eq_local = n_eq_star(A_test, H_local, args)
    R0 = jnp.asarray(0.0)
    state_n_lt = State(A=A_test, T=T_test, n=jnp.zeros(N), R_outlet=R0)
    state_n_eq = State(A=A_test, T=T_test, n=n_eq_local, R_outlet=R0)
    state_n_gt = State(A=A_test, T=T_test, n=n_eq_local + 1.0, R_outlet=R0)
    dn_lt = float(jnp.mean(vector_field(0.0, state_n_lt, args).n))
    dn_eq = float(jnp.mean(vector_field(0.0, state_n_eq, args).n))
    dn_gt = float(jnp.mean(vector_field(0.0, state_n_gt, args).n))
    ok_i = (dn_lt > 0) and (abs(dn_eq) < 1e-9) and (dn_gt < 0)

    nd_da0 = default_params(Da=1e-6)
    metrics = qois(nd_da0)
    U_b_da0 = float(metrics["U_b"])
    ok_ii = U_b_da0 < 1e-3 and np.isfinite(U_b_da0)

    omegas = [1e-13, 1e-9, 1e-6]
    feed_neqs = []
    for w in omegas:
        nd_w = default_params(omega=w)
        feed_neqs.append(float(n_eq_star(jnp.asarray(1.0),
                                         jnp.asarray(float(nd_w.H_in_ads)),
                                         nd_w)))
    feed_rel_spread = (max(feed_neqs) - min(feed_neqs)) / max(np.mean(feed_neqs), 1e-12)
    ok_iii = feed_rel_spread < 1e-12

    ok = ok_i and ok_ii and ok_iii
    detail = (
        f"LDF signs: <,=,> -> ({dn_lt:+.3e}, {dn_eq:+.3e}, {dn_gt:+.3e}); "
        f"Da=1e-6 U_b={U_b_da0:.2e}; "
        f"n*_eq(1,H_in_ads) over ω∈[1e-13,1e-6] spread={feed_rel_spread:.2e}"
    )
    return ok, detail


# --- 7.6 differentiability smoke -------------------------------------------

def verify_differentiable() -> tuple[bool, str]:
    """`jax.grad` of qois['productivity'] w.r.t. log10(Λ) returns finite."""
    nd = default_params()

    def loss(log10_Lambda):
        Lambda = 10.0 ** log10_Lambda
        nd_l = nd.replace(Lambda=Lambda)
        return qois(nd_l)["productivity"]

    grad_fn = jax.grad(loss)
    g = float(grad_fn(jnp.asarray(3.0)))
    ok = np.isfinite(g) and abs(g) > 0.0
    return ok, f"d productivity / d log10(Λ) at Λ=1e3 = {g:.3e}"


# --- runner ---------------------------------------------------------------

_CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("7.1 isotherm at feed",   verify_isotherm_at_feed),
    ("7.2 mass conservation",  verify_mass_conservation),
    ("7.3 grid convergence",   verify_grid_convergence),
    ("7.4 event localization", verify_event_localization),
    ("7.5 physics limits",     verify_physics_limits),
    ("7.6 differentiability",  verify_differentiable),
]


def run_all() -> bool:
    _log.info("running %d non-dim verification checks (q_max basis) ...", len(_CHECKS))
    all_ok = True
    for name, fn in _CHECKS:
        _log.info("starting check: %s", name)
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"raised {type(e).__name__}: {e}"
            _log.exception("check '%s' raised", name)
        dt = time.perf_counter() - t0
        all_ok &= ok
        tag = "PASS" if ok else "FAIL"
        if ok:
            _log.info("[%s] %s  (%.2fs)  %s", tag, name, dt, detail)
        else:
            _log.error("[%s] %s  (%.2fs)  %s", tag, name, dt, detail)
        print(f"[{tag}] {name:<28s}  {detail}")
    print("---")
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    _log.info("non-dim verification (q_max) %s", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from utils.logging_setup import configure_logging
    configure_logging()
    run_all()
