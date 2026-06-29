import logging

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

_log = logging.getLogger(__name__)
# `run_cycle` below is @eqx.filter_jit; do NOT put _log calls inside its body.
# Callers (multi_optim.py, Sensitivity/sobol_driver.py) handle runtime logging.

N_SPATIAL = 50  # number of spatial grid points (matches IX non-dim N=50
                # default, so cross-model trapezoid/mean approximations
                # carry the same discretization error budget)


# ---------------------------------------------------------------------------
# Isotherm
# ---------------------------------------------------------------------------
# q_max-normalised variant of Model/AlLDH/diffrax_non_dim.py.
#
# Loading normalisation:   n* = q / q_max   (was q / q_eq(c_feed) in the
# feed-equilibrium-normalised sibling). The Langmuir isotherm
#     q_eq = q_max · K c / (1 + K c)
# under C* = c/c_feed and θ = K c_feed therefore reads
#     n*_eq(C*) = θ C* / (1 + θ C*),     n*_eq ∈ [0, 1].
# At feed (C*=1) n*_eq = θ/(1+θ), not 1 — the loading axis now measures
# fraction of total Langmuir capacity occupied.
def langmuir_isotherm_non_dim(C_star, theta):
    return theta * C_star / (1 + theta * C_star)


# ---------------------------------------------------------------------------
# Column state
# ---------------------------------------------------------------------------
class ColumnState(eqx.Module):
    C_star: Float[Array, "n"]
    n_star: Float[Array, "n"]
    R_outlet: Float[Array, ""]        # ∫ C*(ζ=1) dτ


# ---------------------------------------------------------------------------
# Dimensionless numbers (physical parameters only)
# ---------------------------------------------------------------------------
# Time normalisation: τ = t · u_int / L  (INTERSTITIAL residence time, same
# convention as Model/IX/NonDim/ix_nondim_qmax.py). Under this convention the
# bed-porosity ε is absorbed into Λ and Da and does not appear explicitly
# in the PDE:
#   Λ  = (1−ε)/ε · ρ_p · q_max / c_feed   (loading scale is q_max here, NOT
#                                          q_eq(c_feed); equals Λ_feed · (1+θ)/θ
#                                          if you are converting from the
#                                          feed-equilibrium-normalised model)
#   Da = k · L / u_int                    (interstitial Damköhler, unchanged)
#   θ  = K · c_feed                       (Langmuir feed favorability, unchanged)
class NonDimNumbers(eqx.Module):
    Da: Float[Array, ""]       # interstitial Damkoehler  k L / u_int
    Lambda: Float[Array, ""]   # sorbent-to-fluid capacity ratio (q_max basis)
    theta: Float[Array, ""]    # isotherm steepness

    def __post_init__(self):
        object.__setattr__(self, "Da", jnp.asarray(self.Da))
        object.__setattr__(self, "Lambda", jnp.asarray(self.Lambda))
        object.__setattr__(self, "theta", jnp.asarray(self.theta))


# ---------------------------------------------------------------------------
# ODE args passed to diffrax (includes inlet BC and grid spacing)
# ---------------------------------------------------------------------------
class ColumnArgs(eqx.Module):
    Da: Float[Array, ""]
    Lambda: Float[Array, ""]
    theta: Float[Array, ""]
    c_inlet: Float[Array, ""]   # 1.0 for adsorption, 0.0 for desorption
    dzeta: Float[Array, ""]     # spatial grid spacing

    @classmethod
    def from_non_dim(cls, nd: NonDimNumbers, c_inlet: float):
        return cls(
            Da=nd.Da, Lambda=nd.Lambda, theta=nd.theta,
            c_inlet=jnp.asarray(c_inlet),
            dzeta=jnp.asarray(1.0 / (N_SPATIAL - 1)),
        )


# ---------------------------------------------------------------------------
# ODE right-hand side  (standard LDF kinetics)
# ---------------------------------------------------------------------------
# Under interstitial-time normalisation the PDE is
#   dC/dτ + Λ · dn/dτ = -dC/dζ            (no 1/ε prefactor anywhere)
# mirroring Model/IX/NonDim/ix_nondim_qmax.py:vector_field exactly. Form is
# identical to the feed-normalised sibling; only the isotherm numerator and
# the q_max content of Λ differ.
def column_ode(t, state: ColumnState, args: ColumnArgs):
    C = state.C_star
    n = state.n_star

    n_eq = langmuir_isotherm_non_dim(C, args.theta)

    dn_dt = args.Da * (n_eq - n)

    C_prev = jnp.roll(C, 1).at[0].set(args.c_inlet)
    advection = (C - C_prev) / args.dzeta
    dC_dt = -advection - args.Lambda * dn_dt

    return ColumnState(
        C_star=dC_dt,
        n_star=dn_dt,
        R_outlet=jnp.asarray(C[-1]),           # dR_outlet/dτ = C*(ζ=1, τ)
    )


# ---------------------------------------------------------------------------
# Event functions (closure pattern — threshold captured at creation time)
# ---------------------------------------------------------------------------
def make_ads_event(c_thresh):
    """Stop adsorption when outlet concentration reaches *c_thresh*."""
    def _event(t, y: ColumnState, args, **kwargs):
        return y.C_star[-1] >= c_thresh
    return _event


def make_des_event(c_thresh):
    """Stop desorption when outlet concentration drops below *c_thresh*.

    Switched from mean-loading to outlet-concentration semantics for
    consistency with the IX non-dim model (Model/IX/NonDim/ix_nondim_qmax.py)
    and to fix the "barely loaded → desorb runs to t_max" failure mode
    observed in the IX sweep at Results/Sensitivity/IX/1.5_alldhlo (see
    the event_reformulation_report.html there for the diagnosis).
    """
    def _event(t, y: ColumnState, args, **kwargs):
        return y.C_star[-1] <= c_thresh
    return _event


# ---------------------------------------------------------------------------
# Shared solver settings
# ---------------------------------------------------------------------------
_SOLVER = diffrax.Tsit5()
# Kvaerno5/Kvaerno3 were tried as a fix for the stiffness cliff at
# log10(Λ·Da·θ) ≳ 4 but were unacceptably slow (5-10× Tsit5). Pragmatic fix:
# trim the Sobol box via --log10-da-range to keep the stiff corner out.
# See Model/IX/NonDim/ix_nondim_qmax.py for the same change rationale.
_RTOL = 1e-3
_ATOL = 1e-6
_DT0 = 1e-4
_MAX_STEPS = 5_000_000


_CONTROLLER = diffrax.PIDController(
    pcoeff=0.3, icoeff=0.4, rtol=_RTOL, atol=_ATOL
)


def ads_state(c_threshold_des):
    return ColumnState(
        C_star=jnp.zeros(N_SPATIAL),
        n_star=jnp.full(N_SPATIAL, c_threshold_des),
        R_outlet=jnp.array(0.0)
    )



# ---------------------------------------------------------------------------
# Main cycle solver
# ---------------------------------------------------------------------------
@eqx.filter_jit
def run_cycle(
    non_dim: NonDimNumbers,
    c_thresh_ads: Float[Array, ""],
    c_thresh_des: Float[Array, ""],
):
    """
    Run a full adsorption-desorption cycle and return performance metrics.

    Parameters
    ----------
    non_dim : NonDimNumbers
        Physical dimensionless groups (Da, Lambda, theta). Bed porosity ε
        is absorbed into Λ via the interstitial-time normalisation; here
        Λ uses q_max (not q_eq(c_feed)) as the loading scale. See the
        NonDimNumbers docstring for the conversion.
    c_thresh_ads : float
        Outlet concentration threshold to end adsorption  (0 < . < 1).
    c_thresh_des : float
        Outlet concentration threshold to end desorption   (0 < . < 1).
        Desorption stops when the elution peak has passed and the outlet
        drops below this value.
    Returns
    -------
    tau_ads      - dimensionless adsorption time at cutoff (interstitial τ)
    tau_des      - dimensionless desorption time at cutoff (interstitial τ)
    U_b          - bed utilisation at end of adsorption (fraction of q_max
                   occupied; max attainable at feed equilibrium is θ/(1+θ))
    R_outlet_des - dimensionless Li collected at outlet during desorption ∫C*(ζ=1)dτ
    productivity - R_outlet_des / (tau_ads + tau_des)
    R_release    - Li released from the adsorbed phase: Λ · (U_b − ⟨n⟩_des_end)
    R_wash       - solution-phase wash mass: ⟨C⟩_load_end − ⟨C⟩_des_end
                   By desorb-phase mass balance under interstitial-time
                   normalisation, R_release + R_wash ≈ R_outlet_des (modulo
                   solver tolerance + the max(0,·) clamps). Identical form
                   to Model/IX/NonDim/ix_nondim_qmax.py:qois.
    """
    zeta = jnp.linspace(0.0, 1.0, N_SPATIAL)
    # ---- Phase 1: Adsorption ------------------------------------------------
    ads_args = ColumnArgs.from_non_dim(non_dim, c_inlet=1.0)
    ads_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(column_ode),
        _SOLVER,
        t0=0.0,
        t1=jnp.inf,
        dt0=_DT0,
        y0=ads_state(c_thresh_des),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=_CONTROLLER,
        event=diffrax.Event(make_ads_event(c_thresh_ads)),
        args=ads_args,
        max_steps=_MAX_STEPS,
    )
    ads_final = ads_sol.ys
    tau_ads = ads_sol.ts[0]


    # Bed utilisation
    U_b = jnp.trapezoid(ads_final.n_star, zeta)[0]
    # ---- Phase 2: Desorption ------------------------------------------------
    des_args = ColumnArgs.from_non_dim(non_dim, c_inlet=0.0)
    des_y0 = ColumnState(
        C_star=ads_final.C_star[-1],
        n_star=ads_final.n_star[-1],
        R_outlet=jnp.array(0.0),       # reset accumulator for desorption
    )

    des_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(column_ode),
        _SOLVER,
        t0=0.0,
        t1=jnp.inf,
        dt0=_DT0,
        y0=des_y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=_CONTROLLER,
        event=diffrax.Event(make_des_event(c_thresh_des)),
        args=des_args,
        max_steps=_MAX_STEPS,
    )
    des_final = des_sol.ys
    tau_des = des_sol.ts[0]

    # Li collected at the outlet during desorption: ∫₀^τ_des C*(ζ=1, τ) dτ
    R_outlet_des = des_final.R_outlet[0]

    R_outlet_des = jnp.maximum(R_outlet_des, 1e-12)

    productivity = R_outlet_des / (tau_ads + tau_des)

    # Wash / release decomposition of R_outlet_des. Mass balance over the
    # desorb phase (c_inlet = 0) under interstitial-time normalisation:
    #   R_outlet_des = (⟨C⟩_load_end − ⟨C⟩_des_end) + Λ · (U_b − ⟨n⟩_des_end)
    # Identical form to Model/IX/NonDim/ix_nondim_qmax.py:qois because both
    # models now use the same τ convention (ε absorbed into Λ) and the same
    # q_max loading basis.
    n_des_end = jnp.trapezoid(des_final.n_star, zeta)[0]
    C_load_end = jnp.trapezoid(ads_final.C_star, zeta)[0]
    C_des_end  = jnp.trapezoid(des_final.C_star, zeta)[0]
    R_release = non_dim.Lambda * jnp.maximum(U_b - n_des_end, 0.0)
    R_wash    = jnp.maximum(C_load_end - C_des_end, 0.0)

    return tau_ads, tau_des, U_b, R_outlet_des, productivity, R_release, R_wash
