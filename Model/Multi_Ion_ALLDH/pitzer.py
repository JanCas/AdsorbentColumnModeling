"""Pitzer-Kim activity model for the Li+/Na+/Mg2+/Cl- brine — JAX port.

Port of `pitzer-kim functions.m` in this folder. That MATLAB function is the oracle:
`pitzer-parity-test-cases.md` specifies a differential test in which every quantity here
is compared against the MATLAB at the same inputs.

Unit convention:
    m_LiCl, m_NaCl, m_MgCl2   mol / kg water (molality)
    lngamma_*                 dimensionless (natural log of the mean ionic activity coeff.)
    osmo_w                    dimensionless (osmotic coefficient of water)

THE TRANSLATION IS DELIBERATELY LITERAL. The MATLAB is reproduced term for term,
including behaviour that looks wrong:

  * `m_NaCl` doubles as the Na+ molality throughout (the MATLAB never defines `m_Na`),
    so `m_NaCl` appears where `m_Na` is meant.
  * No guards on I_tot = 0 or m_tot = 0. Those inputs produce NaN in the MATLAB and
    produce NaN here, by the same route.

A port that "improves" on the original cannot be tested against it, and every mismatch
becomes unattributable. Do not add guards or refactor the algebra without re-running the
parity suite.

VALIDITY. The parity suite proves this reproduces the MATLAB; it says nothing about whether
the MATLAB is right. Two limits found while validating `water_activity` against independent
data (parity/test_water_activity.py):

  * Single salts to ~6 mol/kg are excellent — saturated NaCl and MgCl2 reproduce the
    Greenspan (1977) humidity standards to 0.0002 and 0.0024 in water activity.
  * Beyond ~6 mol/kg the virial parameters extrapolate badly. Saturated LiCl (19.6 mol/kg)
    gives a_w = 0.027 against a standard of 0.113. Do not trust any output above ~6 mol/kg
    in a single salt.
  * SUSPECT PARAMETER: _PSI_NAMGCL = -0.0517 disagrees with the standard Harvie-Moller-
    Weare value of -0.012 (as shipped in PHREEQC's pitzer.dat, the reference
    parameterisation for the seawater system) by a factor of 4.3. _THETA_NAMG = 0.0970
    likewise vs 0.07.

    In mixed NaCl/MgCl2 brines this drives phi below PHREEQC by up to 0.29. The gap is
    quantitatively explained by psi alone: the analytic term
    (2/sum m) * m_Na * m_Mg * m_Cl * delta_psi predicts it to within 0.012 at every
    composition tested, and substituting the two HMW values cuts the error to <= 0.022,
    the same order as the single-salt parameter differences. Na-Mg mixing is nearly ideal
    in reality; psi = -0.0517 makes it strongly non-ideal.

    Left as-is deliberately — the port must reproduce the MATLAB, and changing this is a
    modelling decision that belongs in the .m source. Impact if you rely on absolute
    values in mixtures: phi off by up to 0.39, a_w by up to 0.06, gamma by 4-13 %.

    Not checked, for want of a reference: _THETA_LIMG = 0.2198 and _PSI_LIMGCL. PHREEQC's
    pitzer.dat carries no Li mixing parameters at all, so the Li terms are unverified.

This module is pure JAX: no I/O, no matplotlib, and no dependency on anything in parity/.
The differential test suite lives in parity/ — see parity/test_parity.py.

Imported flat, matching Model/IX/dim — this folder must be on sys.path:

    sys.path.insert(0, ".../Model/Multi_Ion_ALLDH")
    from pitzer import pitzer_mix
"""

import logging

import jax
import numpy as np

_log = logging.getLogger(__name__)
# NOTE: do NOT put _log.* calls inside @eqx.filter_jit bodies (pitzer_terms); they only
# fire at trace time, not runtime. Log at the orchestration layer.

# float64 is mandatory here. The parity spec holds the closed-form algebra to rtol 1e-13,
# which float32 cannot express at all, and J0 is evaluated as `0.25*X - 1 + I/X` — a
# cancellation of two O(1) terms that yields O(1e-6) at low ionic strength. Enable float64
# at import time so any module importing pitzer gets it.
jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# --- physical constants (MATLAB lines 3-9) ---

_N_A = 6.0221408e23
_KB = 1.380649e-23
_EPS0 = 8.854e-12
_E_CHARGE = 1.602e-19
_EPS_R = 78.0
_RHO_W = 998.0  # kg/m^3
_TEMP = 298.15  # K

# Molar mass of water, kg/mol. Not used by the MATLAB — it enters only in the
# osmotic-coefficient -> water-activity conversion below.
_M_W = 0.01801528

# --- ionic charges (MATLAB line 11) ---

_Z_LI = 1.0
_Z_NA = 1.0
_Z_MG = 2.0
_Z_CL = -1.0

# --- Debye-Huckel parameter (MATLAB lines 24-25) ---
# Evaluated in float64 numpy at import so it is a compile-time constant, bit-identical to
# the MATLAB expression.
_A_PHI = (1.0 / 3.0) * (2.0 * np.pi * _N_A * _RHO_W) ** 0.5 * (
    _E_CHARGE**2 / (4.0 * np.pi * _EPS0 * _EPS_R * _KB * _TEMP)
) ** 1.5
_B = 1.2

# --- Pitzer virial coefficients (MATLAB lines 28-30) ---

_BETA0_LICL, _BETA1_LICL, _CPHI_LICL = 0.14667, 0.33703, 0.00393
_BETA0_NACL, _BETA1_NACL, _CPHI_NACL = 0.07722, 0.25183, 0.00106
_BETA0_MGCL2, _BETA1_MGCL2, _CPHI_MGCL2 = 0.35372, 1.70054, 0.00524

# --- mixing parameters (MATLAB lines 33-34) ---

_THETA_LINA, _THETA_LIMG, _THETA_NAMG = 0.0120, 0.2198, 0.0970
_PSI_LINACL, _PSI_LIMGCL, _PSI_NAMGCL = -0.0022, -0.01930, -0.0517


# --- quadrature for the unsymmetric-mixing J integrals -------------------------------
#
# The MATLAB uses adaptive `integral(..., 0, Inf)`. JAX has no jittable adaptive
# quadrature, so this is fixed-node Gauss-Legendre. Two choices matter:
#
# 1. The substitution Y = exp(u). The integrand's transition — where (X/Y)exp(-Y) crosses
#    unity — sits at Y ~ X and therefore sweeps four decades as X ranges over the values
#    the model produces. In u the feature has width O(1) at every X, so one fixed node set
#    resolves all of them. Composite Gauss-Legendre directly in Y is off by 5e-2 at small X.
#
# 2. The range u in [-45, 5] and the panel/order split. Below the transition the integrand
#    is Y^3 = exp(3u), which is exp(-135) at the lower limit; above it decays like
#    X*Y^2*exp(-Y), which is exp(-148) at the upper. Both ends are far past float64 reach.
#
# Benchmarked against mpmath at 40 digits: relative error is <= 3e-12 for X >= 0.5 and
# <= 6e-10 for X >= 0.05. What is left at small X is cancellation in `0.25*X - 1 + I/X`,
# not quadrature error — MATLAB and Octave disagree with each other by 3e-6 there. Model
# X values are X = 6*|z_i z_j|*A_phi*sqrt(I), so X >= 0.74 whenever I >= 0.1.
#
# If a quadrature-dependent tier of the parity suite ever fails, raise _GL_ORDER or
# _GL_PANELS. Do not loosen the tolerance.

_GL_ORDER = 32
_GL_PANELS = 10
_GL_U_MIN = -45.0
_GL_U_MAX = 5.0


def _build_gl_nodes() -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes/weights on [_GL_U_MIN, _GL_U_MAX] in float64."""
    x, w = np.polynomial.legendre.leggauss(_GL_ORDER)
    edges = np.linspace(_GL_U_MIN, _GL_U_MAX, _GL_PANELS + 1)
    half = (edges[1:] - edges[:-1]) / 2.0
    mid = (edges[1:] + edges[:-1]) / 2.0
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


_GL_U_NP, _GL_W_NP = _build_gl_nodes()
_GL_U = jnp.asarray(_GL_U_NP)
_GL_W = jnp.asarray(_GL_W_NP)
_GL_Y = jnp.exp(_GL_U)
_GL_Y3 = _GL_Y**3  # Y^2 from the integrand, times Y from dY = Y du


def j_integrals(X):
    """Pitzer unsymmetric-mixing integrals J0(X) and J1(X). MATLAB lines 38-39.

    J0(X) = X/4 - 1 + (1/X) * int_0^inf Y^2 (1 - exp(-t)) dY
    J1(X) = X/4     - (1/X) * int_0^inf Y^2 (1 - (1+t) exp(-t)) dY,   t = (X/Y) exp(-Y)

    X = 0 returns NaN for both, as the MATLAB does: there the integral vanishes and the
    prefactor 1/X is Inf, so MATLAB evaluates Inf*0 while this evaluates 0/0. Same NaN.

    Broadcasts over X, so jax.vmap works.
    """
    X = jnp.asarray(X)
    t = (X[..., None] / _GL_Y) * jnp.exp(-_GL_Y)
    exp_t = jnp.exp(-t)

    int0 = jnp.sum(_GL_W * _GL_Y3 * (1.0 - exp_t), axis=-1)
    int1 = jnp.sum(_GL_W * _GL_Y3 * (1.0 - (1.0 + t) * exp_t), axis=-1)

    return 0.25 * X - 1.0 + int0 / X, 0.25 * X - int1 / X


class PitzerTerms(eqx.Module):
    """Every quantity the parity spec compares, in its dependency order.

    Ordering is load-bearing: the differential test asserts field by field down this list
    and stops at the first mismatch, so the first failure is the cause and everything
    after it is downstream contamination.
    """

    # composition and Debye-Huckel
    I_tot: jax.Array
    Z_sum: jax.Array
    m_Cl: jax.Array
    m_tot: jax.Array
    A_phi: jax.Array
    # unsymmetric mixing
    thetaE_LiMg: jax.Array
    dthetaE_LiMg: jax.Array
    Phi_LiMg: jax.Array
    Phiphi_LiMg: jax.Array
    # binary interaction, activity form
    g_chi: jax.Array
    dg_chi: jax.Array
    B_LiCl: jax.Array
    dB_LiCl: jax.Array
    C_LiCl: jax.Array
    BC_LiCl: jax.Array
    # osmotic assembly
    phi_DH: jax.Array
    bin_phi: jax.Array
    tert_phi: jax.Array
    F_mix: jax.Array
    osmo_w: jax.Array
    # outputs
    lngamma_LiCl: jax.Array
    lngamma_NaCl: jax.Array
    lngamma_MgCl2: jax.Array


@eqx.filter_jit
def pitzer_terms(m_LiCl, m_NaCl, m_MgCl2) -> PitzerTerms:
    """Full Pitzer-Kim evaluation, exposing the intermediates the parity spec compares.

    Line-for-line transcription of `pitzer-kim functions.m`. See the module docstring on
    why the quirks are preserved rather than fixed.

    Parameters
    ----------
    m_LiCl, m_NaCl, m_MgCl2
        Salt molalities, mol/kg water. Scalars; vmap for batches.
    """
    m_LiCl = jnp.asarray(m_LiCl, dtype=jnp.float64)
    m_NaCl = jnp.asarray(m_NaCl, dtype=jnp.float64)
    m_MgCl2 = jnp.asarray(m_MgCl2, dtype=jnp.float64)

    # --- derived concentrations (MATLAB lines 14-21) ---
    # NOTE: m_NaCl is the Na+ molality as well as the salt molality, per the MATLAB.
    m_Li = m_LiCl
    m_Mg = m_MgCl2
    m_Cl = m_LiCl + m_NaCl + 2.0 * m_MgCl2
    m_tot = m_Li + m_NaCl + m_Mg + m_Cl

    I_tot = m_LiCl + m_NaCl + 3.0 * m_MgCl2  # (1/2) sum m_i z_i^2
    Z_sum = m_Li + m_NaCl + 2.0 * m_Mg + m_Cl  # sum m_i |z_i|

    sqrt_I = jnp.sqrt(I_tot)

    # --- unsymmetric mixing (MATLAB lines 41-49) ---
    # Li-Na have equal charge, so thetaE_LiNa = 0 identically and the J terms cancel.
    X_LiLi = 6.0 * _Z_LI**2 * _A_PHI * sqrt_I  # = X_NaNa since z_Li = z_Na
    X_MgMg = 6.0 * _Z_MG**2 * _A_PHI * sqrt_I
    X_LiMg = 6.0 * _Z_LI * _Z_MG * _A_PHI * sqrt_I  # = X_NaMg since z_Li = z_Na

    J0_all, J1_all = j_integrals(jnp.stack([X_LiMg, X_LiLi, X_MgMg]))
    J0_LiMg, J0_LiLi, J0_MgMg = J0_all[0], J0_all[1], J0_all[2]
    J1_LiMg, J1_LiLi, J1_MgMg = J1_all[0], J1_all[1], J1_all[2]

    J0_comb = J0_LiMg - 0.5 * J0_LiLi - 0.5 * J0_MgMg
    J1_comb = J1_LiMg - 0.5 * J1_LiLi - 0.5 * J1_MgMg

    dthetaE_LiNa = jnp.zeros_like(I_tot)

    thetaE_LiMg = (_Z_LI * _Z_MG / (4.0 * I_tot)) * J0_comb
    thetaE_NaMg = (_Z_NA * _Z_MG / (4.0 * I_tot)) * J0_comb
    dthetaE_LiMg = (_Z_LI * _Z_MG / (8.0 * I_tot**2)) * J1_comb - thetaE_LiMg / I_tot
    dthetaE_NaMg = (_Z_NA * _Z_MG / (8.0 * I_tot**2)) * J1_comb - thetaE_NaMg / I_tot

    # --- Phi^phi (osmotic) and Phi (activity) (MATLAB lines 53-59) ---
    # thetaE_LiNa = 0, so it is dropped from the LiNa lines exactly as it cancels there.
    Phiphi_LiNa = _THETA_LINA + 0.0 + I_tot * dthetaE_LiNa
    Phiphi_LiMg = _THETA_LIMG + thetaE_LiMg + I_tot * dthetaE_LiMg
    Phiphi_NaMg = _THETA_NAMG + thetaE_NaMg + I_tot * dthetaE_NaMg

    Phi_LiNa = _THETA_LINA + 0.0
    Phi_LiMg = _THETA_LIMG + thetaE_LiMg
    Phi_NaMg = _THETA_NAMG + thetaE_NaMg

    # --- osmotic coefficient (MATLAB lines 64-75) ---
    exp_2sqrtI = jnp.exp(-2.0 * sqrt_I)
    Bphi_LiCl = _BETA0_LICL + _BETA1_LICL * exp_2sqrtI
    Bphi_NaCl = _BETA0_NACL + _BETA1_NACL * exp_2sqrtI
    Bphi_MgCl2 = _BETA0_MGCL2 + _BETA1_MGCL2 * exp_2sqrtI

    phi_DH = -_A_PHI * I_tot**1.5 / (1.0 + _B * sqrt_I)
    bin_phi = (
        m_Li * m_Cl * (Bphi_LiCl + Z_sum * _CPHI_LICL / (2.0 * np.sqrt(abs(_Z_LI * _Z_CL))))
        + m_NaCl * m_Cl * (Bphi_NaCl + Z_sum * _CPHI_NACL / (2.0 * np.sqrt(abs(_Z_NA * _Z_CL))))
        + m_Mg * m_Cl * (Bphi_MgCl2 + Z_sum * _CPHI_MGCL2 / (2.0 * np.sqrt(abs(_Z_MG * _Z_CL))))
    )
    tert_phi = (
        m_Li * m_NaCl * (Phiphi_LiNa + m_Cl * _PSI_LINACL)
        + m_Li * m_Mg * (Phiphi_LiMg + m_Cl * _PSI_LIMGCL)
        + m_NaCl * m_Mg * (Phiphi_NaMg + m_Cl * _PSI_NAMGCL)
    )
    osmo_w = 1.0 + (2.0 / m_tot) * (phi_DH + bin_phi + tert_phi)

    # --- B in activity form: g(chi), chi = 2 sqrt(I) (MATLAB lines 82-91) ---
    chi = 2.0 * sqrt_I
    exp_chi = jnp.exp(-chi)
    g_chi = 2.0 * (1.0 - (1.0 + chi) * exp_chi) / chi**2
    # satisfies dB/dI = beta1 * dg_chi / I
    dg_chi = -2.0 * (1.0 - (1.0 + chi + 0.5 * chi**2) * exp_chi) / chi**2

    B_LiCl = _BETA0_LICL + _BETA1_LICL * g_chi
    B_NaCl = _BETA0_NACL + _BETA1_NACL * g_chi
    B_MgCl2 = _BETA0_MGCL2 + _BETA1_MGCL2 * g_chi
    dB_LiCl = _BETA1_LICL * dg_chi / I_tot
    dB_NaCl = _BETA1_NACL * dg_chi / I_tot
    dB_MgCl2 = _BETA1_MGCL2 * dg_chi / I_tot

    # --- C in activity form: C = Cphi / (2 sqrt|z+ z-|) (MATLAB lines 94-96) ---
    C_LiCl = _CPHI_LICL / (2.0 * np.sqrt(abs(_Z_LI * _Z_CL)))
    C_NaCl = _CPHI_NACL / (2.0 * np.sqrt(abs(_Z_NA * _Z_CL)))
    C_MgCl2 = _CPHI_MGCL2 / (2.0 * np.sqrt(abs(_Z_MG * _Z_CL)))

    # --- F_mix, shared by all salts (MATLAB lines 99-101) ---
    F_mix = (
        -_A_PHI * (sqrt_I / (1.0 + _B * sqrt_I) + (2.0 / _B) * jnp.log(1.0 + _B * sqrt_I))
        + m_Li * m_Cl * dB_LiCl
        + m_NaCl * m_Cl * dB_NaCl
        + m_Mg * m_Cl * dB_MgCl2
        + m_Li * m_NaCl * dthetaE_LiNa
        + m_Li * m_Mg * dthetaE_LiMg
        + m_NaCl * m_Mg * dthetaE_NaMg
    )

    # --- B + (Z/2) C (MATLAB lines 104-106) ---
    BC_LiCl = B_LiCl + (Z_sum / 2.0) * C_LiCl
    BC_NaCl = B_NaCl + (Z_sum / 2.0) * C_NaCl
    BC_MgCl2 = B_MgCl2 + (Z_sum / 2.0) * C_MgCl2

    # --- ln gamma+- LiCl: nu+ = nu- = 1 (MATLAB lines 109-114) ---
    lngamma_LiCl = (
        abs(_Z_LI * _Z_CL) * F_mix
        + (m_Li + m_Cl) * BC_LiCl
        + m_NaCl * (Phi_LiNa + BC_NaCl)
        + m_Mg * (Phi_LiMg + BC_MgCl2)
        + 0.5 * (m_Li + m_Cl) * (m_NaCl * _PSI_LINACL + m_Mg * _PSI_LIMGCL)
        + 0.5 * m_NaCl * m_Mg * _PSI_NAMGCL
    )

    # --- ln gamma+- NaCl: nu+ = nu- = 1 (MATLAB lines 117-122) ---
    lngamma_NaCl = (
        abs(_Z_NA * _Z_CL) * F_mix
        + (m_NaCl + m_Cl) * BC_NaCl
        + m_Li * (Phi_LiNa + BC_LiCl)
        + m_Mg * (Phi_NaMg + BC_MgCl2)
        + 0.5 * (m_NaCl + m_Cl) * (m_Li * _PSI_LINACL + m_Mg * _PSI_NAMGCL)
        + 0.5 * m_Li * m_Mg * _PSI_LIMGCL
    )

    # --- ln gamma+- MgCl2: nu+ = 1, nu- = 2 (MATLAB lines 125-132) ---
    lngamma_MgCl2 = (
        abs(_Z_MG * _Z_CL) * F_mix
        + (2.0 / 3.0) * m_Cl * BC_MgCl2
        + (4.0 / 3.0) * m_Mg * BC_MgCl2
        + (4.0 / 3.0) * m_Li * (BC_LiCl + 0.5 * Phi_LiMg)
        + (4.0 / 3.0) * m_NaCl * (BC_NaCl + 0.5 * Phi_NaMg)
        + (1.0 / 3.0) * m_Li * (m_Cl + 2.0 * m_Mg) * _PSI_LIMGCL
        + (1.0 / 3.0) * m_NaCl * (m_Cl + 2.0 * m_Mg) * _PSI_NAMGCL
        + (2.0 / 3.0) * m_Li * m_NaCl * _PSI_LINACL
    )

    return PitzerTerms(
        I_tot=I_tot,
        Z_sum=Z_sum,
        m_Cl=m_Cl,
        m_tot=m_tot,
        A_phi=jnp.asarray(_A_PHI, dtype=jnp.float64),
        thetaE_LiMg=thetaE_LiMg,
        dthetaE_LiMg=dthetaE_LiMg,
        Phi_LiMg=Phi_LiMg,
        Phiphi_LiMg=Phiphi_LiMg,
        g_chi=g_chi,
        dg_chi=dg_chi,
        B_LiCl=B_LiCl,
        dB_LiCl=dB_LiCl,
        C_LiCl=jnp.asarray(C_LiCl, dtype=jnp.float64),
        BC_LiCl=BC_LiCl,
        phi_DH=phi_DH,
        bin_phi=bin_phi,
        tert_phi=tert_phi,
        F_mix=F_mix,
        osmo_w=osmo_w,
        lngamma_LiCl=lngamma_LiCl,
        lngamma_NaCl=lngamma_NaCl,
        lngamma_MgCl2=lngamma_MgCl2,
    )


def pitzer_mix(m_LiCl, m_NaCl, m_MgCl2):
    """Mean ionic activity coefficients and the water osmotic coefficient.

    Signature matches the MATLAB `pitzer_mix` exactly.

    Parameters
    ----------
    m_LiCl, m_NaCl, m_MgCl2
        Salt molalities, mol/kg water.

    Returns
    -------
    lngamma_LiCl, lngamma_NaCl, lngamma_MgCl2 : jax.Array
        ln of the mean ionic activity coefficient of each salt.
    osmo_w : jax.Array
        Osmotic coefficient of water.
    """
    t = pitzer_terms(m_LiCl, m_NaCl, m_MgCl2)
    return t.lngamma_LiCl, t.lngamma_NaCl, t.lngamma_MgCl2, t.osmo_w


def water_activity(osmo_w, m_tot):
    """Water activity from the osmotic coefficient.

        ln a_w = -M_w * phi * sum_i m_i

    which is the defining relation for phi, rearranged — exact, not a correlation.
    M_w is the molar mass of water in kg/mol and the sum runs over every solute
    *species*, i.e. the dissociated ions, not the salts.

    Note the second argument. The conversion is not a function of phi alone: two
    solutions with the same osmotic coefficient and different total molality have
    different water activities. Pass `pitzer_terms(...).m_tot`, which is exactly this
    sum (m_Li + m_Na + m_Mg + m_Cl); `pitzer_mix` does not return it, so use
    `water_activity_of` below if you only have the molalities.

    This has no counterpart in `pitzer-kim functions.m` and so is not covered by the
    MATLAB parity suite. It is validated against literature NaCl and MgCl2 data in
    parity/test_water_activity.py instead.

    Parameters
    ----------
    osmo_w
        Osmotic coefficient of water, dimensionless (the 4th output of pitzer_mix).
    m_tot
        Total molality of all dissolved ions, mol/kg water.

    Returns
    -------
    a_w : jax.Array
        Water activity, dimensionless. 1 for pure water, decreasing with salinity;
        equals the equilibrium relative humidity over the solution.
    """
    return jnp.exp(-_M_W * jnp.asarray(osmo_w) * jnp.asarray(m_tot))


def water_activity_of(m_LiCl, m_NaCl, m_MgCl2):
    """Water activity straight from the salt molalities.

    Convenience wrapper: runs the model and supplies `m_tot` to `water_activity` for you.

    Parameters
    ----------
    m_LiCl, m_NaCl, m_MgCl2
        Salt molalities, mol/kg water.

    Returns
    -------
    a_w : jax.Array
        Water activity, dimensionless.
    """
    t = pitzer_terms(m_LiCl, m_NaCl, m_MgCl2)
    return water_activity(t.osmo_w, t.m_tot)


# Field order of PitzerTerms, which is also the assertion order of the parity suite and
# the column order of the oracle CSV. parity/cases.py carries a JAX-free copy of this list
# (the oracle driver needs it but must not import JAX); parity/test_parity.py asserts the
# two have not drifted apart.
TERM_NAMES: tuple[str, ...] = tuple(PitzerTerms.__dataclass_fields__)
