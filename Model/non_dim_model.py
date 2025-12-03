# column_temkin_star.py

import numpy as np


def n_eq_star_temkin(C_star: np.ndarray | float, Lambda_star: float) -> np.ndarray:
    """
    Dimensionless Temkin isotherm:

        n_eq*(C*) = 1 + ln(C*) / Lambda*

    Parameters
    ----------
    C_star : float or ndarray
        Dimensionless liquid concentration C* = C / C0.
    Lambda_star : float
        Dimensionless Temkin parameter, Lambda* = ln(A_si * C0).

    Returns
    -------
    n_eq_star : ndarray
        Dimensionless equilibrium loading n_eq* = n_eq / n0.
    """
    C_arr = np.asarray(C_star, dtype=float)
    C_safe = np.clip(C_arr, 1e-30, None)  # avoid log(0)
    return 1.0 + np.log(C_safe) / Lambda_star

def n_eq_star_langmuir(C_star: np.ndarray | float, Lambda_star: float) -> np.ndarray:
    C_arr = np.asarray(C_star, dtype=float)
    return (1.0 + Lambda_star)*C_star / (1 + Lambda_star * C_arr)


def simulate_column_temkin_star(
    phi_star: float,
    Da2_star: float,
    Lambda_star: float,
    eps: float = 0.4,
    Nx: int = 100,
    tau_star_max: float = 15.0,
    C_star_thresh: float = 0.05,
    cfl: float = 0.5,
    store_history: bool = False,
):
    """
    1D dimensionless column model (no dispersion) with Temkin isotherm,
    using explicit Euler in time.

        C*_tau + phi* n*_tau + (1/eps) C*_x = 0
        n*_tau = Da2* (n_eq*(C*) - n*)^2
        n_eq*(C*) = 1 + ln(C*) / Lambda*

    Discretization:
        - x* in [0, 1], Nx points, uniform grid.
        - first-order upwind for advection.
        - explicit Euler in tau*.
        - inlet BC: C*(0, tau*) = 1.
        - IC: C*(x*, 0) = 0, n*(x*, 0) = 0.

    Parameters
    ----------
    phi_star : float
        Dimensionless capacity ratio φ*.
    Da2_star : float
        Dimensionless pseudo–second-order Damköhler number Da2*.
    Lambda_star : float
        Dimensionless Temkin parameter Λ* = ln(A_si * C0).
    eps : float
        Porosity ε.
    Nx : int
        Number of spatial grid points.
    tau_star_max : float
        Max dimensionless time τ*.
    C_star_thresh : float
        Breakthrough threshold at outlet, in terms of C*.
    cfl : float
        CFL safety factor; Δτ* <= cfl * eps * Δx*.
    store_history : bool
        If True, also returns histories of C* and n*.

    Returns
    -------
    tau_star_break : float
        Dimensionless breakthrough time τ* when C*_out >= C_star_thresh.
        If not reached, returns tau_star_max.
    C_star_hist : ndarray or None
        Shape (Nt_eff, Nx) if store_history=True, else None.
    n_star_hist : ndarray or None
        Same as above.
    """
    # Spatial grid and step
    x_star = np.linspace(0.0, 1.0, Nx)
    dx_star = x_star[1] - x_star[0]

    # Time step from CFL-like condition (dimensionless velocity = 1/eps)
    dtau_star = cfl * eps * dx_star
    Nt = int(np.ceil(tau_star_max / dtau_star))

    # State arrays
    C_star = np.zeros(Nx)
    n_star = np.zeros(Nx)

    if store_history:
        C_star_hist = np.zeros((Nt, Nx))
        n_star_hist = np.zeros((Nt, Nx))
    else:
        C_star_hist = None
        n_star_hist = None

    tau_star = 0.0
    tau_star_break = tau_star_max
    k_eff = 0  # last used step index

    for k in range(Nt):
        # Inlet BC: C*(0) = 1
        C_star[0] = 1.0

        # Equilibrium loading
        n_eq_star = n_eq_star_langmuir(C_star, Lambda_star)

        # Kinetics: n*_tau
        n_star_tau = Da2_star * (n_eq_star - n_star) ** 2

        # Upwind advection for C*
        C_shift = np.empty_like(C_star)
        C_shift[0] = 1.0
        C_shift[1:] = C_star[:-1]
        dCdx_star = (C_star - C_shift) / dx_star

        # Mass balance: C*_tau = -phi* n*_tau - (1/eps) C*_x
        C_star_tau = -phi_star * n_star_tau - (1.0 / eps) * dCdx_star

        # Explicit Euler update
        C_star_new = C_star + dtau_star * C_star_tau
        n_star_new = n_star + dtau_star * n_star_tau

        # Enforce inlet BC after update
        C_star_new[0] = 1.0

        C_star = C_star_new
        n_star = n_star_new
        tau_star += dtau_star
        k_eff = k

        if store_history:
            C_star_hist[k, :] = C_star
            n_star_hist[k, :] = n_star

        # Breakthrough check at outlet
        if C_star[-1] >= C_star_thresh:
            tau_star_break = tau_star
            break

    if store_history:
        C_star_hist = C_star_hist[:k_eff+1, :]
        n_star_hist = n_star_hist[:k_eff+1, :]

    # if not store_history:
    #     return tau_star_break


    return tau_star_break, C_star_hist, n_star_hist
