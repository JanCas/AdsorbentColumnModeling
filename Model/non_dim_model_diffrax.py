# non_dim_model_diffrax.py

import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from typing import Tuple, Optional


def n_eq_star_temkin_jax(C_star: jnp.ndarray, Lambda_star: float) -> jnp.ndarray:
    """
    Dimensionless Temkin isotherm (JAX version):

        n_eq*(C*) = 1 + ln(C*) / Lambda*

    Parameters
    ----------
    C_star : jnp.ndarray
        Dimensionless liquid concentration C* = C / C0.
    Lambda_star : float
        Dimensionless Temkin parameter, Lambda* = ln(A_si * C0).

    Returns
    -------
    n_eq_star : jnp.ndarray
        Dimensionless equilibrium loading n_eq* = n_eq / n0.
    """
    C_safe = jnp.clip(C_star, 1e-30, None)  # avoid log(0)
    return 1.0 + jnp.log(C_safe) / Lambda_star


def column_ode_system(
    t: float,
    y: jnp.ndarray,
    args: Tuple[float, float, float, float, int]
) -> jnp.ndarray:
    """
    ODE system for the column model with Temkin isotherm.

    The PDE system:
        C*_tau + phi* n*_tau + (1/eps) C*_x = 0
        n*_tau = Da2* (n_eq*(C*) - n*)^2

    is discretized spatially using upwind finite differences.

    Parameters
    ----------
    t : float
        Dimensionless time tau*
    y : jnp.ndarray
        State vector [C_star; n_star] of shape (2*Nx,)
    args : tuple
        (phi_star, Da2_star, Lambda_star, eps, Nx)

    Returns
    -------
    dydt : jnp.ndarray
        Time derivatives [dC_star/dtau; dn_star/dtau]
    """
    phi_star, Da2_star, Lambda_star, eps, Nx = args
    dx_star = 1.0 / (Nx - 1)

    # Extract C* and n* from state vector
    C_star = y[:Nx]
    n_star = y[Nx:]

    # Apply inlet BC: C*(0) = 1
    C_star = C_star.at[0].set(1.0)

    # Equilibrium loading
    n_eq_star = n_eq_star_temkin_jax(C_star, Lambda_star)

    # Kinetics: n*_tau = Da2* (n_eq* - n*)^2
    n_star_tau = Da2_star * (n_eq_star - n_star) ** 2

    # Upwind advection for C* (first-order upwind)
    # dC/dx ≈ (C_i - C_{i-1}) / dx for positive velocity
    C_shift = jnp.concatenate([jnp.array([1.0]), C_star[:-1]])
    dCdx_star = (C_star - C_shift) / dx_star

    # Mass balance: C*_tau = -phi* n*_tau - (1/eps) C*_x
    C_star_tau = -phi_star * n_star_tau - (1.0 / eps) * dCdx_star

    # Enforce inlet BC derivative (C* at inlet is fixed)
    C_star_tau = C_star_tau.at[0].set(0.0)

    # Combine derivatives
    dydt = jnp.concatenate([C_star_tau, n_star_tau])

    return dydt


def simulate_column_temkin_star_diffrax(
    phi_star: float,
    Da2_star: float,
    Lambda_star: float,
    eps: float = 0.4,
    Nx: int = 200,
    tau_star_max: float = 15.0,
    C_star_thresh: float = 0.05,
    solver: str = "Dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_steps: int = 100000,
    store_history: bool = False,
    saveat_points: Optional[int] = None,
):
    """
    1D dimensionless column model with Temkin isotherm using diffrax ODE solver.

    The PDE system:
        C*_tau + phi* n*_tau + (1/eps) C*_x = 0
        n*_tau = Da2* (n_eq*(C*) - n*)^2
        n_eq*(C*) = 1 + ln(C*) / Lambda*

    is solved using method-of-lines with diffrax adaptive ODE solvers.

    Parameters
    ----------
    phi_star : float
        Dimensionless capacity ratio φ*.
    Da2_star : float
        Dimensionless pseudo-second-order Damköhler number Da2*.
    Lambda_star : float
        Dimensionless Temkin parameter Λ* = ln(A_si * C0).
    eps : float
        Porosity ε.
    Nx : int
        Number of spatial grid points.
    tau_star_max : float
        Max dimensionless time τ*.
    C_star_thresh : float
        Breakthrough threshold at outlet.
    solver : str
        Diffrax solver name: "Dopri5", "Dopri8", "Tsit5", "Heun", "Euler"
    rtol : float
        Relative tolerance for adaptive solvers.
    atol : float
        Absolute tolerance for adaptive solvers.
    max_steps : int
        Maximum number of steps for the solver.
    store_history : bool
        If True, returns solution history.
    saveat_points : int or None
        Number of points to save in history (if store_history=True).

    Returns
    -------
    tau_star_break : float
        Dimensionless breakthrough time when C*_out >= C_star_thresh.
    C_star_hist : ndarray or None
        Shape (Nt_save, Nx) if store_history=True.
    n_star_hist : ndarray or None
        Shape (Nt_save, Nx) if store_history=True.
    """

    # Select solver
    solver_map = {
        "Dopri5": diffrax.Dopri5(),
        "Dopri8": diffrax.Dopri8(),
        "Tsit5": diffrax.Tsit5(),
        "Heun": diffrax.Heun(),
        "Euler": diffrax.Euler(),
    }

    if solver not in solver_map:
        raise ValueError(f"Unknown solver: {solver}. Choose from {list(solver_map.keys())}")

    diffrax_solver = solver_map[solver]

    # Initial conditions
    y0 = jnp.zeros(2 * Nx)

    # ODE term
    ode_term = diffrax.ODETerm(column_ode_system)

    # Arguments for the ODE system
    args = (phi_star, Da2_star, Lambda_star, eps, Nx)

    # Time points to save
    if store_history and saveat_points is not None:
        ts = jnp.linspace(0, tau_star_max, saveat_points)
        saveat = diffrax.SaveAt(ts=ts)
    else:
        saveat = diffrax.SaveAt(t1=True)  # Only save final time

    # Step size controller (for adaptive solvers)
    if solver in ["Dopri5", "Dopri8", "Tsit5"]:
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        dt0 = None  # Adaptive initial step
    else:
        # For fixed-step solvers like Euler
        dt0 = tau_star_max / 1000  # Fixed step size
        stepsize_controller = diffrax.ConstantStepSize()

    # Event function to detect breakthrough
    def event_fn(state, **kwargs):
        """Detect when C* at outlet exceeds threshold."""
        y = state.y
        C_star_outlet = y[Nx - 1]  # Last spatial point
        return C_star_outlet - C_star_thresh

    event = diffrax.Event(event_fn)

    # Solve the ODE system
    sol = diffrax.diffeqsolve(
        ode_term,
        diffrax_solver,
        t0=0.0,
        t1=tau_star_max,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        event=event,
    )

    # Extract breakthrough time
    if sol.event_mask is not None and jnp.any(sol.event_mask):
        # Event was triggered (breakthrough occurred)
        tau_star_break = float(sol.ts[-1])
    else:
        tau_star_break = tau_star_max

    # Process history if requested
    if store_history:
        # Convert JAX arrays to numpy
        if saveat_points is not None:
            # Multiple time points saved
            ys = np.array(sol.ys)
            C_star_hist = ys[:, :Nx]
            n_star_hist = ys[:, Nx:]
        else:
            # Only final time saved
            y_final = np.array(sol.ys)
            C_star_hist = y_final[:Nx].reshape(1, -1)
            n_star_hist = y_final[Nx:].reshape(1, -1)

        return tau_star_break, C_star_hist, n_star_hist
    else:
        return tau_star_break


def simulate_with_event_detection(
    phi_star: float,
    Da2_star: float,
    Lambda_star: float,
    eps: float = 0.4,
    Nx: int = 200,
    tau_star_max: float = 15.0,
    C_star_thresh: float = 0.05,
    solver: str = "Dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    Simplified version focused on breakthrough time detection with efficient event handling.

    This version uses diffrax's event detection to stop integration immediately
    when breakthrough occurs, making it more efficient for parameter studies.

    Parameters
    ----------
    phi_star : float
        Dimensionless capacity ratio φ*.
    Da2_star : float
        Dimensionless pseudo-second-order Damköhler number Da2*.
    Lambda_star : float
        Dimensionless Temkin parameter Λ* = ln(A_si * C0).
    eps : float
        Porosity ε.
    Nx : int
        Number of spatial grid points.
    tau_star_max : float
        Max dimensionless time τ*.
    C_star_thresh : float
        Breakthrough threshold at outlet.
    solver : str
        Diffrax solver name.
    rtol : float
        Relative tolerance for adaptive solvers.
    atol : float
        Absolute tolerance for adaptive solvers.

    Returns
    -------
    tau_star_break : float
        Dimensionless breakthrough time.
    C_star_final : ndarray
        Final concentration profile.
    n_star_final : ndarray
        Final loading profile.
    """

    # Select solver
    solver_map = {
        "Dopri5": diffrax.Dopri5(),
        "Dopri8": diffrax.Dopri8(),
        "Tsit5": diffrax.Tsit5(),
        "Heun": diffrax.Heun(),
        "Euler": diffrax.Euler(),
    }

    diffrax_solver = solver_map.get(solver, diffrax.Dopri5())

    # Initial conditions
    y0 = jnp.zeros(2 * Nx)

    # ODE term
    ode_term = diffrax.ODETerm(column_ode_system)

    # Arguments
    args = (phi_star, Da2_star, Lambda_star, eps, Nx)

    # Step size controller
    if solver in ["Dopri5", "Dopri8", "Tsit5"]:
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        dt0 = None
    else:
        dt0 = tau_star_max / 1000
        stepsize_controller = diffrax.ConstantStepSize()

    # Event function for breakthrough
    def event_fn(state, **kwargs):
        y = state.y
        C_star_outlet = y[Nx - 1]
        return C_star_outlet - C_star_thresh

    event = diffrax.Event(event_fn)

    # Solve
    sol = diffrax.diffeqsolve(
        ode_term,
        diffrax_solver,
        t0=0.0,
        t1=tau_star_max,
        dt0=dt0,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
        event=event,
    )

    # Extract results
    tau_star_break = float(sol.ts[-1])
    y_final = np.array(sol.ys[-1])
    C_star_final = y_final[:Nx]
    n_star_final = y_final[Nx:]

    return tau_star_break, C_star_final, n_star_final


if __name__ == "__main__":
    # Example usage and comparison with original method
    import time

    # Test parameters
    phi_star = 1.5
    Da2_star = 10.0
    Lambda_star = 2.0
    eps = 0.4
    Nx = 100
    tau_star_max = 15.0
    C_star_thresh = 0.05

    print("Testing diffrax implementation...")
    print(f"Parameters: phi*={phi_star}, Da2*={Da2_star}, Lambda*={Lambda_star}")
    print(f"Grid: Nx={Nx}, eps={eps}")
    print()

    # Test different solvers
    solvers = ["Euler", "Heun", "Dopri5", "Tsit5"]

    for solver_name in solvers:
        print(f"Testing {solver_name} solver...")
        start = time.time()

        try:
            if solver_name in ["Euler", "Heun"]:
                # Fixed-step solvers need smaller tolerances
                tau_break, C_final, n_final = simulate_with_event_detection(
                    phi_star, Da2_star, Lambda_star, eps, Nx,
                    tau_star_max, C_star_thresh, solver=solver_name
                )
            else:
                # Adaptive solvers
                tau_break, C_final, n_final = simulate_with_event_detection(
                    phi_star, Da2_star, Lambda_star, eps, Nx,
                    tau_star_max, C_star_thresh, solver=solver_name,
                    rtol=1e-5, atol=1e-8
                )

            elapsed = time.time() - start
            print(f"  Breakthrough time: τ* = {tau_break:.4f}")
            print(f"  Outlet concentration: C*_out = {C_final[-1]:.6f}")
            print(f"  Computation time: {elapsed:.3f} s")

        except Exception as e:
            print(f"  Error: {e}")

        print()

    # Test with history storage
    print("Testing with history storage (Dopri5)...")
    start = time.time()
    tau_break, C_hist, n_hist = simulate_column_temkin_star_diffrax(
        phi_star, Da2_star, Lambda_star, eps, Nx,
        tau_star_max, C_star_thresh,
        solver="Dopri5", rtol=1e-5, atol=1e-8,
        store_history=True, saveat_points=100
    )
    elapsed = time.time() - start

    print(f"Breakthrough time: τ* = {tau_break:.4f}")
    print(f"History shape: C* {C_hist.shape}, n* {n_hist.shape}")
    print(f"Computation time: {elapsed:.3f} s")