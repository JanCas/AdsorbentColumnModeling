"""Containers for model / analysis outputs.

Holds the small result records that the AlLDH and IX drivers hand back to the
orchestration layer for tabulation and plotting. Right now this is limited to
the per-breakthrough-curve dimensionless-number diagnostics computed from a
``Study`` (see ``Dataclasses.Study``); the richer per-cycle QoIs (tau_ads,
tau_des, U_b, recovery, productivity, outlet pH …) are returned as plain dicts
by the Sobol adapters, not through this module.
"""

from dataclasses import dataclass
from .Dataclasses import BreakthroughCurve
import uuid


@dataclass
class SimulationResult:
    """Dimensionless transport diagnostics for one breakthrough-curve operating point.

    These characterize the flow/mass-transfer regime of a column run and are
    used to size axial dispersion and mass-transfer terms before/around a
    simulation. All are dimensionless except the flowrate and D_L.
    """
    curve_flowrate: float                    # flowrate of the source curve (units as stored on the curve, e.g. BV/h)
    Reynolds_number: float                   # particle Reynolds number Re = rho*u*d_p/mu (flow regime)
    Peclet_number_particle: float            # particle Peclet number Pe = u*d_p/D_mol (advection vs molecular diffusion)
    Peclet_number_axial_particle: float      # axial Peclet based on particle diameter (Pe_L,d from the D_L correlation)
    Peclet_number_axial_column: float        # axial Peclet based on column length (Pe_L = u*L/D_L)
    D_L: float                               # axial dispersion coefficient [m^2/s]
    Schmidt_number: float                    # Schmidt number Sc = mu/(rho*D_mol) (momentum vs mass diffusivity)
    Sherwood_number: float                   # Sherwood number Sh (dimensionless external mass-transfer coefficient)