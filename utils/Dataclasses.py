"""Parameter dataclasses describing an experimental adsorption study.

This is the shared "input side" of the modeling framework. It parses raw
literature/experimental data (loaded from JSON) into typed, immutable
dataclasses and exposes:

* unit-aware ``*_si`` accessors that convert every stored quantity into SI
  units (m, s, mol/m^3, mol/kg, kg/m^3, K) so the physics code never has to
  worry about the units a given source reported;
* derived engineering quantities for a packed bed — velocities, Reynolds /
  Peclet / Schmidt / Sherwood numbers, axial dispersion, mass-transfer
  coefficients — computed from a ``Study``;
* ``ColumnParameters``, an Equinox module of JAX arrays that packages the
  handful of parameters the dimensional AlLDH forward model actually needs.

The top-level object is ``Study`` (built via ``Study.from_json``); it aggregates
column geometry, sorbent properties, an equilibrium isotherm fit, and
pseudo-second-order kinetics experiments. See PROJECT_GUIDE.html sections 4.1
(dimensional AlLDH parameter table) and 2/3 (physical system) for the meaning of
these quantities.

Convention: fields hold values in the units the source reported (with a paired
``*_units`` string); the matching ``*_si`` property performs the conversion.
"""

from dataclasses import dataclass, asdict, field
import numpy as np
from typing import ClassVar
from abc import ABC, abstractmethod
import json
from pathlib import Path
import casadi as ca
from thermo import IAPWS95Liquid as Water
import uuid
from pyEQL import Solution
import equinox as eqx
import jax.numpy as jnp

Li_MW = 6.94  # molar mass of lithium [g/mol]; used for mg<->mol conversions
# Infinite-dilution-style diffusion coefficient of Li+ in water at temperature T [K]
# and concentration c [mol/m^3], via pyEQL. Returns D in m^2/s.
D_Li_in_H2O = lambda T,c: Solution({"Li+": f"{c} mol/m^3"}, temperature=f'{T}K').get_diffusion_coefficient("Li+").magnitude  # m²/s
ideal_gas_constant = 8.314  # universal gas constant R [J/(mol·K)]

def convert_mg_per_L_to_mol_per_m3(concentration_mg_per_L: float | np.ndarray) -> float | np.ndarray:
    "Converts concentration from mg/L to mol/m³."
    return concentration_mg_per_L / Li_MW  # Convert to mol/m³

def convert_mg_per_g_to_mol_per_kg(q_mg_per_g: float | np.ndarray) -> float | np.ndarray:
    "Converts adsorption capacity from mg/g to mol/kg."
    return q_mg_per_g / Li_MW  # Convert to mol/kg


@dataclass(frozen=True)
class PseudoSecondOrderKineticsParameters:
    """Fitted pseudo-second-order (PSO) uptake kinetics for one batch experiment.

    PSO law: dq/dt = k2 (q_e - q)^2. Provides SI accessors so the column model
    can use k2 in mol/kg/s and q_e in mol/kg regardless of the reported units.
    """
    k2: float          # PSO rate constant (units given by k2_units)
    q_e: float         # fitted equilibrium solid loading (units given by q_e_units)
    k2_units: str      # e.g. "mol/kg/s" or "g/mg/min"
    q_e_units: str     # e.g. "mol/kg" or "mg/g"

    @property
    def k2_si(self) -> float:
        """
            Returns the rate constant in mol_Li/kg_sorbent/s
        """
        match self.k2_units:
            case "mol/kg/s":
                return self.k2
            case "g/mg/min":
                return self.k2 * Li_MW / 60  # Convert to mol/kg/s
            case _:
                raise ValueError(f"Unsupported k2 units: {self.k2_units}")

    @property
    def q_e_si(self) -> float:
        """
            returns the "fitted" equilibrium concentration in mol_li/kg_sorbent
        """
        match self.q_e_units:
            case "mol/kg":
                return self.q_e
            case "mg/g":
                return convert_mg_per_g_to_mol_per_kg(self.q_e)
            case _:
                raise ValueError(f"Unsupported q_e units: {self.q_e_units}")
    
    @classmethod
    def from_dict(cls, data: dict) -> "PseudoSecondOrderKineticsParameters":
        return cls(
            k2=data["k2"],
            q_e=data["q_e"],
            k2_units=data["k2_units"],
            q_e_units=data["q_e_units"]
        )

@dataclass(frozen=True)
class KineticsUnits:
    """Units declared for a batch-kinetics dataset (parsed alongside the data)."""
    time: str    # units of the time series, e.g. "min" or "s"
    q_t: str     # units of solid loading q(t), e.g. "mg/g" or "mol/kg"
    T: str       # units of temperature, e.g. "K"
    C_e: str     # units of equilibrium liquid conc., e.g. "mg/L" or "mol/m3"

    @classmethod
    def from_dict(cls, data: dict) -> "KineticsUnits":
        return cls(
            time=data["time"],
            q_t=data["q_t"],
            T=data["T"],
            C_e=data["C_e"]
        )

@dataclass(frozen=True)
class KineticsExperiment:
    """One batch-uptake experiment: a q(t) curve at fixed T, pH and C_e, plus its PSO fit."""
    T: float                                              # temperature [K]
    C_e: float                                            # equilibrium liquid concentration (units in kinetics_units)
    Ph: float                                             # solution pH [-]
    time: list[float]                                     # time samples (units in kinetics_units)
    q_t: list[float]                                      # measured solid loading at each time (units in kinetics_units)
    kinetics_params: PseudoSecondOrderKineticsParameters  # fitted PSO rate constant / equilibrium loading
    kinetics_units: KineticsUnits                         # units metadata for the fields above

    @property
    def time_si(self) -> np.ndarray:
        "Converts the time array to seconds depending on the units specified."
        match self.kinetics_units.time:
            case "min":
                return np.array(self.time) * 60  # Convert to seconds
            case "s":
                return np.array(self.time)
            case _:
                raise ValueError(f"Unsupported time units: {self.kinetics_units.time}")
    
    @property
    def q_t_si(self) -> np.ndarray:
        "Converts the q_t array to mol/kg depending on the units specified."
        match self.kinetics_units.q_t:
            case "mg/g":
                return convert_mg_per_g_to_mol_per_kg(np.array(self.q_t))
            case "mol/kg":
                return np.array(self.q_t)
            case _:
                raise ValueError(f"Unsupported q_t units: {self.kinetics_units.q_t}")

    @property
    def C_e_si(self) -> float:
        "Converts the equilibrium concentration to mol/m³ depending on the units specified."
        match self.kinetics_units.C_e:
            case "mg/L":
                return convert_mg_per_L_to_mol_per_m3(self.C_e)
            case "mol/m3":
                return self.C_e
            case _:
                raise ValueError(f"Unsupported C_e units: {self.kinetics_units.C_e}")
         

    @classmethod
    def from_dict(cls, data: dict, units: KineticsUnits) -> "KineticsExperiment":
        kinetics_params = PseudoSecondOrderKineticsParameters.from_dict(data["FitParameters"])
        return cls(
            T=data["T"],
            C_e=data["C_e"],
            time=data["time"],
            q_t=data["q_t"],
            Ph=data["Ph"],
            kinetics_params=kinetics_params,
            kinetics_units=units
        )

@dataclass(frozen=True)
class ColumnProperties:
    """Geometry and packing of the physical column (bed dimensions + void fraction)."""
    Length: float          # packed-bed length (units in Length_units)
    Length_units: str      # e.g. "m"
    Diameter: float        # internal column diameter (units in Diameter_units)
    Diameter_units: str    # e.g. "m"
    porosity: float        # bed void fraction epsilon [-] (fraction of volume that is liquid)

    @property
    def tortuosity(self) -> float:
        "Calculates the tortuosity of the column based on the bruggeman correlation."
        return (self.porosity) ** (-1/2)
    
    @property
    def Length_si(self) -> float:
        "Converts the column length to meters depending on the units specified."
        match self.Length_units:
            case "m":
                return self.Length
            case _:
                raise ValueError(f"Unsupported Length units: {self.Length_units}")
            
    @property
    def Diameter_si(self) -> float:
        "Converts the column diameter to meters depending on the units specified."
        match self.Diameter_units:
            case "m":
                return self.Diameter
            case _:
                raise ValueError(f"Unsupported Diameter units: {self.Diameter_units}")
            
    @property
    def cross_sectional_area_si(self) -> float:
        "Calculates the cross-sectional area of the column in m²."
        radius = self.Diameter_si / 2
        return np.pi * radius ** 2
    
    @property
    def volume_si(self) -> float:
        "Calculates the volume of the column in m³."
        return self.cross_sectional_area_si * self.Length_si

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnProperties":
        return cls(
            Length=data["Length"],
            Length_units=data["Length_units"],
            Diameter=data["Diameter"],
            Diameter_units=data["Diameter_units"],
            porosity=data["porosity"]
        )

@dataclass(frozen=True)
class BreakthroughCurveUnits:
    """Units declared for a set of column breakthrough curves."""
    flowrate: str                 # e.g. "BV/h" (bed volumes per hour) or "m3/s"
    influent_concentration: str   # e.g. "mg/L" or "mol/m3"
    T: str                        # e.g. "K"

    @classmethod
    def from_dict(cls, data: dict) -> "BreakthroughCurveUnits":
        return cls(
            flowrate=data["flowrate"],
            influent_concentration=data["influent_concentration"],
            T=data["T"]
        )

@dataclass(frozen=True)
class BreakthroughCurve:
    """A single measured column breakthrough curve (outlet/inlet vs bed volumes)."""
    flowrate: int                    # feed flowrate at this operating point (units in BreakthroughCurveUnits)
    T: float                         # temperature [K]
    PH: float                        # feed pH [-]
    influent_concentration: float    # feed (inlet) concentration C_in (units in BreakthroughCurveUnits)
    BV: list[float]                  # x-axis: cumulative bed volumes processed [-]
    C_out_over_C_in: list[float]     # y-axis: normalized outlet concentration C_out/C_in in [0,1]
    uuid: uuid.UUID = field(default_factory=uuid.uuid4)  # stable id for this curve (matching/filtering)

    @property
    def run_length_hours(self) -> float:
        "Calculates the run length in hours based on the bed volumes and flowrate."
        total_BV = self.BV[-1]
        return total_BV / self.flowrate  # in hours

    @classmethod
    def from_dict(cls, data: dict) -> "BreakthroughCurve":
        return cls(
            flowrate=data["flowrate"],
            T=data["T"],
            PH=data["PH"],
            influent_concentration=data["influent_concentration"],
            BV=data["BV"],
            C_out_over_C_in=data["C_out/C_in"]
        )
    
@dataclass(frozen=True)
class BreakthroughCurves:
    """A collection of breakthrough curves sharing one set of units, with filtering."""
    curves: tuple[BreakthroughCurve, ...]   # all measured curves in the study
    units: BreakthroughCurveUnits           # units shared by every curve

    def filter(self, **filter) -> tuple[BreakthroughCurve, ...]:
        if not filter:
            return self.curves

        filtered_curves = [
            curve for curve in self.curves
            if all(getattr(curve, key) == value for key, value in filter.items())
        ]
        return filtered_curves
    
    @classmethod
    def from_dict(cls, data: dict) -> "BreakthroughCurves":
        units = BreakthroughCurveUnits.from_dict(data["Units"])
        curves = [
            BreakthroughCurve.from_dict(curve_data)
            for curve_data in data["Curves"]
        ]
        return cls(
            curves=curves,
            units=units
        )

@dataclass(frozen=True)
class ColumnExperiments:
    """Column geometry paired with its breakthrough curves.

    Provides SI conversions and derived hydrodynamics (superficial/interstitial
    velocity) for any curve or subset selected by ``**filter``.
    """
    column_properties: ColumnProperties       # bed geometry + porosity
    breakthrough_curves: BreakthroughCurves    # measured curves for this column

    def superficial_flowrate_si(self, curve: BreakthroughCurve=None, **filter) -> float:
        "Converts the flowrate to m3/s depending on the units specified."
        if curve is not None:
            filter.update(asdict(curve))
        filtered_curves = self.breakthrough_curves.filter(**filter)

        match self.breakthrough_curves.units.flowrate:
            case "BV/h":
                return [curve.flowrate * self.column_properties.volume_si / 3600 for curve in filtered_curves]  # Convert to m3/s
            case "m3/s":
                return [curve.flowrate for curve in filtered_curves]
            case _:
                raise ValueError(f"Unsupported flowrate units: {self.breakthrough_curves.units.flowrate}")

    def influent_concentration_si(self, curve: BreakthroughCurve=None, **filter) -> float:
        "Converts the influent concentration to mol/m³ depending on the units specified."
        if curve is not None:
            filter.update(asdict(curve))
        filtered_curves = self.breakthrough_curves.filter(**filter)

        match self.breakthrough_curves.units.influent_concentration:
            case "mg/L":
                return [convert_mg_per_L_to_mol_per_m3(curve.influent_concentration) for curve in filtered_curves]
            case "mol/m3":
                return [curve.influent_concentration for curve in filtered_curves]
            case _:
                raise ValueError(f"Unsupported influent concentration units: {self.breakthrough_curves.units.influent_concentration}")

    def superficial_velocity_si(self, curve: BreakthroughCurve=None, **filter) -> float:
        "Calculates the superficial velocity in m/s."
        return np.array(self.superficial_flowrate_si(curve, **filter)) / self.column_properties.cross_sectional_area_si
    
    def interstitial_velocity_si(self, curve: BreakthroughCurve=None, **filter) -> float:
        "Calculates the interstitial velocity in m/s."
        return np.array(self.superficial_velocity_si(curve, **filter)) / self.column_properties.porosity

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnExperiments":
        return cls(
            column_properties=ColumnProperties.from_dict(data["Properties"]),
            breakthrough_curves=BreakthroughCurves.from_dict(data["BreakthroughCurves"])
        )
    
@dataclass(frozen=True)
class SorbentProperties:
    """Physical properties of the packed sorbent grains."""
    density: float                  # particle (grain) density rho_p (units in density_units)
    density_units: str              # e.g. "kg/m3"
    particle_diameter: float        # sorbent particle diameter d_p (units in particle_diameter_units)
    particle_diameter_units: str    # e.g. "m"

    @property
    def density_si(self) -> float:
        "Converts the sorbent density to kg/m³ depending on the units specified."
        match self.density_units:
            case "kg/m3":
                return self.density
            case _:
                raise ValueError(f"Unsupported density units: {self.density_units}")

    @property
    def particle_diameter_si(self) -> float:
        "Converts the particle diameter to meters depending on the units specified."
        match self.particle_diameter_units:
            case "m":
                return self.particle_diameter
            case _:
                raise ValueError(f"Unsupported particle diameter units: {self.particle_diameter_units}")

    @classmethod
    def from_dict(cls, data: dict) -> "SorbentProperties":
        return cls(
            density=data["density"],
            density_units=data["density_units"],
            particle_diameter=data["particle_diameter"],
            particle_diameter_units=data["particle_diameter_units"]
        )

class BaseIsothermFit(eqx.Module):
    """
    Abstract base class for isotherm fit models.

    An isotherm maps liquid concentration C to the equilibrium solid loading
    q_eq(C). Subclasses (Temkin, Sips) implement ``q_eq_si`` in SI units and are
    passed to the column model as a callable isotherm. Subclasses are Equinox
    modules so they can live inside a JAX pytree.
    """

    fit_type: ClassVar[str]   # short string tag identifying the isotherm family

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseIsothermFit":
        ...

    @abstractmethod
    def q_eq_si(self, C: float) -> float:
        ...

    def __call__(self, C: float) -> float:
        return self.q_eq_si_jax(C)


class ColumnParameters(eqx.Module):
    """JAX-array parameter bundle consumed by the dimensional AlLDH column model.

    This is exactly the parameter set in PROJECT_GUIDE §4.1: the leaves are JAX
    arrays so the forward model can be JIT-compiled once and reused across
    parameter values, and ``isotherm`` is a callable pytree leaf.
    """
    u_inter: jnp.ndarray        # interstitial (pore) velocity u_int = u_super/epsilon [m/s]
    k_s: jnp.ndarray            # PSO kinetic rate constant k_s [1/s]
    epsilon: jnp.ndarray        # bed porosity [-]
    C_in: jnp.ndarray           # feed concentration [mol/m^3]
    L: jnp.ndarray              # column length [m]
    rho_p: jnp.ndarray          # particle density [kg/m^3]
    isotherm: BaseIsothermFit   # equilibrium isotherm callable C -> q_eq [mol/kg]

    def __post_init__(self):
        # Keep numeric leaves as JAX arrays so JIT can reuse one compiled executable
        # across different parameter values instead of recompiling per Python float.
        object.__setattr__(self, "u_inter", jnp.asarray(self.u_inter))
        object.__setattr__(self, "k_s", jnp.asarray(self.k_s))
        object.__setattr__(self, "epsilon", jnp.asarray(self.epsilon))
        object.__setattr__(self, "C_in", jnp.asarray(self.C_in))
        object.__setattr__(self, "L", jnp.asarray(self.L))
        object.__setattr__(self, "rho_p", jnp.asarray(self.rho_p))

    def replace(self, **kwargs) -> "ColumnParameters":
        """Return a new instance with updated values."""
        return eqx.tree_at(
            lambda x: [getattr(x, k) for k in kwargs.keys()],
            self,
            list(kwargs.values()),
        )

class SipsIsothermFit(BaseIsothermFit):
    """Sips (Langmuir-Freundlich) isotherm: q_eq = Q_max (K_s C)^n / (1 + (K_s C)^n)."""
    fit_type: ClassVar[str] = "Sips"
    K_s: float          # Sips affinity constant (units in K_s_units)
    n: float            # Sips heterogeneity exponent [-]
    Q_max: float        # saturation capacity (units in Q_max_units)
    K_s_units: str      # e.g. "L/mg" or "m3/mol"
    Q_max_units: str    # e.g. "mg/g" or "mol/kg"

    def __post_init__(self):
        Warning.warn("Sips Isotherm model is not yet implemented.")

    @property
    def K_s_si(self) -> float:
        """
        Converts the Sips isotherm parameter K_s to SI units (m³/mol).
        """
        match self.K_s_units:
            case "L/mg":
                return self.K_s * Li_MW  # Convert to m³/mol
            case "m3/mol":
                return self.K_s
            case _:
                raise ValueError(f"Unsupported K_s units: {self.K_s_units}")
            
    @property
    def Q_max_si(self) -> float:
        """
        Converts the Sips isotherm parameter Q_max to SI units (mol/kg).
        """
        match self.Q_max_units:
            case "mg/g":
                return convert_mg_per_g_to_mol_per_kg(self.Q_max)
            case "mol/kg":
                return self.Q_max
            case _:
                raise ValueError(f"Unsupported Q_max units: {self.Q_max_units}")
            
    def q_eq_si(self, C_eq_si: float) -> float:
        """
        Calculates the equilibrium uptake (q_eq) in mol/kg using the Sips isotherm model.
        C_eq_si: Equilibrium concentration in mol/m³
        Returns q_eq in mol/kg
        """
        q_e = (self.Q_max_si * (self.K_s_si * C_eq_si) ** self.n) / (1 + (self.K_s_si * C_eq_si) ** self.n)
        return q_e

class TemkinIsothermFit(BaseIsothermFit):
    """Temkin isotherm: q_eq = (RT/B) ln(A C), floored at 0 (used as the AlLDH isotherm)."""
    fit_type: ClassVar[str] = "Temkin"
    A: float          # Temkin equilibrium binding constant (units in A_units)
    B: float          # Temkin energy parameter (units in B_units); related to heat of sorption
    A_units: str      # e.g. "L/mg" or "m3/mol"
    B_units: str      # e.g. "kJ/mol" or "J/mol"
    T: float          # temperature at which the fit applies [K]

    @property
    def A_si(self) -> float:
        """
        Converts the Temkin isotherm parameter A to SI units (m³/mol).
        """
        match self.A_units:
            case "L/mg":
                return self.A * Li_MW  # Convert to m³/mol
            case "m3/mol":
                return self.A
            case _:
                raise ValueError(f"Unsupported A units: {self.A_units}")
    
    @property
    def B_si(self) -> float:
        """
        Converts the Temkin isotherm parameter B to SI units (J/mol).
        """
        match self.B_units:
            case "kJ/mol":
                return self.B * 1000 * Li_MW  # Convert kJ/mol to J/mol
            case "J/mol":
                return self.B
            case _:
                raise ValueError(f"Unsupported B units: {self.B_units}")

    def q_eq_si(self, C_eq_si: float) -> float:
        """
        Calculates the equilibrium uptake (q_eq) in mol/kg using the Temkin isotherm model.
        C_eq_si: Equilibrium concentration in mol/m³
        T: Temperature in K
        Returns q_eq in mol/kg
        """
        interm = ideal_gas_constant * self.T / self.B_si
        return ca.fmax(interm * ca.log(self.A_si) + interm * ca.log(C_eq_si), 0)
    
    def q_eq_si_jax(self, C_eq_si):
        """
        Calculates the equilibrium uptake (q_eq) in mol/kg using the Temkin isotherm model.
        C_eq_si: Equilibrium concentration in mol/m³
        T: Temperature in K
        Returns q_eq in mol/kg
        """
        interm = ideal_gas_constant * self.T / self.B_si
        return jnp.fmax(interm * jnp.log(self.A_si) + interm * jnp.log(C_eq_si), 0)

    @classmethod
    def from_dict(cls, data: dict, T: float) -> "TemkinIsothermFit":
        return cls(
            A=data["A"],
            B=data["B"],
            A_units=data["A_units"],
            B_units=data["B_units"],
            T = T
        )


# Registry mapping a fit_type tag (as stored in JSON) to its isotherm class.
IsothermFitDirectory = {
    TemkinIsothermFit.fit_type: TemkinIsothermFit,
    SipsIsothermFit.fit_type: SipsIsothermFit
}

@dataclass(frozen=True)
class IsothermUnits:
    """Units declared for an equilibrium isotherm dataset."""
    C_e: str    # units of equilibrium liquid concentration, e.g. "mg/L" or "mol/m3"
    q_e: str    # units of equilibrium solid loading, e.g. "mg/g" or "mol/kg"
    T: str      # units of temperature, e.g. "K"

    @classmethod
    def from_dict(cls, data: dict) -> "IsothermUnits":
        return cls(
            C_e=data["C_e"],
            q_e=data["q_e"],
            T=data["T"],
        )

@dataclass(frozen=True)
class Isotherm:
    """Measured equilibrium isotherm (q_e vs C_e) plus the fitted model for it."""
    isotherm_fit: BaseIsothermFit            # fitted callable isotherm (Temkin/Sips)
    T: float                                 # temperature of the isotherm [K]
    Ph: float                                # pH of the isotherm measurements [-]
    equilibrium_concentration_units: str     # units of the C_e list, e.g. "mg/L"
    equilibrium_concentration: list[float]   # measured equilibrium liquid concentrations C_e
    fit_type: str                            # which model was fit ("Temkin"/"Sips")
    equilibrium_uptake: list[float]          # measured equilibrium solid loadings q_e
    equilibrium_uptake_units: str            # units of the q_e list, e.g. "mg/g"
    units: IsothermUnits                     # units metadata bundle

    @property
    def equilibrium_concentration_si(self) -> np.ndarray:
        "Converts the equilibrium concentration to mol/m³ depending on the units specified."
        match self.equilibrium_concentration_units:
            case "mg/L":
                return convert_mg_per_L_to_mol_per_m3(np.array(self.equilibrium_concentration))
            case "mol/m3":
                return np.array(self.equilibrium_concentration)
            case _:
                raise ValueError(f"Unsupported equilibrium concentration units: {self.equilibrium_concentration_units}")
    
    @property
    def equilibrium_uptake_si(self) -> np.ndarray:
        "Converts the equilibrium uptake to mol/kg depending on the units specified."
        match self.equilibrium_uptake_units:
            case "mg/g":
                return convert_mg_per_g_to_mol_per_kg(np.array(self.equilibrium_uptake))
            case "mol/kg":
                return np.array(self.equilibrium_uptake)
            case _:
                raise ValueError(f"Unsupported equilibrium uptake units: {self.equilibrium_uptake_units}")

    @classmethod
    def from_dict(cls, data: dict, units: IsothermUnits) -> "Isotherm":
        match data["FitType"]:
            case "Temkin":
                isotherm_fit = TemkinIsothermFit.from_dict(data["FitParameters"], data["T"])
            case "Sips":
                isotherm_fit = SipsIsothermFit.from_dict(data["FitParameters"])
            case _:
                raise ValueError(f"Unsupported FitType: {data['FitType']}")

        return cls(
            isotherm_fit=isotherm_fit,
            T=data["T"],
            Ph=data["Ph"],
            equilibrium_concentration_units=data["C_e_units"],
            equilibrium_concentration=data["C_e"],
            fit_type=data["FitType"],
            equilibrium_uptake=data["q_e"],
            equilibrium_uptake_units=data["q_e_units"],
            units=units
        )


@dataclass(frozen=True)
class Study:
    """Top-level container for one adsorbent study loaded from JSON.

    Aggregates everything needed to parameterize and characterize a column:
    geometry + breakthrough curves, sorbent properties, the equilibrium
    isotherm, and batch kinetics experiments. Beyond storage it computes the
    engineering diagnostics (Reynolds/Peclet/Schmidt/Sherwood numbers, axial
    dispersion, mass-transfer coefficient, capacity factor & Damkohler number)
    and produces the JAX ``ColumnParameters`` the forward model runs on.
    Build via ``Study.from_json``.
    """
    column_experiments: ColumnExperiments             # geometry + breakthrough curves
    sorbent_properties: SorbentProperties             # grain density, particle diameter
    isotherm: Isotherm                                # equilibrium isotherm + fit
    kinetics_experiments: list[KineticsExperiment]    # batch PSO kinetics at various conditions

    def get_kinetics_experiment_from_curve(self, curve: BreakthroughCurve) -> KineticsExperiment:
        "Returns the kinetics experiment that matches the temperature and pH of the given breakthrough curve."
        for exp in self.kinetics_experiments:
            if (exp.T == curve.T) and (exp.Ph == curve.PH) and (exp.C_e_si == self.column_experiments.influent_concentration_si(**asdict(curve))[0]):
                return exp
        raise ValueError("No matching kinetics experiment found for the given breakthrough curve.")
    
    def get_kinetics_experiment_from_concentration(self, C) -> KineticsExperiment:
        return min(self.kinetics_experiments, key=lambda exp: abs(exp.C_e_si - C))

    def particle_reynolds(self, curve: BreakthroughCurve = None, **filter) -> list[float]:
        "Calculates the reynolds number for using the particle diameter for the breakthrough curves matching the filter."
        if curve is not None:
            filter.update(asdict(curve))
        curves= self.column_experiments.breakthrough_curves.filter(**filter)
        
        velocities = self.column_experiments.interstitial_velocity_si(**filter)
        dp = self.sorbent_properties.particle_diameter_si
        rho = [Water(T=curve.T).rho_mass() for curve in curves]
        mu = [Water(T=curve.T).mu() for curve in curves]

        return rho * velocities * dp / mu
    
    def particle_peclet_number(self, curve: BreakthroughCurve = None, **filter) -> list[float]:
        "calculates the peclet number based on the interstitial velocity for the breakthrough curves matching the filter."
        if curve is not None:
            filter.update(asdict(curve))
        velocities = self.column_experiments.interstitial_velocity_si(**filter)
        temperatures = [curve.T for curve in self.column_experiments.breakthrough_curves.filter(**filter)]
        concentrations = self.column_experiments.influent_concentration_si(**filter)
        D_Li_in_H2O_list = [D_Li_in_H2O(T, c) for T,c in zip(temperatures, concentrations)]

        return velocities * self.sorbent_properties.particle_diameter_si / D_Li_in_H2O_list

    def schmidt_number(self,curve: BreakthroughCurve = None,T: float = None) -> float:
        "Calculates the schmidt number at temperature T (K)."
        if curve is not None:
            T = [curve.T]
        elif T is None:
            T = [curve.T for curve in self.column_experiments.breakthrough_curves.curves]
        elif not isinstance(T, list):
            T = [T]
        
        water = [Water(T=temp) for temp in T]
        mu = [w.mu() for w in water]  # Dynamic viscosity in Pa.s
        rho = [w.rho_mass() for w in water]  # Density in kg/m³
        concentration = self.column_experiments.influent_concentration_si(curve, T=T[0])
        D_Li_in_H2O_list = [D_Li_in_H2O(temp, c) for temp, c in zip(T, concentration)]

        return np.array(mu) / (np.array(rho) * np.array(D_Li_in_H2O_list))

    def axial_dispersion_coefficient(self, curve: BreakthroughCurve = None, **filter) -> list[float]:
        '''
        Calculates the axial dispersion coefficient (D_L) for the breakthrough curves matching the filter.
        The axial dispersion is calculated based on DOI 10.1007/s00231-005-0019-0 equations 18 and 19.

        Returns: 
            D_L: list of axial dispersion coefficients in m²/s
            Pe_L_d: list of Peclet numbers based on sorbent particle diameter
            Pe_L: list of Peclet numbers based on column length
        '''
        
        # if curve is not None:
        #     filter.update(asdict(curve))

        velocities = self.column_experiments.interstitial_velocity_si(curve, **filter)
        dp = self.sorbent_properties.particle_diameter_si
        peclet_numbers = self.particle_peclet_number(curve, **filter)
        Sc = self.schmidt_number(curve, **filter)

        p = .48 / Sc**(.15) + (.5 - .48 / Sc**(.15)) * np.exp(-75 * Sc / peclet_numbers)
        term1 = (1-p)**2 * peclet_numbers / 5
        term2 = peclet_numbers**2 / 25 * p * (1-p)**3 * (np.exp(-5 / (p*(1-p)*peclet_numbers)) -1)
        term3 = 1 / (self.column_experiments.column_properties.tortuosity * peclet_numbers)
        
        inv_peclet = np.array(term1) + np.array(term2) + np.array(term3)
        Pe_L_d = 1 / inv_peclet

        D_L = velocities * dp / Pe_L_d
        Pe_L = velocities * self.column_experiments.column_properties.Length_si / D_L

        return D_L, Pe_L_d, Pe_L

    def sherwood_kataoka_1972(self, curve: BreakthroughCurve = None, **filter) -> list[float]:
        '''
        Calculates the Sherwood number using the correlation from Kataoka 1972 for the breakthrough curves matching the filter.

        Returns:
            Sh: list of Sherwood numbers
        '''
        
        porosity = self.column_experiments.column_properties.porosity
        Re = self.particle_reynolds(curve, **filter) * porosity
        Sc = self.schmidt_number(curve, **filter)

        Sh = 1.85 * ((1-porosity) / porosity)**(1/3) * Re**(1/3) * Sc**(1/3)

        return Sh

    def external_mass_transfer_coefficient(self, curve: BreakthroughCurve = None, **filter) -> list[float]:
        '''
        Calculates the external mass transfer coefficient (k_f) for the breakthrough curves matching the filter.

        Returns:
            k_f: list of external mass transfer coefficients in m/s
        '''
        
        Sh = self.sherwood_kataoka_1972(curve, **filter)
        dp = self.sorbent_properties.particle_diameter_si
        D_Li_in_H2O_list = [D_Li_in_H2O(curve.T, c) for curve, c in zip(self.column_experiments.breakthrough_curves.filter(**filter), self.column_experiments.influent_concentration_si(curve, **filter))]

        k_f = Sh * np.array(D_Li_in_H2O_list) / dp

        return k_f
    
    def non_dim_numbers(self, epsilon: float = None, C_0: float = None) -> tuple[list[float], list[float]]:
        """
            This function returns 2 different non dimensional numbers to give and order or magnitude 
            for designing the sensitivity analysis:
            
            parameters:
                - epsilon: bed porosity
                - C_0: influent concentration (mol/m^3)
            
            returns:
                - capacity factor (PI1) =  (1-epsilon)*rho_b*q_0(@C_0) / (epsilon*C_0)
                - Damkohler number (PI2) = k_s * q_0(@C_0) * L / u_0
        """
        if C_0 is None:
            # retrieve the C_0 from the breakthrough curves in the study object
            C_0 = self.column_experiments.influent_concentration_si()

        if epsilon is None:
            # retrieve the epsilon from the column properties in the study object
            epsilon = self.column_experiments.column_properties.porosity

        q_0 = self.isotherm.isotherm_fit.q_eq_si(C_0, self.isotherm.T)
        k_s = [k_s.kinetics_params.k2_si for k_s in self.kinetics_experiments]

        capacity_factors = (1-epsilon) / epsilon * self.sorbent_properties.density_si * q_0 / C_0

        damkohler_numbers = k_s * q_0 * self.column_experiments.column_properties.Length_si / self.column_experiments.superficial_velocity_si()

        return capacity_factors, damkohler_numbers
    
    def to_column_parameter(self, L, C_in, u_super) -> ColumnParameters:
        """Assemble the JAX ``ColumnParameters`` for the dimensional AlLDH model.

        Given a design (length L [m], feed concentration C_in [mol/m^3],
        superficial velocity u_super [m/s]) it converts to interstitial velocity,
        picks the kinetics experiment nearest to C_in for k_s, and pulls porosity,
        particle density and the isotherm from this study.
        """
        # u_inter = 4 * Q / (np.pi * D**2) /self.column_experiments.column_properties.porosity
        u_inter = u_super / self.column_experiments.column_properties.porosity

        kinetics_experiment = self.get_kinetics_experiment_from_concentration(C_in)

        return ColumnParameters(
            L=L,
            u_inter=u_inter,
            C_in=C_in,
            k_s=kinetics_experiment.kinetics_params.k2_si,
            rho_p = self.sorbent_properties.density_si,
            epsilon=self.column_experiments.column_properties.porosity,
            isotherm=self.isotherm.isotherm_fit
        )


    @classmethod
    def from_dict(cls, data: dict) -> "Study":
        column_experiments = ColumnExperiments.from_dict(data["ColumnExperiments"])
        sorbent_properties = SorbentProperties.from_dict(data["SorbentProperties"])
        isotherm_units = IsothermUnits.from_dict(data["IsothermUnits"])
        isotherm = Isotherm.from_dict(data["Isotherm"], isotherm_units)
        kinetics_units = KineticsUnits.from_dict(data["KineticsUnits"])
        kinetics_experiments = [
            KineticsExperiment.from_dict(exp_data, kinetics_units)
            for exp_data in data["KineticsExperiments"]
        ]

        return cls(
            column_experiments=column_experiments,
            sorbent_properties=sorbent_properties,
            isotherm=isotherm,
            kinetics_experiments=kinetics_experiments
        )

    @classmethod
    def from_json(cls, file_path: str | Path, study_key: str) -> "Study":
        """
        Load a Study from a JSON file.

        Parameters:
            file_path: Path to the JSON file
            study_key: Key in the JSON file to extract the study data from

        Returns:
            Study instance with data loaded from the JSON file
        """
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)

        study_data = data[study_key]
        return cls.from_dict(study_data)
