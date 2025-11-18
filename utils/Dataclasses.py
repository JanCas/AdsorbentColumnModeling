from dataclasses import dataclass, asdict
import numpy as np
from typing import ClassVar
from abc import ABC, abstractmethod
import json
from pathlib import Path
import casadi as ca
from thermo import IAPWS95Liquid as Water

Li_MW = 6.94  # g/mol
D_Li_in_H2O = 1.03e-9  # m²/s at 25°C
ideal_gas_constant = 8.314  # J/(mol·K)

def convert_mg_per_L_to_mol_per_m3(concentration_mg_per_L: float | np.ndarray) -> float | np.ndarray:
    "Converts concentration from mg/L to mol/m³."
    return concentration_mg_per_L / Li_MW  # Convert to mol/m³

def convert_mg_per_g_to_mol_per_kg(q_mg_per_g: float | np.ndarray) -> float | np.ndarray:
    "Converts adsorption capacity from mg/g to mol/kg."
    return q_mg_per_g / Li_MW  # Convert to mol/kg

@dataclass(frozen=True)
class PseudoSecondOrderKineticsParameters:
    k2: float
    q_e: float
    k2_units: str
    q_e_units: str

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
    time: str
    q_t: str
    T: str
    C_e: str

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
    T: float
    C_e: float
    Ph: float
    time: list[float]
    q_t: list[float]
    kinetics_params: PseudoSecondOrderKineticsParameters
    kinetics_units: KineticsUnits

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
    Length: float
    Length_units: str
    Diameter: float
    Diameter_units: str
    porosity: float

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
    flowrate: str
    influent_concentration: str
    T: str

    @classmethod
    def from_dict(cls, data: dict) -> "BreakthroughCurveUnits":
        return cls(
            flowrate=data["flowrate"],
            influent_concentration=data["influent_concentration"],
            T=data["T"]
        )

@dataclass(frozen=True)
class BreakthroughCurve:
    flowrate: int
    T: float
    PH: float
    influent_concentration: float
    BV: list[float]
    C_out_over_C_in: list[float]

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
    curves: tuple[BreakthroughCurve, ...]
    units: BreakthroughCurveUnits

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
    column_properties: ColumnProperties
    breakthrough_curves: BreakthroughCurves

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
    density: float
    density_units: str
    particle_diameter: float
    particle_diameter_units: str

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

@dataclass(frozen=True)
class BaseIsothermFit(ABC):
    """
    Abstract base class for isotherm fit models.
    """

    fit_type: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseIsothermFit":
        ...

    @abstractmethod
    def q_eq_si(self, C_eq_si: float) -> float:
        print("Here")
        ...

@dataclass(frozen=True)
class SipsIsothermFit(BaseIsothermFit):
    fit_type: ClassVar[str] = "Sips"
    K_s: float
    n: float
    Q_max: float
    K_s_units: str
    Q_max_units: str

    def __post_init__(self):
        Warning.warn("Sips Isotherm model is not yet implemented.")

    @property
    def K_s_si(self) -> float:
        ...

@dataclass(frozen=True)
class TemkinIsothermFit(BaseIsothermFit):
    fit_type: ClassVar[str] = "Temkin"
    A: float
    B: float
    A_units: str
    B_units: str

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

    def q_eq_si(self, C_eq_si: float, T: float) -> float:
        """
        Calculates the equilibrium uptake (q_eq) in mol/kg using the Temkin isotherm model.
        C_eq_si: Equilibrium concentration in mol/m³
        T: Temperature in K
        Returns q_eq in mol/kg
        """
        interm = ideal_gas_constant * T / self.B_si
        return ca.fmax(interm * ca.log(self.A_si) + interm * ca.log(C_eq_si), 0)

    @classmethod
    def from_dict(cls, data: dict) -> "TemkinIsothermFit":
        return cls(
            A=data["A"],
            B=data["B"],
            A_units=data["A_units"],
            B_units=data["B_units"]
        )


IsothermFitDirectory = {
    TemkinIsothermFit.fit_type: TemkinIsothermFit
}  

@dataclass(frozen=True)
class IsothermUnits:
    C_e: str
    q_e: str
    T: str

    @classmethod
    def from_dict(cls, data: dict) -> "IsothermUnits":
        return cls(
            C_e=data["C_e"],
            q_e=data["q_e"],
            T=data["T"],
        )

@dataclass(frozen=True)
class Isotherm:
    isotherm_fit: BaseIsothermFit
    T: float
    Ph: float
    equilibrium_concentration_units: str
    equilibrium_concentration: list[float]
    fit_type: str
    equilibrium_uptake: list[float]
    equilibrium_uptake_units: str
    units: IsothermUnits

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
                isotherm_fit = TemkinIsothermFit.from_dict(data["FitParameters"])
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
    column_experiments: ColumnExperiments
    sorbent_properties: SorbentProperties
    isotherm: Isotherm
    kinetics_experiments: list[KineticsExperiment]

    def get_kinetics_experiment_from_curve(self, curve: BreakthroughCurve) -> KineticsExperiment:
        "Returns the kinetics experiment that matches the temperature and pH of the given breakthrough curve."
        for exp in self.kinetics_experiments:
            if (exp.T == curve.T) and (exp.Ph == curve.PH) and (exp.C_e_si == self.column_experiments.influent_concentration_si(**asdict(curve))[0]):
                return exp
        raise ValueError("No matching kinetics experiment found for the given breakthrough curve.")

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

        return velocities * self.sorbent_properties.particle_diameter_si / D_Li_in_H2O

    def schmidt_number(self,curve: BreakthroughCurve = None,T: float = None) -> float:
        "Calculates the schmidt number at temperature T (K)."
        if curve is not None:
            T = curve.T
        water = Water(T=T)
        mu = water.mu()  # Dynamic viscosity in Pa.s
        rho = water.rho_mass()  # Density in kg/m³

        return mu / (rho * D_Li_in_H2O)
        

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