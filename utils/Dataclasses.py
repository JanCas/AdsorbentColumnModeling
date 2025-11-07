from dataclasses import dataclass
import numpy as np
from typing import ClassVar
from abc import ABC, abstractmethod
import json
from pathlib import Path
import casadi as ca

Li_MW = 6.94  # g/mol
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
            kinetics_params=kinetics_params,
            kinetics_units=units
        )

@dataclass(frozen=True)
class ColumnParameters:
    Length: float
    Length_units: str
    Diameter: float
    Diameter_units: str
    porosity: float
    flowrate: list[float]
    influent_concentration: float
    flowrate_units: str
    influent_concentration_units: str

    @property
    def superficial_flowrate_si(self) -> float:
        "Converts the flowrate to m3/s depending on the units specified."
        match self.flowrate_units:
            case "BV/h":
                return np.array(self.flowrate) * (np.pi * (self.Diameter / 2) ** 2 * self.Length) / 3600  # Convert to m3/s
            case "m3/s":
                return self.flowrate
            case _:
                raise ValueError(f"Unsupported flowrate units: {self.flowrate_units}")

    @property
    def influent_concentration_si(self) -> float:
        "Converts the influent concentration to mol/m³ depending on the units specified."
        match self.influent_concentration_units:
            case "mg/L":
                return convert_mg_per_L_to_mol_per_m3(self.influent_concentration)
            case "mol/m3":
                return self.influent_concentration
            case _:
                raise ValueError(f"Unsupported influent concentration units: {self.influent_concentration_units}")

    @property
    def superficial_velocity_si(self) -> float:
        "Calculates the superficial velocity in m/s."
        cross_sectional_area = np.pi * (self.Diameter / 2) ** 2
        return self.superficial_flowrate_si / cross_sectional_area
    
    @property
    def interstitial_velocity_si(self) -> float:
        "Calculates the interstitial velocity in m/s."
        return self.superficial_velocity_si / self.porosity

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnParameters":
        return cls(
            Length=data["Length"],
            Length_units=data.get("Length_units", "m"),
            Diameter=data["Diameter"],
            Diameter_units=data.get("Diameter_units", "m"),
            porosity=data["porosity"],
            flowrate=data["flowrate"],
            influent_concentration=data["influent_concentration"],
            flowrate_units=data["flowrate_units"],
            influent_concentration_units=data["influent_concentration_units"]
        )
    
@dataclass(frozen=True)
class SorbentProperties:
    density: float
    density_units: str

    @classmethod
    def from_dict(cls, data: dict) -> "SorbentProperties":
        return cls(
            density=data["density"],
            density_units=data["density_units"]
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
class Isotherm:
    isotherm_fit: BaseIsothermFit
    T: float
    equilibrium_concentration_units: str
    equilibrium_concentration: list[float]
    fit_type: str
    equilibrium_uptake: list[float]
    equilibrium_uptake_units: str

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
    def from_dict(cls, data: dict) -> "Isotherm":
        match data["FitType"]:
            case "Temkin":
                isotherm_fit = TemkinIsothermFit.from_dict(data["FitParameters"])
            case _:
                raise ValueError(f"Unsupported FitType: {data['FitType']}")

        return cls(
            isotherm_fit=isotherm_fit,
            T=data["T"],
            equilibrium_concentration_units=data["C_e_units"],
            equilibrium_concentration=data["C_e"],
            fit_type=data["FitType"],
            equilibrium_uptake=data["q_e"],
            equilibrium_uptake_units=data["q_e_units"]
        )

@dataclass(frozen=True)
class BreakthroughCurve:
    flowrate: int
    flowrate_units: str
    BV: list[float]
    C_out_over_C_in: list[float]

    @classmethod
    def from_dict(cls, data: dict) -> "BreakthroughCurve":
        return cls(
            flowrate=data["flowrate"],
            flowrate_units=data["flowrate_units"],
            BV=data["BV"],
            C_out_over_C_in=data["C_out/C_in"]
        )

@dataclass(frozen=True)
class Study:
    column_parameters: ColumnParameters
    sorbent_properties: SorbentProperties
    isotherm: Isotherm
    kinetics_experiments: list[KineticsExperiment]
    breakthrough_curves: list[BreakthroughCurve]

    @classmethod
    def from_dict(cls, data: dict) -> "Study":
        column_parameters = ColumnParameters.from_dict(data["ColumnProperties"])
        sorbent_properties = SorbentProperties.from_dict(data["SorbentProperties"])
        isotherm = Isotherm.from_dict(data["Isotherm"])
        kinetics_units = KineticsUnits.from_dict(data["KineticsUnits"])
        kinetics_experiments = [
            KineticsExperiment.from_dict(exp_data, kinetics_units)
            for exp_data in data["KineticsExperiments"]
        ]
        breakthrough_curves = [
            BreakthroughCurve.from_dict(bc_data)
            for bc_data in data["ColumnBreakthroughData"]
        ]

        return cls(
            column_parameters=column_parameters,
            sorbent_properties=sorbent_properties,
            isotherm=isotherm,
            kinetics_experiments=kinetics_experiments,
            breakthrough_curves=breakthrough_curves
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