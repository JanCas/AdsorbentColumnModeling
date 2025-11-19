from dataclasses import dataclass
from .Dataclasses import BreakthroughCurve
import uuid

@dataclass
class SimulationResult:
    curve_flowrate: float
    Reynolds_number: float
    Peclet_number_particle: float
    Peclet_number_axial_particle: float
    Peclet_number_axial_column: float
    D_L: float
    Schmidt_number: float
    Sherwood_number: float