from dataclasses import dataclass
from .Dataclasses import BreakthroughCurve
import uuid

@dataclass
class SimulationResult:
    curve_uuid: uuid.UUID
    Reynolds_number: float
    Peclet_number: float
    Schmidt_number: float
