from ..utils.Dataclasses import Study
from pathlib import Path

script_dir = Path(__file__).parent

if __name__ == "__main__":
    s = Study.from_json(f"{script_dir}/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")
    print(s.non_dim_numbers())