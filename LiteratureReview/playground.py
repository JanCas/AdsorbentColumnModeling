import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.Dataclasses import Study

script_dir = Path(__file__).parent

if __name__ == "__main__":
    s = Study.from_json(f"{script_dir}/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2019")
    print(s.to_column_parameter(10, .5, 50, 10))
