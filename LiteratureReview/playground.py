"""Scratch script for inspecting literature-derived isotherm/kinetics parameters.

Adds the repo root to sys.path so ``utils.Dataclasses`` is importable, loads one
literature Study (isotherm + kinetics fits) from ``isotherm_kinetics.json`` by its
citation key, converts it to column-scale parameters, and prints them. Handy for
sanity-checking that a paper's fitted constants map to sensible packed-bed inputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.Dataclasses import Study

script_dir = Path(__file__).parent

if __name__ == "__main__":
    # Load the Jiang et al. (2020) Li adsorption study and translate its
    # isotherm/kinetics fit into column-scale parameters (args are the column
    # operating conditions passed to Study.to_column_parameter).
    s = Study.from_json(f"{script_dir}/isotherm_kinetics.json", "jiangAdsorptionLithiumIons2020")
    print(s.to_column_parameter(10, .5, 50, 10))