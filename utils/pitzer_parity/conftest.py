"""pytest configuration for the Pitzer parity suite.

Puts this folder on ``sys.path`` for the JAX-free oracle helpers and the repository root
on ``sys.path`` for ``utils.pitzer``.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
for _d in (HERE, REPO_ROOT):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


def pytest_addoption(parser):
    parser.addoption(
        "--oracle-backend",
        choices=("matlab", "oct2py"),
        default="matlab",
        help="which cached oracle to compare against (default: matlab, the oracle the "
             "spec names). Both CSVs are committed.",
    )
    parser.addoption(
        "--regenerate-oracle",
        action="store_true",
        help="re-run the oracle before comparing, instead of using the committed CSV. "
             "Requires Octave or MATLAB installed.",
    )
