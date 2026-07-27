"""pytest configuration for the Pitzer parity suite.

Puts this folder and the model folder above it on sys.path, so `import cases` and
`import pitzer` both resolve flat — matching the rest of the repo's sibling-import habit
(`from ix_model import ...` in Model/IX/dim).
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
for _d in (HERE, HERE.parent):
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
