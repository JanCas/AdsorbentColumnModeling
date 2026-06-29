"""Shared logging configuration for entry scripts.

Library modules should NOT import this — they only need
``_log = logging.getLogger(__name__)`` and they pick up the config from the
running entry script.

Entry scripts call ``configure_logging()`` once near the top of ``main()``.
The LOG_LEVEL environment variable overrides the level if set
(e.g. ``LOG_LEVEL=DEBUG python -m Sensitivity.run_alldh``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_DEFAULT_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    stream=None,
    silence: Optional[list[str]] = None,
) -> None:
    """Apply a basic logging config suitable for interactive / demo runs.

    Parameters
    ----------
    level :
        Default log level (overridden by the LOG_LEVEL env var if set).
    fmt, datefmt :
        Format strings passed to ``logging.basicConfig``.
    stream :
        Destination stream; defaults to ``sys.stderr``.
    silence :
        Names of noisy third-party loggers to clamp to WARNING (e.g. matplotlib,
        jax, PIL). Always safe to extend.
    """
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        level = env_level.upper()

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=stream if stream is not None else sys.stderr,
        force=True,
    )

    default_silence = ["matplotlib", "PIL", "fontTools", "jax._src"]
    for name in default_silence + list(silence or []):
        logging.getLogger(name).setLevel(logging.WARNING)
