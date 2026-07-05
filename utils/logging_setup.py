"""Shared logging configuration for entry scripts.

Logging philosophy for this project
------------------------------------
All runtime logging lives in the *orchestration layer* — the Sobol driver,
the NSGA-II optimizer, and their entry scripts — using stdlib ``logging``.
The forward column models are wrapped in ``@eqx.filter_jit`` (JAX JIT). A hard
rule of this codebase: NEVER put a Python ``print`` or ``logging`` call inside a
JIT-compiled body. Such a call fires once at *trace* time (when JAX builds the
computation graph), not per run, so it silently misreports what happened.
Inside a jitted body use ``jax.debug.print`` instead (typically behind a flag).

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


# Compact line format: HH:MM:SS, fixed-width level, source logger name, message.
# The short clock-only timestamp keeps interactive/demo output readable during
# long Sobol sweeps where hundreds of INFO lines scroll by.
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
    # An explicit LOG_LEVEL env var always wins over the caller-supplied level,
    # so verbosity can be bumped without editing code.
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        level = env_level.upper()

    # force=True tears down any handlers a library may have installed on import,
    # guaranteeing our single stderr handler / format is the one in effect.
    # Logs go to stderr so stdout stays clean for piped data / results.
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=stream if stream is not None else sys.stderr,
        force=True,
    )

    # Clamp chatty third-party loggers to WARNING so their DEBUG/INFO noise does
    # not drown the project's own progress messages during a sweep.
    default_silence = ["matplotlib", "PIL", "fontTools", "jax._src"]
    for name in default_silence + list(silence or []):
        logging.getLogger(name).setLevel(logging.WARNING)
