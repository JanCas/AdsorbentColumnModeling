"""Adsorption-model packages.

The legacy root-level modules referenced by older revisions are optional. They
are not present in the current repository layout, so importing ``Model`` must
not prevent access to maintained subpackages such as ``Model.AlLDH``.
"""

__all__: list[str] = []

try:
    from .non_dim_model import n_eq_star_temkin, simulate_column_temkin_star
except ImportError:
    pass
else:
    __all__ += ["n_eq_star_temkin", "simulate_column_temkin_star"]

try:
    from .non_dim_model_diffrax import (
        n_eq_star_temkin_jax,
        simulate_column_temkin_star_diffrax,
        simulate_with_event_detection,
    )
except ImportError:
    DIFFRAX_AVAILABLE = False
else:
    DIFFRAX_AVAILABLE = True
    __all__ += [
        "n_eq_star_temkin_jax",
        "simulate_column_temkin_star_diffrax",
        "simulate_with_event_detection",
    ]

__all__.append("DIFFRAX_AVAILABLE")
