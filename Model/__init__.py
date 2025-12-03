# Model package initialization

# Original explicit Euler implementation
from .non_dim_model import (
    n_eq_star_temkin,
    simulate_column_temkin_star
)

# Diffrax-based implementation
try:
    from .non_dim_model_diffrax import (
        n_eq_star_temkin_jax,
        simulate_column_temkin_star_diffrax,
        simulate_with_event_detection
    )
    DIFFRAX_AVAILABLE = True
except ImportError:
    DIFFRAX_AVAILABLE = False
    print("Warning: diffrax not available. Install with: pip install diffrax")

__all__ = [
    "n_eq_star_temkin",
    "simulate_column_temkin_star",
    "n_eq_star_temkin_jax",
    "simulate_column_temkin_star_diffrax",
    "simulate_with_event_detection",
    "DIFFRAX_AVAILABLE"
]