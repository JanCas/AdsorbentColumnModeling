"""JAX-compatible stream data for LiCl/NaCl/MgCl2 brines."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .pitzer import activities_of, log_activities_of


def _as_float64(value):
    return jnp.asarray(value, dtype=jnp.float64)


class Composition(eqx.Module):
    """Finite, nonnegative scalar molalities.

    Pitzer activities are fixed at 298.15 K.
    """

    m_LiCl: jax.Array = eqx.field(converter=_as_float64)
    m_NaCl: jax.Array = eqx.field(converter=_as_float64)
    m_MgCl2: jax.Array = eqx.field(converter=_as_float64)

    def log_activities(self):
        """Return ``(ln(a_LiCl), ln(a_water))`` in stable log space."""
        return log_activities_of(self.m_LiCl, self.m_NaCl, self.m_MgCl2)

    def activities(self):
        """Return ``(a_LiCl, a_water)`` as dimensionless activities."""
        return activities_of(self.m_LiCl, self.m_NaCl, self.m_MgCl2)

    def li_cl_activity(self):
        """Return the neutral LiCl component activity."""
        return self.activities()[0]

    def water_activity(self):
        """Return water activity."""
        return self.activities()[1]


class Stream(eqx.Module):
    """Brine plus metadata.

    ``temp_K`` does not alter the fixed-temperature Pitzer model.
    """

    comp: Composition
    pH: jax.Array = eqx.field(converter=_as_float64)
    temp_K: jax.Array = eqx.field(converter=_as_float64)
