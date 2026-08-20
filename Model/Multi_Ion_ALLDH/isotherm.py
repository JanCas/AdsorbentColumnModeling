"""Activity-based equilibrium isotherm for LiCl insertion into ALLDH."""

import equinox as eqx
import jax
import jax.numpy as jnp

from utils.StreamData import Composition


def log_reaction_activity(composition: Composition, n_h2o):
    """Return ln(A_aq) for A_aq = a_LiCl * a_water**n_h2o.

    ``n_h2o`` is the *net* liquid-water stoichiometric coefficient between
    the lower and upper solid endpoints per inserted LiCl, not the hydration
    number of either endpoint.

    ``composition`` contains finite, nonnegative scalar molalities in mol/kg
    water; use ``jax.vmap`` for batches. The Pitzer model fixes temperature at
    298.15 K and supports the LiCl/NaCl/MgCl2 system only.
    """
    n_h2o = jnp.asarray(n_h2o, dtype=jnp.float64)
    log_a_LiCl, log_a_water = composition.log_activities()

    # ponytail: the exact a_LiCl=0 boundary makes its derivative zero; revisit if a
    # future transport model needs the one-sided dilute-limit derivative.
    return log_a_LiCl + n_h2o * log_a_water


def equilibrium_occupancy(log_A_aq, log_K_i):
    """Return theta = K_i A_aq / (1 + K_i A_aq) in stable log space."""
    return jax.nn.sigmoid(
        jnp.asarray(log_K_i, dtype=jnp.float64)
        + jnp.asarray(log_A_aq, dtype=jnp.float64)
    )


def equilibrium_loading(
    composition: Composition, q_max, log_K_i, n_h2o
):
    """Return reversible Li loading above the regenerated solid state.

    ``q_max`` is the reversible loading span.
    """
    log_A_aq = log_reaction_activity(composition, n_h2o)
    return jnp.asarray(q_max, dtype=jnp.float64) * equilibrium_occupancy(
        log_A_aq, log_K_i
    )


class ALLDHIsotherm(eqx.Module):
    """Reversible uptake measured relative to the lower solid endpoint.

    ``q_max`` is the working loading span and ``n_h2o`` is the net endpoint
    hydration change, not an absolute hydrate number.
    """

    q_max: jax.Array = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )
    log_K_i: jax.Array = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )
    n_h2o: jax.Array = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )

    def __call__(self, composition: Composition):
        return equilibrium_loading(
            composition,
            self.q_max,
            self.log_K_i,
            self.n_h2o,
        )
