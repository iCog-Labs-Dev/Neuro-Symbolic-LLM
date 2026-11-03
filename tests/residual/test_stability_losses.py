"""Essential A3 contracts: fixed references and invariant reduction strength."""

import jax
import jax.numpy as jnp
import numpy as np

from metrics.representation import settling_loss


def test_settling_reduction_averages_steps_dimensions_sites_and_valid_tokens():
    reference = jnp.ones((1, 3, 2))
    mask = jnp.array([[True, True, False]])
    states = jnp.stack((reference, reference + 2))
    states = states.at[:, :, 2].set(1000)
    one, _ = settling_loss({1: states}, {1: reference}, mask=mask)
    repeated = jnp.concatenate((states[:1], jnp.repeat(states[1:], 4, axis=0)))
    repeated = jnp.repeat(repeated, 3, axis=-1)
    ref = jnp.repeat(reference, 3, axis=-1)
    many, _ = settling_loss({1: repeated, 2: repeated}, {1: ref, 2: ref}, mask=mask)
    np.testing.assert_allclose(one, 4.0)
    np.testing.assert_allclose(many, one)


def test_settling_reference_is_detached_but_states_and_nudges_are_live():
    reference = jnp.ones((1, 2, 3))
    states = jnp.stack((reference, reference + 2))
    nudges = jnp.ones_like(states[1:])

    def loss(z, ref, nudge):
        return settling_loss(
            {1: z},
            {1: ref},
            nudges={1: nudge},
            anchor_weight=1.0,
            step_weight=0.5,
            nudge_weight=0.1,
        )[0]

    gz, gr, gn = jax.grad(loss, argnums=(0, 1, 2))(states, reference, nudges)
    np.testing.assert_array_equal(gr, jnp.zeros_like(reference))
    assert bool(jnp.any(gz != 0)) and bool(jnp.any(gn != 0))
    zero, _ = settling_loss({1: states[:1]}, {1: reference})
    assert float(zero) == 0.0
