"""Anchor and HEP-nudge energies on FabricPC nodes against plain JAX (the oracle)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from losses.a2_loss import anchor_loss
from losses.stability_loss import hep_nudge_loss
from residual.fpc_energy import anchor_sq_distance
from residual.hep import init_hep

D, V, R = 16, 40, 4


def _x(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


# --- the energies --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "z_shape, a_shape",
    [
        ((7, D), (7, D)),
        ((3, 5, D), (3, 5, D)),
        ((2, 3, 5, D), (3, 5, D)),
        ((2, 3, 5, D), (2, 3, 5, D)),
    ],
)
def test_anchor_energy_is_the_squared_distance(z_shape, a_shape) -> None:
    z, a = _x(*z_shape, seed=1), _x(*a_shape, seed=2)
    out = anchor_sq_distance(z, a)
    assert out.shape == z_shape[:-1] and out.dtype == jnp.float32
    np.testing.assert_allclose(out, jnp.sum((z - a) ** 2, axis=-1), rtol=1e-6)


# --- the losses, FabricPC against JAX --------------------------------------------------------


def _mask(batch: int = 3, seq: int = 6) -> jax.Array:
    return jnp.arange(seq)[None, :] < jnp.asarray([6, 4, 5][:batch])[:, None]


@pytest.mark.parametrize("normalize", ["relative", "none"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("steps", [False, True])
def test_anchor_loss_backends_agree(normalize, masked, steps) -> None:
    z = _x(2, 3, 6, D, seed=13) if steps else _x(3, 6, D, seed=13)
    anchor = _x(3, 6, D, seed=14)
    mask = _mask() if masked else None

    def run(backend):
        def f(z):
            return anchor_loss(
                z, anchor, normalize=normalize, mask=mask, backend=backend
            )

        (loss, m), g = jax.value_and_grad(f, has_aux=True)(z)
        return loss, m, g

    (l_f, m_f, g_f), (l_j, m_j, g_j) = run("fabricpc"), run("jax")
    np.testing.assert_allclose(l_f, l_j, rtol=1e-6)
    np.testing.assert_allclose(g_f, g_j, rtol=1e-5, atol=1e-7)
    for k in m_j:
        np.testing.assert_allclose(m_f[k], m_j[k], rtol=1e-5)


@pytest.mark.parametrize("kind", ["dense", "lowrank"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("alpha", [1.0, 0.3])
def test_hep_nudge_loss_backends_agree(kind, masked, alpha) -> None:
    eps = _x(3, 6, V, seed=15)
    ref = _x(3, 6, D, seed=16)
    p = init_hep(kind, 0, [3], V, D, rank=R)["hep_3"]
    mask = _mask() if masked else None

    def run(backend):
        def f(p, eps):
            v = p["V"] if kind == "dense" else (p["U"], p["W"])
            return hep_nudge_loss(
                eps, v, alpha, reference=ref, mask=mask, backend=backend
            )

        (loss, m), g = jax.value_and_grad(f, has_aux=True, argnums=(0, 1))(p, eps)
        return loss, m, g

    (l_f, m_f, g_f), (l_j, m_j, g_j) = run("fabricpc"), run("jax")
    np.testing.assert_allclose(l_f, l_j, rtol=1e-5)
    for k in m_j:
        np.testing.assert_allclose(m_f[k], m_j[k], rtol=1e-5)
    for x, y in zip(_leaves(g_f), _leaves(g_j), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)
    assert not np.any(g_f[1])  # eps is stop-gradient in both backends
