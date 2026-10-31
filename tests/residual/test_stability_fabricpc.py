"""Anchor and HEP-nudge energies on FabricPC nodes against plain JAX (the oracle)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fabricpc.nodes import IdentityNode, Linear

from residual import fpc_energy
from residual.a2_losses import anchor_loss, hep_nudge_loss, output_error
from residual.fpc_energy import BACKENDS, anchor_sq_distance, highway_sq_norm
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


def test_anchor_energy_gradient_is_two_times_the_difference() -> None:
    z, a = _x(3, 5, D, seed=3), _x(3, 5, D, seed=4)
    g = jax.grad(lambda z: anchor_sq_distance(z, a).sum())(z)
    np.testing.assert_allclose(g, 2.0 * (z - a), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["dense", "lowrank"])
def test_highway_energy_is_the_squared_norm_of_the_nudge(kind) -> None:
    eps = _x(3, 5, V, seed=5)
    p = init_hep(kind, 0, [3], V, D, rank=R)["hep_3"]
    v = p["V"] if kind == "dense" else (p["U"], p["W"])
    ref = (eps @ p["V"]) if kind == "dense" else (eps @ p["U"] @ p["W"])
    out = highway_sq_norm(eps, v)
    assert out.shape == (3, 5)
    np.testing.assert_allclose(out, jnp.sum(ref**2, axis=-1), rtol=1e-5)

    def f(p, eps):
        vv = p["V"] if kind == "dense" else (p["U"], p["W"])
        return highway_sq_norm(eps, vv).sum()

    def g(p, eps):
        r = (eps @ p["V"]) if kind == "dense" else (eps @ p["U"] @ p["W"])
        return jnp.sum(r**2)

    for x, y in zip(
        _leaves(jax.grad(f, argnums=(0, 1))(p, eps)),
        _leaves(jax.grad(g, argnums=(0, 1))(p, eps)),
        strict=True,
    ):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)


def test_the_low_rank_highway_is_never_materialised() -> None:
    p = init_hep("lowrank", 0, [3], V, D, rank=R)["hep_3"]
    eps = _x(2, 3, V, seed=6)
    jaxpr = jax.make_jaxpr(lambda eps: highway_sq_norm(eps, (p["U"], p["W"])))(eps)
    shapes = {tuple(var.aval.shape) for eqn in jaxpr.eqns for var in eqn.outvars}
    assert (V, D) not in shapes


def test_energy_is_computed_in_float32() -> None:
    z, a = _x(2, 3, D, seed=7), _x(2, 3, D, seed=8)
    out = anchor_sq_distance(z.astype(jnp.bfloat16), a.astype(jnp.bfloat16))
    assert out.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(out)))
    eps = _x(2, 3, V, seed=9)
    p = init_hep("dense", 0, [3], V, D)["hep_3"]
    assert highway_sq_norm(eps.astype(jnp.bfloat16), p["V"]).dtype == jnp.float32


def test_jit_matches_eager() -> None:
    z, a = _x(3, 5, D, seed=10), _x(3, 5, D, seed=11)
    f = lambda z: anchor_sq_distance(z, a).sum()  # noqa: E731
    np.testing.assert_allclose(jax.jit(f)(z), f(z), rtol=1e-6)


def test_the_graphs_are_fabricpc_nodes_with_the_expected_parameters() -> None:
    """Guards the node names and weight keys the module relies on."""
    from fabricpc.graph import initialize_params

    anchor = fpc_energy._anchor_graph(D)
    assert isinstance(anchor.nodes["anch"], IdentityNode)
    params = initialize_params(anchor, jax.random.PRNGKey(0))
    assert params.nodes["anch"].weights == {} and params.nodes["anch"].biases == {}

    dense = fpc_energy._highway_graph(V, D, None)
    assert isinstance(dense.nodes["hep"], Linear)
    p = initialize_params(dense, jax.random.PRNGKey(0))
    assert (
        set(p.nodes["hep"].weights) == {"eps->hep:in"} and p.nodes["hep"].biases == {}
    )

    low = fpc_energy._highway_graph(V, D, R)
    p = initialize_params(low, jax.random.PRNGKey(0))
    assert set(p.nodes["mid"].weights) == {"eps->mid:in"}
    assert set(p.nodes["hep"].weights) == {"mid->hep:in"}
    assert p.nodes["mid"].weights["eps->mid:in"].shape == (V, R)
    assert p.nodes["hep"].weights["mid->hep:in"].shape == (R, D)
    assert fpc_energy._highway_graph(V, D, R) is low  # built once per shape


def test_malformed_highways_are_rejected() -> None:
    eps = _x(2, 3, V, seed=12)
    with pytest.raises(ValueError, match="matrix"):
        highway_sq_norm(eps, jnp.ones((V, D, 1)))
    with pytest.raises(ValueError, match="low-rank"):
        highway_sq_norm(eps, (jnp.ones((V, R)), jnp.ones((R + 1, D))))


def test_backends_are_listed_and_checked() -> None:
    assert BACKENDS == ("fabricpc", "jax")
    for bad in ("torch", "", None):
        with pytest.raises(ValueError, match="backend"):
            fpc_energy.check_backend(bad)  # type: ignore[arg-type]


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


def test_hep_nudge_on_a_fixed_highway_has_no_gradient_with_either_backend() -> None:
    eps = _x(3, 6, V, seed=17)
    v = _x(V, D, seed=18)  # a constant matrix, not a parameter
    for backend in BACKENDS:
        g = jax.grad(lambda eps, b=backend: hep_nudge_loss(eps, v, backend=b)[0])(eps)
        assert not np.any(g)


def test_zero_gain_gives_a_zero_penalty() -> None:
    eps = _x(3, 6, V, seed=19)
    v = _x(V, D, seed=20)
    for backend in BACKENDS:
        loss, m = hep_nudge_loss(eps, v, 0.0, backend=backend)
        assert float(loss) == 0.0 and float(m["hep_nudge_norm"]) < 1e-3


def test_output_error_feeds_the_highway_energy_like_the_reference() -> None:
    logits = _x(3, 6, V, seed=21)
    labels = (
        jax.random.randint(jax.random.PRNGKey(22), (3, 6), 0, V).at[1, 3:].set(-100)
    )
    eps, mask = output_error(logits, labels)
    v = _x(V, D, seed=23)
    ref = _x(3, 5, D, seed=24)
    a = hep_nudge_loss(eps, v, 1.0, reference=ref, mask=mask, backend="fabricpc")
    b = hep_nudge_loss(eps, v, 1.0, reference=ref, mask=mask, backend="jax")
    np.testing.assert_allclose(a[0], b[0], rtol=1e-5)


@pytest.mark.parametrize("backend, expect", [("fabricpc", True), ("jax", False)])
def test_the_backend_choice_reaches_fabricpc(monkeypatch, backend, expect) -> None:
    calls = []
    real = fpc_energy._per_token_energy

    def spy(*a, **k):
        calls.append(a[-1])
        return real(*a, **k)

    monkeypatch.setattr(fpc_energy, "_per_token_energy", spy)
    z, anchor = _x(3, 6, D, seed=25), _x(3, 6, D, seed=26)
    anchor_loss(z, anchor, backend=backend)
    eps, v = _x(3, 6, V, seed=27), _x(V, D, seed=28)
    hep_nudge_loss(eps, v, backend=backend)
    assert (calls == ["anch", "hep"]) is expect and bool(calls) is expect


@pytest.mark.parametrize("fn", ["anchor", "hep"])
def test_unknown_backend_is_rejected(fn) -> None:
    with pytest.raises(ValueError, match="backend"):
        if fn == "anchor":
            anchor_loss(_x(3, 6, D), _x(3, 6, D, seed=1), backend="torch")
        else:
            hep_nudge_loss(_x(3, 6, V), _x(V, D, seed=1), backend="torch")
