"""The CPC heads ``q`` and ``A`` as FabricPC nodes against the pure-JAX reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import Linear

from residual import cpc_heads
from residual.cpc_heads import (
    BACKENDS,
    a_key,
    init_cpc_heads,
    predict_future,
    predict_future_reference,
    project,
    project_reference,
    q_key,
)

D, M = 16, 8


def _heads(seed: int = 0):
    p = init_cpc_heads(seed, [3], D, M, offsets=(1,), a_noise_std=0.1)
    return p[q_key(3)], p[a_key(3, 1)]


def _x(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


def test_backends_are_listed() -> None:
    assert BACKENDS == ("fabricpc", "jax")


@pytest.mark.parametrize("shape", [(7, D), (3, 5, D), (2, 3, 5, D)])
def test_projection_matches_the_reference(shape) -> None:
    q, _ = _heads()
    h = _x(*shape, seed=1)
    out = project(q, h)
    assert out.shape == (*shape[:-1], M) and out.dtype == jnp.float32
    np.testing.assert_allclose(out, project_reference(q, h), atol=1e-6)
    np.testing.assert_allclose(out, project(q, h, "jax"), atol=1e-6)


@pytest.mark.parametrize("shape", [(7, M), (3, 5, M), (2, 3, 5, M)])
def test_predictor_matches_the_reference(shape) -> None:
    _, a = _heads()
    c = _x(*shape, seed=2)
    out = predict_future(a, c)
    assert out.shape == shape
    np.testing.assert_allclose(out, predict_future_reference(a, c), atol=1e-6)


def test_gradients_match_the_reference_for_weights_and_input() -> None:
    q, a = _heads()
    h, w = _x(3, 5, D, seed=3), _x(3, 5, M, seed=4)

    def loss(backend):
        def f(q, a, h):
            z = predict_future(a, project(q, h, backend), backend)
            return jnp.sum(z * w)

        return f

    g_f = jax.grad(loss("fabricpc"), argnums=(0, 1, 2))(q, a, h)
    g_j = jax.grad(loss("jax"), argnums=(0, 1, 2))(q, a, h)
    for x, y in zip(
        jax.tree_util.tree_leaves(g_f), jax.tree_util.tree_leaves(g_j), strict=True
    ):
        assert float(jnp.abs(y).max()) > 0
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def test_jit_matches_eager_and_does_not_retrace_per_call() -> None:
    q, a = _heads()
    h = _x(2, 6, D, seed=5)
    traces = []

    def f(q, a, h):
        traces.append(1)
        return predict_future(a, project(q, h))

    jf = jax.jit(f)
    np.testing.assert_allclose(jf(q, a, h), f(q, a, h), atol=1e-6)
    n = len(traces)
    jf(q, a, h + 1.0)
    assert len(traces) == n  # same shapes -> compiled once


def test_low_precision_input_is_computed_in_float32() -> None:
    q, a = _heads()
    h = _x(2, 4, D, seed=6)
    out = predict_future(a, project(q, h.astype(jnp.bfloat16)))
    assert out.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(out, predict_future(a, project(q, h)), atol=0.1)


def test_layer_norm_makes_the_projection_scale_and_shift_invariant() -> None:
    q, _ = _heads()
    h = _x(4, D, seed=7)
    np.testing.assert_allclose(project(q, 3.0 * h + 5.0), project(q, h), atol=2e-3)


def test_the_head_is_a_bias_free_linear_node_with_the_expected_edge_key() -> None:
    """Guards the weight-key convention the bridge relies on."""
    for src, dst, edge, d_in, d_out in (
        ("x", "q", "x->q:in", D, M),
        ("c", "A", "c->A:in", M, M),
    ):
        structure = cpc_heads._linear_head(src, dst, d_in, d_out)
        assert isinstance(structure.nodes[dst], Linear)  # a FabricPC Linear node
        assert structure.nodes[dst].shape == (d_out,)
        params = initialize_params(structure, jax.random.PRNGKey(0))
        assert set(params.nodes) == {src, dst}
        assert set(params.nodes[dst].weights) == {edge}
        assert params.nodes[dst].weights[edge].shape == (d_in, d_out)
        assert params.nodes[dst].biases == {}  # bias-free, like the reference
        assert params.nodes[src].weights == {}


def test_the_graphs_are_built_once_per_shape() -> None:
    assert cpc_heads._linear_head("x", "q", D, M) is cpc_heads._linear_head(
        "x", "q", D, M
    )
    assert cpc_heads._linear_head("x", "q", D, M) is not cpc_heads._linear_head(
        "x", "q", D, M + 1
    )


def test_stored_parameters_stay_plain_arrays_in_the_flat_tree() -> None:
    p = init_cpc_heads(0, [3, 4], D, M, offsets=(1, 2))
    assert all(isinstance(v, dict) for v in p.values())
    assert all(isinstance(w, jax.Array) for v in p.values() for w in v.values())
    assert set(p) == {
        q_key(3),
        q_key(4),
        a_key(3, 1),
        a_key(3, 2),
        a_key(4, 1),
        a_key(4, 2),
    }


def test_a_zero_predictor_gives_a_zero_prediction() -> None:
    _, a = _heads()
    zero = {"M": jnp.zeros_like(a["M"])}
    assert not np.any(predict_future(zero, _x(3, M, seed=8)))


@pytest.mark.parametrize("bad", ["torch", "", None])
def test_unknown_backend_is_rejected(bad) -> None:
    q, a = _heads()
    with pytest.raises(ValueError, match="backend"):
        project(q, _x(2, D), bad)
    with pytest.raises(ValueError, match="backend"):
        predict_future(a, _x(2, M), bad)


def test_non_matrix_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="matrix"):
        project({"W": jnp.ones((D, M, 1))}, _x(2, D))


def test_the_fabricpc_backend_really_runs_the_heads_through_fabricpc(
    monkeypatch,
) -> None:
    """Equal values cannot tell the backends apart, so count the node calls."""
    calls = []
    real = cpc_heads._run_head

    def spy(*args, **kwargs):
        calls.append(args[1:3])  # (source name, head name)
        return real(*args, **kwargs)

    monkeypatch.setattr(cpc_heads, "_run_head", spy)
    q, a = _heads()
    h = _x(2, 3, D, seed=9)
    predict_future(a, project(q, h))
    assert calls == [("x", "q"), ("c", "A")]
    calls.clear()
    predict_future(a, project(q, h, "jax"), "jax")
    assert calls == []