"""Tests for the CPC heads (``residual/cpc_heads.py``).

It tests the initialisation, keys, shapes and the reference maths.
and the FabricPC-node backend against the pure-JAX reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fabricpc.graph import initialize_params
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
    split_cpc,
)

D, M = 24, 8


def test_keys_and_shapes() -> None:
    params = init_cpc_heads(0, [7, 9], D, M, offsets=(1, 2))
    assert set(params) == {
        q_key(7),
        q_key(9),
        a_key(7, 1),
        a_key(7, 2),
        a_key(9, 1),
        a_key(9, 2),
    }
    assert q_key(7) == "cpcq_7" and a_key(7, 2) == "cpca_7_2"
    for layer in (7, 9):
        assert params[q_key(layer)]["W"].shape == (D, M)
        for k in (1, 2):
            assert params[a_key(layer, k)]["M"].shape == (M, M)
    # Two-level layout required by RunWriter.save_params.
    assert all(set(v) <= {"W", "M"} for v in params.values())
    assert all(v.dtype == jnp.float32 for p in params.values() for v in p.values())


def test_layers_are_deduplicated_and_order_independent() -> None:
    a = init_cpc_heads(0, [9, 7, 7], D, M)
    b = init_cpc_heads(0, [7, 9], D, M)
    assert set(a) == set(b)
    for key in a:
        for name in a[key]:
            np.testing.assert_array_equal(a[key][name], b[key][name])


def test_deterministic_in_seed() -> None:
    a, b, c = (init_cpc_heads(s, [3], D, M) for s in (0, 0, 1))
    np.testing.assert_array_equal(a["cpcq_3"]["W"], b["cpcq_3"]["W"])
    assert not np.array_equal(a["cpcq_3"]["W"], c["cpcq_3"]["W"])


def test_q_is_random_and_has_unit_scale() -> None:
    w = np.asarray(init_cpc_heads(0, [3], 256, 64)["cpcq_3"]["W"])
    assert np.abs(w).max() > 0
    # N(0, 1/d): each column has norm ~ sqrt(m/d)... check the entry std instead.
    assert w.std() == pytest.approx(1 / np.sqrt(256), rel=0.1)


def test_a_is_identity_plus_noise_never_zero_and_differs_per_offset() -> None:
    p = init_cpc_heads(0, [3], D, M, offsets=(1, 2), a_noise_std=0.01)
    a1, a2 = (np.asarray(p[a_key(3, k)]["M"]) for k in (1, 2))
    assert np.abs(a1).max() > 0
    np.testing.assert_allclose(a1, np.eye(M), atol=0.1)
    assert not np.array_equal(a1, np.eye(M))  # noise present
    assert not np.array_equal(a1, a2)  # offsets not tied


def test_zero_noise_gives_exact_identity() -> None:
    p = init_cpc_heads(0, [3], D, M, a_noise_std=0.0)
    np.testing.assert_array_equal(p[a_key(3, 1)]["M"], np.eye(M))


def test_layers_do_not_share_weights() -> None:
    p = init_cpc_heads(0, [3, 4], D, M)
    assert not np.array_equal(p["cpcq_3"]["W"], p["cpcq_4"]["W"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"dim": 0},
        {"a_noise_std": -1.0},
        {"offsets": ()},
        {"offsets": (0,)},
        {"offsets": (True,)},
    ],
)
def test_init_validation(kwargs: dict) -> None:
    args = {"hidden_size": D, "dim": M} | kwargs
    with pytest.raises(ValueError):
        init_cpc_heads(0, [3], **args)


def test_project_shape_dtype_and_layernorm_invariance() -> None:
    q = init_cpc_heads(0, [3], D, M)["cpcq_3"]
    h = jax.random.normal(jax.random.PRNGKey(0), (2, 5, D))
    c = project(q, h)
    assert c.shape == (2, 5, M) and c.dtype == jnp.float32
    # LayerNorm: invariant to a positive rescale and a shift of each state.
    np.testing.assert_allclose(project(q, 3.0 * h + 5.0), c, atol=2e-3)
    # float32 output for bf16 input.
    assert project(q, h.astype(jnp.bfloat16)).dtype == jnp.float32


def test_project_is_linear_after_layernorm() -> None:
    q = init_cpc_heads(0, [3], D, M)["cpcq_3"]
    h = jax.random.normal(jax.random.PRNGKey(1), (4, D))
    expected = (
        (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5)
    ) @ q["W"]
    np.testing.assert_allclose(project(q, h), expected, atol=1e-5)


def test_predict_future_is_a_matmul() -> None:
    p = init_cpc_heads(0, [3], D, M)
    c = jax.random.normal(jax.random.PRNGKey(2), (2, 5, M))
    out = predict_future(p[a_key(3, 1)], c)
    assert out.shape == c.shape
    np.testing.assert_allclose(out, c @ p[a_key(3, 1)]["M"], atol=1e-6)


def test_split_cpc_selects_only_head_entries() -> None:
    tree = {
        "layer_3": {"A": jnp.ones((2, 2)), "B": jnp.ones((2, 2))},
        "pred_3": {"W": jnp.ones((2, 2))},
        **init_cpc_heads(0, [3], D, M),
    }
    assert set(split_cpc(tree)) == {"cpcq_3", "cpca_3_1"}
    assert split_cpc({"layer_3": tree["layer_3"]}) == {}


# ----------------------------------------------------------------------
# FabricPC backend against the pure-JAX reference
# ----------------------------------------------------------------------


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
    np.testing.assert_allclose(out, project_reference(q, h), atol=1e-5)
    np.testing.assert_allclose(out, project(q, h, "jax"), atol=1e-5)


@pytest.mark.parametrize("shape", [(7, M), (3, 5, M), (2, 3, 5, M)])
def test_predictor_matches_the_reference(shape) -> None:
    _, a = _heads()
    c = _x(*shape, seed=2)
    out = predict_future(a, c)
    assert out.shape == shape
    np.testing.assert_allclose(out, predict_future_reference(a, c), atol=1e-5)


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
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)


def test_jit_matches_eager_and_does_not_retrace_per_call() -> None:
    q, a = _heads()
    h = _x(2, 6, D, seed=5)
    traces = []

    def f(q, a, h):
        traces.append(1)
        return predict_future(a, project(q, h))

    jf = jax.jit(f)
    np.testing.assert_allclose(jf(q, a, h), f(q, a, h), atol=1e-5)
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
