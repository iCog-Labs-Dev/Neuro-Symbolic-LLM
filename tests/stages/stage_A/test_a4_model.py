"""A4 layer constraints and disabled-head equivalence with A3 inference."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("fabricpc.graph_assembly", reason="A3 requires FabricPC >= 0.6")

from residual.symbolic_head.jax_head import SymbolicHead  # noqa: E402
from stages.results import RunWriter, load_params  # noqa: E402
from stages.stage_A.A3_model import A3Config, A3Model  # noqa: E402
from stages.stage_A.a3_tail import residual_modify  # noqa: E402
from stages.stage_A.a4.checkpoint import restore_a3_weights  # noqa: E402
from stages.stage_A.a4.model import A4Model  # noqa: E402
from stages.stage_A.a4.objective import (  # noqa: E402
    AlignmentLabels,
    _task_terms_at_latents,
    objective_at_latents,
)
from symbolic_memory.jax_bridge import RetrievedTemplates  # noqa: E402
from tests.stages.stage_A.a4_fixtures import small_substrate  # noqa: E402


def forbidden_retrieval(queries):
    raise AssertionError("Retrieval must not run in this disabled-head check")


@pytest.fixture(scope="module")
def env():
    sub = small_substrate()
    continuous = A3Model(
        sub,
        layers=(6, 7),
        rank=4,
        seq_len=5,
        config=A3Config(infer_steps=1, eta_z=0.05),
    )
    params = continuous.init_params(jax.random.PRNGKey(0))
    ids = jnp.asarray([[1, 4, 9, 3, 2]], dtype=jnp.int32)
    _, aux = continuous.train_grads(params, ids, jax.random.PRNGKey(1))
    head = SymbolicHead(32, 256, (7,), forbidden_retrieval)
    return SimpleNamespace(
        sub=sub,
        continuous=continuous,
        params=params,
        ids=ids,
        z=aux["z_star"],
        head=head,
        a4=A4Model(continuous, head),
    )


def test_disabled_head_matches_existing_a3_forward_under_jit(env):
    def reference(params, ids, z):
        projections = {
            layer: env.continuous.projection(params, layer)
            for layer in env.continuous.layers
        }
        return env.sub.run_with_interception(
            ids,
            modify_fn=residual_modify(z, projections, env.continuous.scale),
            intercept_layers=env.continuous.layers,
        )

    # The disabled path accepts an empty symbolic parameter mapping.
    actual = jax.jit(
        lambda params, ids, z: env.a4.forward_at_latents(
            params, {}, ids, z, symbolic_enabled=False
        )
    )(env.params, env.ids, env.z)
    expected = jax.jit(reference)(env.params, env.ids, env.z)
    np.testing.assert_array_equal(actual.logits, expected.logits)
    for layer in env.continuous.layers:
        np.testing.assert_array_equal(
            actual.hidden_state(layer), expected.hidden_state(layer)
        )
    assert env.sub.verify_frozen()["params_unchanged"]


def test_symbolic_layer_subset_is_retained(env):
    assert env.a4.continuous is env.continuous
    assert env.a4.symbolic is env.head
    assert env.head.layers == (7,)


def test_rejects_mismatched_hidden_dimensions(env):
    head = SymbolicHead(64, 256, (7,), forbidden_retrieval)
    with pytest.raises(ValueError, match="hidden sizes differ"):
        A4Model(env.continuous, head)


def test_rejects_symbolic_layers_outside_continuous_stage(env):
    head = SymbolicHead(32, 256, (8,), forbidden_retrieval)
    with pytest.raises(ValueError, match="must belong"):
        A4Model(env.continuous, head)


@pytest.mark.parametrize("layers", [(6,), (6, 7, 8)])
def test_rejects_missing_or_extra_final_latents(env, layers):
    z = {layer: env.z.get(layer, env.z[6]) for layer in layers}
    with pytest.raises(ValueError, match="cover every"):
        env.a4.forward_at_latents(env.params, {}, env.ids, z, symbolic_enabled=False)


def test_rejects_wrong_sequence_length(env):
    with pytest.raises(ValueError, match="input_ids must have shape"):
        env.a4.forward_at_latents(
            env.params, {}, env.ids[:, :-1], env.z, symbolic_enabled=False
        )


def test_rejects_nonstatic_symbolic_switch(env):
    with pytest.raises(TypeError, match="static Python bool"):
        env.a4.forward_at_latents(env.params, {}, env.ids, env.z, symbolic_enabled=1)


def test_disabled_forward_returns_live_post_pc_states(env):
    result, states = jax.jit(
        lambda params, ids, z: env.a4.forward_with_states_at_latents(
            params, {}, ids, z, symbolic_enabled=False
        )
    )(env.params, env.ids, env.z)
    assert set(states) == {7}
    projections = {
        layer: env.continuous.projection(env.params, layer)
        for layer in env.continuous.layers
    }
    expected = residual_modify(env.z, projections, env.continuous.scale)(
        result.hidden_state(7), 7
    )
    np.testing.assert_array_equal(states[7], expected)


def test_shared_checkpoint_restores_predictor_and_projection(env, tmp_path):
    source_params = env.continuous.init_params(jax.random.PRNGKey(9))
    original = env.continuous.adapter_params(source_params)
    with RunWriter(tmp_path, "a3-source") as writer:
        path = writer.save_checkpoint("final", original, global_step=0)
    restored = restore_a3_weights(env.continuous, load_params(path))
    for expected, actual in zip(
        jax.tree.leaves(source_params), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    for layer in env.continuous.layers:
        np.testing.assert_array_equal(
            env.continuous.predictor_weight(source_params, layer),
            original[f"layer_{layer}"]["A"],
        )


@pytest.mark.parametrize("missing", ["A", "B"])
def test_checkpoint_rejects_partial_continuous_weights(env, missing):
    weights = {
        name: dict(values)
        for name, values in env.continuous.adapter_params(env.params).items()
    }
    del weights["layer_6"][missing]
    with pytest.raises(ValueError, match="both A and B"):
        restore_a3_weights(env.continuous, weights)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_checkpoint_rejects_invalid_weights(env, invalid):
    weights = {
        name: dict(values)
        for name, values in env.continuous.adapter_params(env.params).items()
    }
    weights["layer_6"]["B"] = (
        jnp.zeros((1, 1))
        if invalid == "shape"
        else jnp.full_like(weights["layer_6"]["B"], jnp.nan)
    )
    with pytest.raises(ValueError, match="shape, dtype or values"):
        restore_a3_weights(env.continuous, weights)


def test_objective_rejects_invalid_label_shape_before_retrieval(env):
    with pytest.raises(ValueError, match="position needs one integer target"):
        objective_at_latents(
            env.a4,
            env.params,
            {},
            {"layer_7": {}},
            env.ids,
            env.z,
            {7: AlignmentLabels(jnp.zeros((1, 3), dtype=jnp.int32), jnp.array([0]))},
            jnp.ones((1, 256)),
            jnp.ones((1, 256)),
        )


def test_shared_task_helper_is_exactly_cross_entropy(env):
    result = env.a4.forward_at_latents(
        env.params, {}, env.ids, env.z, symbolic_enabled=False
    )
    expected = env.sub.compute_loss(result.logits, env.ids, losses=("CE",))
    terms, _ = jax.jit(
        lambda params: _task_terms_at_latents(
            env.a4,
            params,
            {},
            env.ids,
            env.z,
            symbolic_enabled=False,
        )
    )(env.params)
    np.testing.assert_allclose(terms["total"], expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_array_equal(terms["total"], terms["task"])
    assert set(terms) == {"total", "task"}


def test_a4_ce_does_not_call_kl_or_spring_predictor(env, monkeypatch):
    original_loss = env.sub.compute_loss
    calls = []

    def ce_only(*args, **kwargs):
        assert kwargs["losses"] == ("CE",)
        calls.append(kwargs["losses"])
        return original_loss(*args, **kwargs)

    def forbidden_predictor(*args, **kwargs):
        raise AssertionError("A4 must not recompute a predictor for a spring term")

    monkeypatch.setattr(
        env.continuous, "cfg", replace(env.continuous.cfg, lambda_kl=0.7)
    )
    monkeypatch.setattr(env.sub, "compute_loss", ce_only)
    monkeypatch.setattr(env.continuous, "predict_latents", forbidden_predictor)
    terms, _ = _task_terms_at_latents(
        env.a4, env.params, {}, env.ids, env.z, symbolic_enabled=False
    )
    assert calls == [("CE",)]
    np.testing.assert_array_equal(terms["total"], terms["task"])
    assert set(terms) == {"total", "task"}


@pytest.fixture(scope="module")
def aligned_env(env):
    # Deterministic unit-test retrieval targets, not a latency/ANN qualification.
    keys = jnp.eye(3, 256, dtype=jnp.float32)
    values = keys * jnp.asarray([[1.0], [2.0], [3.0]])

    def retrieve(queries):
        count = queries.shape[0]
        return RetrievedTemplates(
            jnp.broadcast_to(keys[:2], (count, 2, 256)),
            jnp.broadcast_to(values[:2], (count, 2, 256)),
            jnp.zeros((count, 2), dtype=jnp.float32),
            jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), (count, 2)),
        )

    head = SymbolicHead(32, 256, (7,), retrieve)
    return SimpleNamespace(
        a4=A4Model(env.continuous, head),
        symbolic=head.init_params(2),
        value={"layer_7": {"V_pred": jnp.full((32, 256), 0.01)}},
        labels={7: AlignmentLabels(jnp.array([[0, 1], [0, 3]]), jnp.array([0, 2]))},
        keys=keys,
        values=values,
    )


@pytest.mark.parametrize(
    "key_weight,value_weight", [(1.0, 1.0), (0.0, 0.0), (0.25, 2.0)]
)
def test_aligned_objective_is_exactly_ce_plus_key_and_value(
    env, aligned_env, monkeypatch, key_weight, value_weight
):
    aligned = aligned_env
    monkeypatch.setattr(
        env.continuous, "cfg", replace(env.continuous.cfg, lambda_kl=0.7)
    )
    result, states = aligned.a4.forward_with_states_at_latents(
        env.params, aligned.symbolic, env.ids, env.z
    )
    hidden = states[7][0, jnp.array([1, 3])]
    queries = aligned.a4.symbolic.project(aligned.symbolic["layer_7"], hidden)
    scores = queries @ aligned.keys.T
    expected_key = jnp.mean(
        jax.nn.logsumexp(scores, axis=-1) - scores[jnp.arange(2), jnp.array([0, 2])]
    )
    predictions = hidden @ aligned.value["layer_7"]["V_pred"]
    expected_value = jnp.mean(
        jnp.sum((predictions - aligned.values[jnp.array([0, 2])]) ** 2, axis=-1)
    )
    expected_ce = env.sub.compute_loss(result.logits, env.ids, losses=("CE",))

    total, terms = jax.jit(
        lambda params: objective_at_latents(
            aligned.a4,
            params,
            aligned.symbolic,
            aligned.value,
            env.ids,
            env.z,
            aligned.labels,
            aligned.keys,
            aligned.values,
            key_weight=key_weight,
            value_weight=value_weight,
        )
    )(env.params)
    assert set(terms) == {"total", "task", "key", "value"}
    for actual, expected in (
        (terms["task"], expected_ce),
        (terms["key"], expected_key),
        (terms["value"], expected_value),
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(
        total,
        terms["task"] + key_weight * terms["key"] + value_weight * terms["value"],
        rtol=1e-6,
    )


def test_a4_objective_freezes_residual_weights_and_latents(env, aligned_env):
    aligned = aligned_env

    def loss(continuous, symbolic, value, z):
        return objective_at_latents(
            aligned.a4,
            continuous,
            symbolic,
            value,
            env.ids,
            z,
            aligned.labels,
            aligned.keys,
            aligned.values,
        )[0]

    continuous, symbolic, value, latents = jax.jit(
        jax.grad(loss, argnums=(0, 1, 2, 3))
    )(env.params, aligned.symbolic, aligned.value, env.z)
    for layer in env.continuous.layers:
        predictor_gradient = env.continuous.predictor_weight(continuous, layer)
        np.testing.assert_array_equal(
            predictor_gradient, jnp.zeros_like(predictor_gradient)
        )
        projection_gradient = env.continuous.projection(continuous, layer)
        np.testing.assert_array_equal(
            projection_gradient, jnp.zeros_like(projection_gradient)
        )
        np.testing.assert_array_equal(latents[layer], jnp.zeros_like(latents[layer]))
    for gradients in (symbolic, value):
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
        assert any(np.linalg.norm(leaf) > 0 for leaf in jax.tree.leaves(gradients))


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_alignment_rejects_invalid_weight_before_forward(env, weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        objective_at_latents(
            env.a4,
            env.params,
            {},
            {},
            env.ids,
            env.z,
            {},
            jnp.ones((1, 256)),
            jnp.ones((1, 256)),
            key_weight=weight,
        )
