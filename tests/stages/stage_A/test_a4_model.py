"""A4 layer constraints and disabled-head equivalence with A3 inference."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("fabricpc.graph_assembly", reason="A3 requires FabricPC >= 0.6")

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from frozenllm.substrate.loader import _native_torch_loading  # noqa: E402
from residual.symbolic_head.jax_head import SymbolicHead  # noqa: E402
from stages.results import RunWriter, load_params  # noqa: E402
from stages.stage_A.A3_model import A3Config, A3Model  # noqa: E402
from stages.stage_A.a3_tail import residual_modify  # noqa: E402
from stages.stage_A.a4 import A4Model  # noqa: E402
from stages.stage_A.a4_checkpoint import restore_a3_weights  # noqa: E402
from stages.stage_A.a4_objective import (  # noqa: E402
    AlignmentLabels,
    objective_at_latents,
)
from tests.frozenllm.conftest import _torch_model  # noqa: E402


def forbidden_retrieval(queries):
    raise AssertionError("Retrieval must not run in this disabled-head check")


@pytest.fixture(scope="module")
def env():
    with _native_torch_loading():
        native = _torch_model("gpt2")
    sub = FrozenSubstrate(native)
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
    hidden, _ = env.continuous.clean_pass(env.ids)
    with pytest.raises(ValueError, match="position needs one integer target"):
        objective_at_latents(
            env.a4,
            env.params,
            {},
            {"layer_7": {}},
            env.ids,
            env.z,
            hidden,
            {7: AlignmentLabels(jnp.zeros((1, 3), dtype=jnp.int32), jnp.array([0]))},
            jnp.ones((1, 256)),
            jnp.ones((1, 256)),
        )
