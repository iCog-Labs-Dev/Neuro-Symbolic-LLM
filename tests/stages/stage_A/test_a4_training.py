"""A4 optimizer ownership and frozen-residual invariance."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from residual.symbolic_head.jax_head import SymbolicHead
from stages.results import RunWriter, load_params
from stages.stage_A.A3_model import A3Config, A3Model
from stages.stage_A.a4.config import A4Config
from stages.stage_A.a4.data import A4Data
from stages.stage_A.a4.model import A4Model
from stages.stage_A.a4.settlement import infer_latents
from stages.stage_A.a4.training import (
    init_params,
    make_train_step,
    residual_fingerprint,
    train,
)
from stages.stage_base import TrainingConfig
from symbolic_memory.jax_bridge import RetrievedTemplates
from tests.stages.stage_A.a4_fixtures import small_substrate


@pytest.fixture(scope="module")
def env():
    substrate = small_substrate()
    continuous = A3Model(
        substrate, layers=(7,), rank=4, seq_len=5, config=A3Config(infer_steps=1)
    )
    frozen = continuous.init_params(jax.random.PRNGKey(0))
    keys = jnp.eye(2, 256, dtype=jnp.float32)
    values = keys * 2

    def retrieve(queries):
        count = len(queries)
        return RetrievedTemplates(
            jnp.broadcast_to(keys, (count, 2, 256)),
            jnp.broadcast_to(values, (count, 2, 256)),
            jnp.zeros((count, 2)),
            jnp.broadcast_to(jnp.arange(2), (count, 2)),
        )

    model = A4Model(continuous, SymbolicHead(32, 256, (7,), retrieve))
    config = A4Config(
        Path("a3.yaml"),
        TrainingConfig(
            model="unit-fixture", seq_len=5, batch_size=1, epochs=1, eval_batches=1
        ),
        minimum_templates=1,
    )
    records = tuple(
        SimpleNamespace(
            key_vector=np.asarray(keys[i]), val_vector=np.asarray(values[i])
        )
        for i in range(2)
    )
    bank = SimpleNamespace(records=records, input_snapshot_sha256="unit-bank")
    return SimpleNamespace(
        model=model, frozen=frozen, config=config, bank=bank, keys=keys, values=values
    )


def test_optimizer_tree_contains_only_symbolic_parameters(env):
    params = init_params(env.model, 0)
    assert set(params) == {"symbolic", "value"}
    assert set(params["symbolic"]["layer_7"]) == {"W_sym", "b_sym", "U_out"}
    assert set(params["value"]["layer_7"]) == {"V_pred"}


def test_latent_inference_matches_a3_without_weight_gradient_computation(env):
    ids = jnp.array([[1, 2, 3, 4, 5]])
    key = jax.random.PRNGKey(1)
    actual = infer_latents(env.model.continuous, env.frozen, ids, key)
    _, aux = env.model.continuous.train_grads(env.frozen, ids, key)
    np.testing.assert_array_equal(actual[7], aux["z_star"][7])


@pytest.mark.parametrize("steps", [1, 2])
def test_masked_settlement_matches_fabricpc_when_all_tokens_are_real(env, steps):
    model = A3Model(
        env.model.continuous.sub,
        layers=(7,),
        rank=4,
        seq_len=5,
        config=A3Config(infer_steps=steps),
    )
    frozen = model.init_params(jax.random.PRNGKey(0))
    ids = jnp.array([[1, 2, 3, 4, 5]])
    key = jax.random.PRNGKey(1)
    expected = infer_latents(model, frozen, ids, key)
    actual = jax.jit(lambda *args: infer_latents(model, *args))(
        frozen, ids, key, jnp.ones_like(ids, dtype=bool)
    )
    np.testing.assert_allclose(actual[7], expected[7], rtol=3e-4, atol=3e-4)


def test_masked_settlement_cannot_learn_from_padding_token_values(env):
    mask = jnp.array([[True, True, True, False, False]])
    key = jax.random.PRNGKey(1)
    settle = jax.jit(lambda *args: infer_latents(env.model.continuous, *args))
    left = settle(env.frozen, jnp.array([[1, 2, 3, 4, 5]]), key, mask)
    right = settle(env.frozen, jnp.array([[1, 2, 3, 9, 11]]), key, mask)
    np.testing.assert_allclose(left[7][:, :3], right[7][:, :3], rtol=3e-4, atol=3e-4)


def test_head_step_with_short_context_has_finite_masked_losses(env):
    params = init_params(env.model, 0)
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        env.model, env.frozen, optimizer, env.keys, env.values, env.config
    )
    _, _, terms, finite = step(
        params,
        optimizer.init(params),
        jnp.array([[1, 2, 3, 0, 0]]),
        jax.random.PRNGKey(1),
        jnp.array([[0, 2]]),
        jnp.array([0]),
        jnp.array([[True, True, True, False, False]]),
    )
    assert bool(finite)
    assert set(terms).issuperset({"task", "key", "value", "total"})
    assert "spring" not in terms and "kl" not in terms
    _, _, changed_padding_terms, changed_padding_finite = step(
        params,
        optimizer.init(params),
        jnp.array([[1, 2, 3, 9, 11]]),
        jax.random.PRNGKey(1),
        jnp.array([[0, 2]]),
        jnp.array([0]),
        jnp.array([[True, True, True, False, False]]),
    )
    assert bool(changed_padding_finite)
    for name in ("task", "key", "value", "total"):
        np.testing.assert_allclose(
            terms[name], changed_padding_terms[name], rtol=3e-4, atol=3e-4
        )


def test_head_update_keeps_both_inherited_matrices_frozen(env, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A4 must not compute A3 parameter gradients")

    monkeypatch.setattr(env.model.continuous, "train_grads", forbidden)
    before = residual_fingerprint(env.model, env.frozen)
    params = init_params(env.model, 0)
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        env.model, env.frozen, optimizer, env.keys, env.values, env.config
    )
    updated, _, terms, finite = step(
        params,
        optimizer.init(params),
        jnp.array([[1, 2, 3, 4, 5]]),
        jax.random.PRNGKey(1),
        jnp.array([[0, 3]]),
        jnp.array([0]),
    )
    assert bool(finite)
    assert residual_fingerprint(env.model, env.frozen) == before
    assert set(updated) == {"symbolic", "value"}
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated), strict=True)
    )
    assert float(terms["total"]) == pytest.approx(
        float(terms["task"] + terms["key"] + terms["value"]), rel=1e-6
    )
    assert "kl" not in terms and "spring" not in terms


def test_rejects_continuous_parameters_in_optimizer(env):
    params = {**init_params(env.model, 0), "continuous": env.frozen}
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        env.model, env.frozen, optimizer, env.keys, env.values, env.config
    )
    with pytest.raises(ValueError, match="only symbolic and value"):
        step(
            params,
            optimizer.init(params),
            jnp.array([[1, 2, 3, 4, 5]]),
            jax.random.PRNGKey(1),
            jnp.array([[0, 3]]),
            jnp.array([0]),
        )


def test_training_checkpoint_separates_frozen_and_trainable_weights(env, tmp_path):
    blocks = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    data = A4Data(blocks, blocks, None, np.array([[3, 0]], dtype=np.int32), {})
    with RunWriter(tmp_path, "a4") as writer:
        result = train(env.model, env.frozen, data, env.bank, env.config, writer)
    weights = load_params(tmp_path / "a4/checkpoints/a4-final.npz")
    assert set(weights) == {"symbolic.layer_7", "value.layer_7"}
    assert (tmp_path / "a4/params/a3-frozen.npz").is_file()
    assert result["updates"] == 1
    assert result["frozen"]["params_unchanged"]
    assert result["frozen_residual_sha256"] == residual_fingerprint(
        env.model, env.frozen
    )


def test_eager_train_step_uses_same_parameter_groups(env):
    params = init_params(env.model, 0)
    optimizer = optax.adam(1e-4)
    config = replace(env.config, training=replace(env.config.training, jit=False))
    step = make_train_step(
        env.model, env.frozen, optimizer, env.keys, env.values, config
    )
    updated, _, _, finite = step(
        params,
        optimizer.init(params),
        jnp.array([[1, 2, 3, 4, 5]]),
        jax.random.PRNGKey(1),
        jnp.array([[0, 3]]),
        jnp.array([0]),
    )
    assert bool(finite) and set(updated) == {"symbolic", "value"}
