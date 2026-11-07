"""A4 optimizer ownership and frozen-residual invariance."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from residual.symbolic_head.jax_head import SymbolicHead
from stages.results import RunWriter, load_params
from stages.stage_A.A3_model import A3Config, A3Model
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE, A4Config
from stages.stage_A.a4.data import A4Data
from stages.stage_A.a4.model import A4Model
from stages.stage_A.a4.objective import AlignmentLabels, objective_at_latents
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


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_training_reuses_a3_predictions_without_settlement(
    env, monkeypatch, masked, jit
):
    ids = jnp.array([[1, 2, 3, 4, 5]])
    mask = jnp.array([[True, True, True, False, False]]) if masked else None
    positions, targets = jnp.array([[0, 2]]), jnp.array([0])
    params = init_params(env.model, 0)
    continuous = env.model.continuous
    clean, _ = continuous.clean_pass(ids)
    predicted = continuous.predict_latents(env.frozen, clean)
    _, expected = objective_at_latents(
        env.model,
        env.frozen,
        params["symbolic"],
        params["value"],
        ids,
        predicted,
        {7: AlignmentLabels(positions, targets)},
        env.keys,
        env.values,
        token_mask=mask,
    )
    prediction = Mock(wraps=continuous.predict_latents)
    monkeypatch.setattr(continuous, "predict_latents", prediction)

    def forbidden(*args, **kwargs):
        raise AssertionError("A4 must reuse prediction, not run A3 settlement")

    for method in ("make_batch", "_refresh_state", "train_grads"):
        monkeypatch.setattr(continuous, method, forbidden)
    original_config = continuous.cfg

    class PredictionOnlyConfig:
        def __getattr__(self, name):
            if name in {"infer_steps", "eta_z", "sigma2", "lambda_kl"}:
                raise AssertionError("A4 must not read A3 settlement settings")
            return getattr(original_config, name)

    monkeypatch.setattr(continuous, "cfg", PredictionOnlyConfig())
    config = replace(env.config, training=replace(env.config.training, jit=jit))
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        env.model, env.frozen, optimizer, env.keys, env.values, config
    )
    _, _, terms, finite = step(
        params, optimizer.init(params), ids, positions, targets, mask
    )
    assert bool(finite)
    prediction.assert_called_once()
    for name in ("task", "key", "value", "total"):
        np.testing.assert_allclose(terms[name], expected[name], rtol=3e-4, atol=3e-4)


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
    assert result["training_state_source"] == TRAINING_STATE_SOURCE
    with np.load(tmp_path / "a4/checkpoints/a4-final.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["__metadata__"]))
    assert metadata["training_state_source"] == TRAINING_STATE_SOURCE
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
        jnp.array([[0, 3]]),
        jnp.array([0]),
    )
    assert bool(finite) and set(updated) == {"symbolic", "value"}
