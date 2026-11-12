"""A4 forward composition, objectives, checkpoints and frozen-residual training."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torchax

pytest.importorskip("fabricpc.graph_assembly", reason="A3 requires FabricPC >= 0.6")

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from frozenllm.substrate.torchax_backend import is_torchax_enabled  # noqa: E402
from residual.symbolic_head.jax_head import SymbolicHead  # noqa: E402
from stages.results import RunWriter, load_params  # noqa: E402
from stages.stage_A.A3_model import A3Config, A3Model  # noqa: E402
from stages.stage_A.a3_tail import residual_modify  # noqa: E402
from stages.stage_A.a4.checkpoint import restore_a3_weights  # noqa: E402
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE, A4Config  # noqa: E402
from stages.stage_A.a4.data import A4Data  # noqa: E402
from stages.stage_A.a4.model import A4Model  # noqa: E402
from stages.stage_A.a4.objective import (  # noqa: E402
    AlignmentLabels,
    _task_terms_at_latents,
    objective_at_latents,
)
from stages.stage_A.a4.training import (  # noqa: E402
    init_params,
    make_train_step,
    residual_fingerprint,
    train,
)
from stages.stage_base import TrainingConfig  # noqa: E402
from symbolic_memory.jax_bridge import RetrievedTemplates  # noqa: E402
from tests.frozenllm.conftest import _torch_model  # noqa: E402


def small_substrate():
    enabled = is_torchax_enabled()
    if enabled:
        torchax.disable_globally()
    try:
        native = _torch_model("gpt2")
    finally:
        if enabled:
            torchax.enable_globally()
    return FrozenSubstrate(native)


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


@pytest.mark.parametrize("writeback_scale", [0.0, 0.01])
def test_multilayer_value_loss_reaches_only_upstream_symbolic_heads(
    env, aligned_env, writeback_scale
):
    head = SymbolicHead(32, 256, (6, 7), aligned_env.a4.symbolic.retrieve)
    model = A4Model(env.continuous, head)
    symbolic = head.init_params(2)
    symbolic["layer_6"]["U_out"] = writeback_scale * jax.random.normal(
        jax.random.PRNGKey(19), (256, 32)
    )
    value = {f"layer_{layer}": aligned_env.value["layer_7"] for layer in head.layers}
    labels = {layer: aligned_env.labels[7] for layer in head.layers}

    def value_loss(candidate):
        _, terms = objective_at_latents(
            model,
            env.params,
            candidate,
            value,
            env.ids,
            env.z,
            labels,
            aligned_env.keys,
            aligned_env.values,
        )
        return terms["value"]

    gradients = jax.jit(jax.grad(value_loss))(symbolic)
    for gradient in gradients["layer_7"].values():
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))
    assert np.linalg.norm(gradients["layer_6"]["U_out"]) > 0
    for name in ("W_sym", "b_sym"):
        if writeback_scale == 0:
            np.testing.assert_array_equal(
                gradients["layer_6"][name], jnp.zeros_like(gradients["layer_6"][name])
            )
        else:
            assert np.linalg.norm(gradients["layer_6"][name]) > 0


def test_zero_value_weight_disables_predictor_gradient(env, aligned_env):
    def loss(value):
        return objective_at_latents(
            aligned_env.a4,
            env.params,
            aligned_env.symbolic,
            value,
            env.ids,
            env.z,
            aligned_env.labels,
            aligned_env.keys,
            aligned_env.values,
            value_weight=0,
        )[0]

    gradients = jax.jit(jax.grad(loss))(aligned_env.value)
    for gradient in jax.tree.leaves(gradients):
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


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


@pytest.fixture(scope="module")
def run():
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


def test_optimizer_tree_contains_only_symbolic_parameters(run):
    params = init_params(run.model, 0)
    assert set(params) == {"symbolic", "value"}
    assert set(params["symbolic"]["layer_7"]) == {"W_sym", "b_sym", "U_out"}
    assert set(params["value"]["layer_7"]) == {"V_pred"}


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_training_reuses_a3_predictions_without_settlement(
    run, monkeypatch, masked, jit
):
    ids = jnp.array([[1, 2, 3, 4, 5]])
    mask = jnp.array([[True, True, True, False, False]]) if masked else None
    positions, targets = jnp.array([[0, 2]]), jnp.array([0])
    params = init_params(run.model, 0)
    continuous = run.model.continuous
    clean, _ = continuous.clean_pass(ids)
    predicted = continuous.predict_latents(run.frozen, clean)
    _, expected = objective_at_latents(
        run.model,
        run.frozen,
        params["symbolic"],
        params["value"],
        ids,
        predicted,
        {7: AlignmentLabels(positions, targets)},
        run.keys,
        run.values,
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
    config = replace(run.config, training=replace(run.config.training, jit=jit))
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        run.model, run.frozen, optimizer, run.keys, run.values, config
    )
    _, _, terms, finite = step(
        params, optimizer.init(params), ids, positions, targets, mask
    )
    assert bool(finite)
    prediction.assert_called_once()
    for name in ("task", "key", "value", "total"):
        np.testing.assert_allclose(terms[name], expected[name], rtol=3e-4, atol=3e-4)


def test_head_step_with_short_context_has_finite_masked_losses(run):
    params = init_params(run.model, 0)
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        run.model, run.frozen, optimizer, run.keys, run.values, run.config
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


def test_head_update_keeps_both_inherited_matrices_frozen(run, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A4 must not compute A3 parameter gradients")

    monkeypatch.setattr(run.model.continuous, "train_grads", forbidden)
    before = residual_fingerprint(run.model, run.frozen)
    params = init_params(run.model, 0)
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        run.model, run.frozen, optimizer, run.keys, run.values, run.config
    )
    updated, _, terms, finite = step(
        params,
        optimizer.init(params),
        jnp.array([[1, 2, 3, 4, 5]]),
        jnp.array([[0, 3]]),
        jnp.array([0]),
    )
    assert bool(finite)
    assert residual_fingerprint(run.model, run.frozen) == before
    assert set(updated) == {"symbolic", "value"}
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated), strict=True)
    )
    assert float(terms["total"]) == pytest.approx(
        float(terms["task"] + terms["key"] + terms["value"]), rel=1e-6
    )
    assert "kl" not in terms and "spring" not in terms


def test_rejects_continuous_parameters_in_optimizer(run):
    params = {**init_params(run.model, 0), "continuous": run.frozen}
    optimizer = optax.adam(1e-4)
    step = make_train_step(
        run.model, run.frozen, optimizer, run.keys, run.values, run.config
    )
    with pytest.raises(ValueError, match="only symbolic and value"):
        step(
            params,
            optimizer.init(params),
            jnp.array([[1, 2, 3, 4, 5]]),
            jnp.array([[0, 3]]),
            jnp.array([0]),
        )


@pytest.mark.parametrize("batches", [1, 2])
def test_training_checkpoint_separates_frozen_and_trainable_weights(
    run, tmp_path, caplog, batches
):
    blocks = np.array([[1, 2, 3, 4, 5]] * batches, dtype=np.int32)
    targets = np.array([[3, 0]] * batches, dtype=np.int32)
    data = A4Data(blocks, blocks, None, targets, {})
    with (
        caplog.at_level("INFO", logger="stage_a4"),
        RunWriter(tmp_path, "a4") as writer,
    ):
        result = train(run.model, run.frozen, data, run.bank, run.config, writer)
    weights = load_params(tmp_path / "a4/checkpoints/a4-final.npz")
    assert set(weights) == {"symbolic.layer_7", "value.layer_7"}
    assert (tmp_path / "a4/params/a3-frozen.npz").is_file()
    assert result["updates"] == batches
    assert result["training_state_source"] == TRAINING_STATE_SOURCE
    with np.load(tmp_path / "a4/checkpoints/a4-final.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["__metadata__"]))
    assert metadata["training_state_source"] == TRAINING_STATE_SOURCE
    assert result["frozen"]["params_unchanged"]
    assert result["frozen_residual_sha256"] == residual_fingerprint(
        run.model, run.frozen
    )
    assert "Task 1/1" in caplog.text and "Epoch 1/1" in caplog.text
    assert "Total loss" in caplog.text and "Symbolic loss" in caplog.text
    assert f"{batches}/{batches}" in caplog.text and "Final validation:" in caplog.text
    messages = [record.getMessage() for record in caplog.records]
    headings = [i for i, message in enumerate(messages) if "Symbolic loss" in message]
    assert len(headings) == 2
    assert messages[headings[-1] - 1].strip().startswith(f"{batches}/{batches}")
    assert result["communication"]["status"] == "not_measured"
    mean_ms = result["training_step_mean_ms"]
    assert mean_ms is None if batches == 1 else mean_ms > 0
    records = [
        json.loads(row)
        for row in (tmp_path / "a4/metrics.jsonl").read_text().splitlines()
    ]
    for row in records:
        if row["event"] == "train":
            assert row["ppl"] == pytest.approx(math.exp(row["task"]))
            assert row["symbolic"] == pytest.approx(row["key"] + row["value"])


@pytest.mark.parametrize("latency_every", [0, 2])
def test_communication_samples_real_updates_without_replay(
    run, tmp_path, monkeypatch, caplog, latency_every
):
    from stages.stage_A.a4 import training as module

    class TimingRetriever:
        active = False
        calls = 0

        @contextmanager
        def transfer_timing(self):
            self.active = True
            try:
                yield
            finally:
                self.active = False

        def transfer_stats(self):
            return {
                "calls": self.calls,
                "d2h_gpu_seconds": self.calls * 0.00001,
                "h2d_gpu_seconds": self.calls * 0.00002,
            }

    timing = TimingRetriever()
    # Keep the numerical fixture's retrieval unchanged; simulate only its clock API.
    retrieve = run.model.symbolic.retrieve
    monkeypatch.setattr(
        run.model.symbolic,
        "retrieve",
        SimpleNamespace(
            transfer_timing=timing.transfer_timing, transfer_stats=timing.transfer_stats
        ),
    )
    original = module.make_train_step
    invocations = []

    def make_step(*args):
        monkeypatch.setattr(run.model.symbolic, "retrieve", retrieve)
        numerical_step = original(*args)

        def step(*inputs):
            invocations.append(timing.active)
            output = numerical_step(*inputs)
            if timing.active:
                timing.calls += 1
            return output

        return step

    monkeypatch.setattr(module, "make_train_step", make_step)
    blocks = np.array([[1, 2, 3, 4, 5]] * 12, dtype=np.int32)
    data = A4Data(blocks, blocks[:1], None, np.array([[3, 0]] * 12), {})
    with (
        caplog.at_level("INFO", logger="stage_a4"),
        RunWriter(tmp_path, "a4") as writer,
    ):
        result = train(
            run.model,
            run.frozen,
            data,
            run.bank,
            run.config,
            writer,
            latency_every=latency_every,
        )
    assert result["updates"] == len(invocations) == 12
    assert not any(invocations[:10])
    assert invocations[10:] == [False, bool(latency_every)]
    report = result["communication"]
    assert report["samples"] == (1 if latency_every else 0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "a4/metrics.jsonl").read_text().splitlines()
    ]
    sampled = [
        row for row in rows if row["event"] == "train" and "communication" in row
    ]
    assert len(sampled) == report["samples"]
    if latency_every:
        assert sampled[0]["step"] == 12
        assert report["copy_mean_ms"] == pytest.approx(0.03)
        assert report["copy_ratio_of_means_percent"] == pytest.approx(
            100 * 0.00003 / sampled[0]["step_time"]
        )
        assert "CUDA copies during training:" in caplog.text
    weights = load_params(tmp_path / "a4/checkpoints/a4-final.npz")
    initial = init_params(run.model, run.config.training.seed)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(weights["symbolic.layer_7"]),
            jax.tree.leaves(initial["symbolic"]["layer_7"]),
            strict=True,
        )
    )
    assert result["frozen"]["params_unchanged"]


def test_eager_train_step_uses_same_parameter_groups(run):
    params = init_params(run.model, 0)
    optimizer = optax.adam(1e-4)
    config = replace(run.config, training=replace(run.config.training, jit=False))
    step = make_train_step(
        run.model, run.frozen, optimizer, run.keys, run.values, config
    )
    updated, _, _, finite = step(
        params,
        optimizer.init(params),
        jnp.array([[1, 2, 3, 4, 5]]),
        jnp.array([[0, 3]]),
        jnp.array([0]),
    )
    assert bool(finite) and set(updated) == {"symbolic", "value"}
