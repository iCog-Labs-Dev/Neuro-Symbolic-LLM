"""Exercise sequential A4 with a tiny frozen model and an in-memory bank."""

import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments import run_stage_a4 as runner
from frozenllm.substrate import FrozenSubstrate
from residual.symbolic_head.jax_head import SymbolicHead
from stages.results import RunWriter, load_checkpoint, load_params
from stages.stage_A.a4 import training
from stages.stage_A.a4.checkpoint import unpack_head_params
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE, A4Config
from stages.stage_A.a4.data import ROCSTORIES_EVAL_GROUPS, load_data
from stages.stage_A.a4.model import A4Model
from stages.stage_A.a4.objective import AlignmentLabels, objective_at_latents
from stages.stage_base import TrainingConfig
from symbolic_memory.jax_bridge import RetrievedTemplates


class TinyTokenizer:
    pad_token_id = eos_token_id = 0

    def __call__(self, text, *, return_offsets_mapping=False):
        tokens = [int(token) for token in text.split()]
        encoded = {"input_ids": tokens}
        if return_offsets_mapping:
            # Synthetic documents consist of single-digit tokens separated by spaces.
            encoded["offset_mapping"] = [(2 * i, 2 * i + 1) for i in range(len(tokens))]
        return encoded


class TinySubstrate:
    compute_loss = staticmethod(FrozenSubstrate.compute_loss)

    def __init__(self):
        rng = np.random.default_rng(17)
        self.weights = rng.normal(size=(8, 3)).astype(np.float32)
        self.initial_weights = self.weights.copy()
        self.tokenizer = TinyTokenizer()
        self.architecture = object()

    def run_with_interception(self, ids, *, modify_fn, intercept_layers):
        embeddings = jnp.asarray(self.weights)
        hidden = embeddings[ids]
        states = {}
        for layer in intercept_layers:
            hidden = modify_fn(jnp.tanh(hidden), layer)
            states[layer] = hidden
        return SimpleNamespace(logits=hidden @ embeddings.T, states=states)

    def verify_frozen(self):
        return {"params_unchanged": np.array_equal(self.weights, self.initial_weights)}


class TinyContinuous:
    layers = (0, 1)
    hidden_size = 3
    seq_len = 5
    scale = 0.5

    def __init__(self, substrate):
        self.sub = substrate

    def clean_pass(self, ids):
        clean = self.sub.run_with_interception(
            ids, modify_fn=lambda hidden, layer: hidden, intercept_layers=self.layers
        )
        return clean.states, clean.logits

    def predict_latents(self, params, clean):
        return {
            layer: clean[layer] @ params[f"layer_{layer}"]["A"] for layer in self.layers
        }

    def projection(self, params, layer):
        return params[f"layer_{layer}"]["B"]

    def adapter_params(self, params):
        return params


@pytest.fixture
def tiny_run():
    substrate = TinySubstrate()
    continuous = TinyContinuous(substrate)
    keys = jnp.eye(2, dtype=jnp.float32)
    values = jnp.asarray([[0.7, -0.3], [0.1, 1.2]])

    def retrieve(queries):
        count = len(queries)
        return RetrievedTemplates(
            jnp.broadcast_to(keys, (count, 2, 2)),
            jnp.broadcast_to(values, (count, 2, 2)),
            jnp.zeros((count, 2)),
            jnp.broadcast_to(jnp.arange(2), (count, 2)),
        )

    retrieve.info = {"backend": "in-memory"}
    bank = SimpleNamespace(
        records=[
            SimpleNamespace(
                template_id=str(i), key_vector=keys[i], val_vector=values[i]
            )
            for i in range(2)
        ],
        input_snapshot_sha256="tiny-bank",
        vector_dim=2,
    )
    config = A4Config(
        residual_config=runner.DEFAULT_CONFIG.with_name("a3_hybrid_pc.yaml"),
        training=TrainingConfig(
            model="tiny",
            seq_len=5,
            batch_size=2,
            epochs=1,
            eval_batches=1,
            test_batches=1,
            log_every=1,
            jit=False,
        ),
        key_weight=0.6,
        value_weight=1.4,
        minimum_templates=2,
        top_m=2,
    )
    rng = np.random.default_rng(3)
    frozen = {
        f"layer_{layer}": {
            "A": rng.normal(size=(3, 1)).astype(np.float32),
            "B": rng.normal(size=(1, 3)).astype(np.float32),
        }
        for layer in continuous.layers
    }
    model = A4Model(continuous, SymbolicHead(3, 2, continuous.layers, retrieve))
    return SimpleNamespace(
        model=model,
        config=config,
        frozen=frozen,
        bank=bank,
        keys=keys,
        values=values,
        retrieve=retrieve,
    )


@pytest.fixture
def prepared_tasks(tmp_path, tiny_run):
    def write_sources(name, rows):
        path = tmp_path / name
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        return path

    docs = {1: [1, 2, 3, 4], 2: [2, 3, 1], 3: [3, 4, 5]}
    sources = write_sources(
        "train.jsonl",
        [
            {
                "corpus_id": f"corpus-{4 - task_id}",
                "document_id": "shared",
                "task_id": task_id,
                "source_text": " ".join(map(str, docs[task_id])),
            }
            for task_id in (3, 1, 2)
        ],
    )
    ids, masks, targets, labels = [], [], [], []
    occurrences = [(2, 1), (1, 3), (3, 2), (1, 1), (2, 2), (1, 2)]
    for task_id, position in occurrences:
        tokens = docs[task_id]
        target = position % 2
        ids.append(tokens + [0] * (5 - len(tokens)))
        masks.append([True] * len(tokens) + [False] * (5 - len(tokens)))
        targets.append([position, target])
        labels.append(
            {
                "corpus_id": f"corpus-{4 - task_id}",
                "document_id": "shared",
                "template_id": str(target),
                "token_index": position,
                "window_start": 0,
                "window_end": len(tokens),
                "source_spans": [
                    {
                        "start": 2 * position,
                        "end": 2 * position + 1,
                        "text": str(tokens[position]),
                    }
                ],
            }
        )
    arrays = tmp_path / "aligned.npz"
    np.savez(
        arrays,
        input_ids=np.asarray(ids, dtype=np.int32),
        token_mask=np.asarray(masks),
        targets=np.asarray(targets, dtype=np.int32),
    )
    manifest = tmp_path / "aligned.json"
    manifest.write_text(
        json.dumps(
            {
                "contract_version": "alignment-occurrences-v2",
                "seq_len": 5,
                "bank_snapshot_sha256": tiny_run.bank.input_snapshot_sha256,
                "source_sha256": {
                    "sources": hashlib.sha256(sources.read_bytes()).hexdigest()
                },
                "arrays_file": arrays.name,
                "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                "labels": labels,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        data_mode="sequential-aligned",
        alignment_labels=manifest,
        train_sources=sources,
        validation_sources=write_sources(
            "validation.jsonl",
            [
                {"source_text": "6 1"},
                {"source_text": "6 2"},
                {"source_text": "6 3 4 5 6 7"},
                {"source_text": "6 4"},
                {"source_text": "6 5"},
                {"source_text": "6 6"},
            ],
        ),
        test_sources=write_sources(
            "test.jsonl",
            [
                {"source_text": "7 1"},
                {"source_text": "7 2 3"},
            ],
        ),
        skip_test=False,
        max_train_rows=6,
        max_train_blocks=6,
        minimum_examples=6,
        max_eval_rows=6,
        max_eval_blocks=7,
    )
    data = load_data(args, TinyTokenizer(), tiny_run.bank, tiny_run.config.training)
    return args, data


def test_partition_preserves_occurrences_with_shared_document_ids(prepared_tasks):
    _, data = prepared_tasks
    assert [task.name for task in data.tasks] == [
        "rocstories_task_1",
        "rocstories_task_2",
        "rocstories_task_3",
    ]
    assert [len(task.train_blocks) for task in data.tasks] == [3, 2, 1]
    assert [task.targets.tolist() for task in data.tasks] == [
        [[3, 1], [1, 1], [2, 0]],
        [[1, 1], [2, 0]],
        [[2, 0]],
    ]
    for task, tokens in zip(
        data.tasks, ([1, 2, 3, 4, 0], [2, 3, 1, 0, 0], [3, 4, 5, 0, 0]), strict=True
    ):
        np.testing.assert_array_equal(task.train_blocks, [tokens] * len(task.labels))
        np.testing.assert_array_equal(task.train_mask, task.train_blocks != 0)
        assert {label["document_id"] for label in task.labels} == {"shared"}
    assert [
        counts["source_documents"]
        for counts in data.provenance["alignment_counts"].values()
    ] == [1, 1, 1]


def test_default_validation_groups_partition_whole_documents(tiny_run, prepared_tasks):
    args, data = prepared_tasks
    reloaded = load_data(args, TinyTokenizer(), tiny_run.bank, tiny_run.config.training)
    assert tuple(data.task_eval) == ROCSTORIES_EVAL_GROUPS
    assert data.provenance["continual_evaluation"]["group_order"] == list(
        ROCSTORIES_EVAL_GROUPS
    )
    selected = data.provenance["validation"]["selected"]
    assert any(row["blocks"] > 1 for row in selected)
    ordered = sorted(selected, key=lambda row: row["text_sha256"])
    for index, name in enumerate(ROCSTORIES_EVAL_GROUPS):
        documents = data.provenance["task_eval"][name]["selected"]
        assert documents == ordered[index::3]
        indices = [
            block_index
            for row in documents
            for block_index in range(
                row["block_start"], row["block_start"] + row["blocks"]
            )
        ]
        ids, mask = data.task_eval[name]
        np.testing.assert_array_equal(ids, data.validation[indices])
        np.testing.assert_array_equal(mask, data.validation_mask[indices])
        np.testing.assert_array_equal(ids, reloaded.task_eval[name][0])
        np.testing.assert_array_equal(mask, reloaded.task_eval[name][1])


def test_default_validation_groups_require_three_documents(tiny_run, prepared_tasks):
    args, _ = prepared_tasks
    args.max_eval_rows = 2
    with pytest.raises(ValueError, match="at least three distinct held-out documents"):
        load_data(args, TinyTokenizer(), tiny_run.bank, tiny_run.config.training)


@pytest.fixture
def task_eval_sources(tmp_path):
    path = tmp_path / "task_eval.jsonl"
    rows = [
        {
            "task_id": task_id,
            "corpus_id": "held-out",
            "document_id": str(index),
            "source_text": text,
        }
        for index, (task_id, text) in enumerate(
            [
                (3, "5 3"),
                (1, "4 1"),
                (2, "5 1"),
                (1, "4 2 3"),
                (3, "5 6"),
                (2, "5 2 4 1"),
            ]
        )
    ]
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    return path


@pytest.fixture
def prepared_task_eval(tiny_run, prepared_tasks, task_eval_sources):
    args, _ = prepared_tasks
    args.task_eval_sources = task_eval_sources
    return args, load_data(
        args, TinyTokenizer(), tiny_run.bank, tiny_run.config.training
    )


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("missing_task", "all three ROCStories task IDs"),
        ("invalid_id", "invalid held-out task ID"),
        ("duplicate", "source overlaps"),
        ("training_source", "source overlaps"),
        ("shared_validation_source", "source overlaps"),
    ],
)
def test_rejects_incomplete_or_overlapping_held_out_groups(
    tiny_run, prepared_tasks, task_eval_sources, mutation, error
):
    args, _ = prepared_tasks
    rows = [json.loads(line) for line in task_eval_sources.read_text().splitlines()]
    if mutation == "missing_task":
        rows = [row for row in rows if row["task_id"] != 3]
    elif mutation == "invalid_id":
        rows[0]["task_id"] = 4
    elif mutation == "duplicate":
        rows[0]["source_text"] = rows[1]["source_text"]
    elif mutation == "training_source":
        rows[0]["source_text"] = "1 2 3 4"
    else:
        rows[0]["source_text"] = "6 1"
    task_eval_sources.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    args.task_eval_sources = task_eval_sources
    with pytest.raises(ValueError, match=error):
        load_data(args, TinyTokenizer(), tiny_run.bank, tiny_run.config.training)


def test_continual_evaluation_reports_fixed_groups_and_shared_splits(
    tmp_path, monkeypatch, caplog, tiny_run, prepared_tasks, prepared_task_eval
):
    args, data = prepared_task_eval
    _, default_groups = prepared_tasks
    names = list(ROCSTORIES_EVAL_GROUPS)
    assert list(data.task_eval) == names
    assert (
        data.provenance["source_fingerprints"]["task_eval"]
        == hashlib.sha256(args.task_eval_sources.read_bytes()).hexdigest()
    )
    for actual, expected in zip(data.tasks, default_groups.tasks, strict=True):
        np.testing.assert_array_equal(actual.train_blocks, expected.train_blocks)
        np.testing.assert_array_equal(actual.train_mask, expected.train_mask)
        np.testing.assert_array_equal(actual.targets, expected.targets)
        assert actual.labels == expected.labels
    np.testing.assert_array_equal(data.validation, default_groups.validation)
    np.testing.assert_array_equal(data.test, default_groups.test)
    assert [blocks[:, 0].tolist() for blocks, _ in data.task_eval.values()] == [
        [4, 4],
        [5, 5],
        [5, 5],
    ]
    assert [mask[:, 1:].sum() for _, mask in data.task_eval.values()] == [3, 4, 2]

    splits = {
        "shared_validation": (data.validation, data.validation_mask),
        **data.task_eval,
        "shared_test": (data.test, data.test_mask),
    }
    loss_history = [[8, 9, 10], [3, 6, 7], [2, 4, 5], [3, 5, 2]]
    calls = []

    def evaluate(model, frozen, params, blocks, config, *, token_masks, eval_batches):
        name = next(name for name, (ids, _) in splits.items() if blocks is ids)
        assert token_masks is splits[name][1]
        round_index = len(calls) // 4
        calls.append(name)
        loss = loss_history[round_index][names.index(name)] if name in names else 1.0
        return {"loss": loss, "ppl": float(np.exp(loss)), "batches": eval_batches}

    monkeypatch.setattr(training, "evaluate", evaluate)
    summarize_losses = training.summarize_continual_losses

    def summarize_after_evaluation(losses):
        assert calls == ["shared_validation", *names] * 4 + ["shared_test"]
        np.testing.assert_array_equal(losses, loss_history[1:])
        return summarize_losses(losses)

    summarize = Mock(side_effect=summarize_after_evaluation)
    monkeypatch.setattr(training, "summarize_continual_losses", summarize)
    with (
        caplog.at_level("INFO", logger="stage_a4"),
        RunWriter(tmp_path, "groups") as writer,
    ):
        report = training.train_sequential(
            tiny_run.model,
            tiny_run.frozen,
            data,
            tiny_run.bank,
            tiny_run.config,
            writer,
        )
    assert calls == ["shared_validation", *names] * 4 + ["shared_test"]
    summarize.assert_called_once()
    assert [report["initial"][name]["loss"] for name in names] == loss_history[0]
    assert report["loss_matrix"] == loss_history[1:]
    np.testing.assert_allclose(report["perplexity_matrix"], np.exp(loss_history[1:]))
    assert report["mean_seen_loss"] == pytest.approx([3, 3, 10 / 3])
    assert report["mean_forgetting"] == [None, -1, 1]
    assert report["backward_transfer_ce"] == pytest.approx(-0.5)
    assert report["continual_evaluation"]["scope"] == "fixed_held_out_groups"
    assert (
        "independent semantic domains"
        in report["continual_evaluation"]["interpretation"]
    )
    assert json.loads((writer.dir / "results.json").read_text()) == report
    assert "Validation before training" in caplog.text
    validation = [
        message for message in caplog.messages if "Validation after Task" in message
    ]
    assert len(validation) == 3
    for index, message in enumerate(validation, start=1):
        assert f"Validation after Task {index}/3" in message
        for name, loss in zip(names, loss_history[index], strict=True):
            assert f"{name} CE: {loss:.4f} PPL: {np.exp(loss):.2f}" in message
    assert caplog.text.count("Continual Learning Metrics:") == 1
    assert caplog.text.index(validation[-1]) < caplog.text.index(
        "Continual Learning Metrics:"
    )
    assert caplog.text.count("Mean Forgetting:") == 1
    assert "Mean Forgetting: [None, -1.0000, +1.0000] (Final: +1.0000)" in caplog.text
    assert "Mean Seen Loss : [3.0000, 3.0000, 3.3333] (Final: 3.3333)" in caplog.text
    assert "Backward Transfer (CE): -0.5000" in caplog.text
    events = [
        json.loads(line)
        for line in (writer.dir / "metrics.jsonl").read_text().splitlines()
    ]
    continual = [event for event in events if event["event"] == "continual_evaluation"]
    assert [event["after_task"] for event in continual] == [2]
    assert continual[0]["mean_forgetting"] == 1
    assert continual[0]["backward_transfer_ce"] == -0.5


def test_single_adam_update_combines_the_three_parameter_gradients(
    tiny_run, prepared_tasks
):
    _, data = prepared_tasks
    ids, positions, targets, mask = map(
        jnp.asarray, data.tasks[0].batch_inputs(np.array([1, 0]))
    )
    model, frozen, config = tiny_run.model, tiny_run.frozen, tiny_run.config
    params = training.init_params(model, config.training.seed)
    # A trained writeback lets CE produce query gradients at both layers.
    for layer in params["symbolic"].values():
        layer["U_out"] = jnp.array([[0.10, -0.20, 0.15], [-0.05, 0.25, -0.10]])
    clean, _ = model.continuous.clean_pass(ids)
    latents = model.continuous.predict_latents(frozen, clean)
    labels = dict.fromkeys(model.symbolic.layers, AlignmentLabels(positions, targets))
    names = ("task", "key", "value")

    def component_losses(candidate):
        _, terms = objective_at_latents(
            model,
            frozen,
            candidate["symbolic"],
            candidate["value"],
            ids,
            latents,
            labels,
            tiny_run.keys,
            tiny_run.values,
            token_mask=mask,
            detach_alignment_states=True,
        )
        return jnp.stack([terms[name] for name in names])

    values, pullback = jax.vjp(component_losses, params)
    component_grads = [pullback(basis)[0] for basis in jnp.eye(3)]
    paths = {
        "W_sym": (True, True, False),
        "b_sym": (True, True, False),
        "U_out": (True, False, False),
        "V_pred": (False, False, True),
    }
    for group, layers in params.items():
        for layer, weights in layers.items():
            for name in weights:
                for grads, active in zip(component_grads, paths[name], strict=True):
                    grad = np.asarray(grads[group][layer][name])
                    assert np.isfinite(grad).all()
                    if active:
                        assert np.linalg.norm(grad) > 1e-7
                    else:
                        np.testing.assert_array_equal(grad, np.zeros_like(grad))

    combined = jax.tree.map(
        lambda ce, key, value: ce
        + config.key_weight * key
        + config.value_weight * value,
        *component_grads,
    )
    adam = optax.adam(config.training.learning_rate)
    observed = Mock(wraps=adam.update)
    optimizer = optax.GradientTransformation(adam.init, observed)
    state = optimizer.init(params)
    step = training.make_train_step(
        model,
        frozen,
        optimizer,
        tiny_run.keys,
        tiny_run.values,
        config,
        detach_alignment_states=True,
    )
    updated, next_state, terms, finite = step(
        params, state, ids, positions, targets, mask
    )
    assert bool(finite)
    assert all(float(value) > 0 and np.isfinite(value) for value in values)
    np.testing.assert_allclose([terms[name] for name in names], values, rtol=1e-6)
    np.testing.assert_allclose(
        terms["total"], values @ jnp.array([1, 0.6, 1.4]), rtol=1e-6
    )
    observed.assert_called_once()
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
        observed.call_args.args[0],
        combined,
    )
    expected_updates, expected_state = adam.update(combined, state, params)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
        next_state,
        expected_state,
    )
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
        updated,
        optax.apply_updates(params, expected_updates),
    )
    assert int(next_state[0].count) == 1


@pytest.mark.parametrize("reset_optimizer", [False, True])
def test_transitions_keep_parameters_and_checkpoint_the_exact_optimizer_state(
    tmp_path, monkeypatch, caplog, tiny_run, prepared_tasks, reset_optimizer
):
    _, data = prepared_tasks
    config = replace(
        tiny_run.config,
        training=replace(tiny_run.config.training, reset_optimizer=reset_optimizer),
    )
    make_step = training.make_train_step
    incoming, outgoing, batches = [], [], []

    def capture_step(*args, **kwargs):
        step = make_step(*args, **kwargs)

        def update(params, state, ids, positions, targets, masks):
            incoming.append((params, state))
            np.testing.assert_array_equal(positions[:, 0], np.arange(len(ids)))
            np.testing.assert_array_equal(masks, ids != 0)
            batches.append(
                list(
                    zip(
                        np.asarray(ids[:, 0]).tolist(),
                        np.asarray(positions[:, 1]).tolist(),
                        np.asarray(targets).tolist(),
                        strict=True,
                    )
                )
            )
            result = step(params, state, ids, positions, targets, masks)
            outgoing.append(result[:2])
            return result

        return update

    monkeypatch.setattr(training, "make_train_step", capture_step)
    frozen_before = jax.tree.map(np.copy, tiny_run.frozen)
    substrate = tiny_run.model.continuous.sub
    base_before = substrate.weights.copy()
    verify = Mock(wraps=substrate.verify_frozen)
    monkeypatch.setattr(substrate, "verify_frozen", verify)
    with (
        caplog.at_level("INFO", logger="stage_a4"),
        RunWriter(tmp_path, "sequential") as writer,
    ):
        report = training.train_sequential(
            tiny_run.model, tiny_run.frozen, data, tiny_run.bank, config, writer
        )
    verify.assert_called_once()
    assert report["frozen"]["params_unchanged"]
    np.testing.assert_array_equal(substrate.weights, base_before)
    jax.tree.map(np.testing.assert_array_equal, tiny_run.frozen, frozen_before)
    jax.tree.map(
        np.testing.assert_array_equal,
        load_params(writer.dir / "params/a3-frozen.npz"),
        frozen_before,
    )

    assert [len(batch) for batch in batches] == [2, 1, 2, 1]
    assert Counter(example for batch in batches for example in batch) == Counter(
        [(1, 1, 1), (1, 2, 0), (1, 3, 1), (2, 1, 1), (2, 2, 0), (3, 2, 0)]
    )
    for index in (1, 2, 3):
        jax.tree.map(
            np.testing.assert_array_equal, incoming[index][0], outgoing[index - 1][0]
        )
        if reset_optimizer and index in (2, 3):
            for leaf in jax.tree.leaves(incoming[index][1]):
                np.testing.assert_array_equal(leaf, np.zeros_like(leaf))
        else:
            jax.tree.map(
                np.testing.assert_array_equal,
                incoming[index][1],
                outgoing[index - 1][1],
            )

    assert report["updates"] == 4
    interpretation = report["continual_evaluation"]["interpretation"]
    assert "three fixed held-out evaluation groups" in interpretation
    assert caplog.text.count("Continual Learning Metrics:") == 1
    assert caplog.text.index("Validation after Task 3/3") < caplog.text.index(
        interpretation
    )
    assert np.asarray(report["loss_matrix"]).shape == (3, 3)
    assert np.asarray(report["perplexity_matrix"]).shape == (3, 3)
    assert report["mean_forgetting"][0] is None
    assert len(report["mean_forgetting"]) == len(report["mean_seen_loss"]) == 3
    assert np.isfinite(report["backward_transfer_ce"])
    assert json.loads((writer.dir / "results.json").read_text()) == report
    assert len(report["after_task"]) == 3
    assert all(
        set(row) == {"shared_validation", *ROCSTORIES_EVAL_GROUPS}
        for row in [report["initial"], *report["after_task"]]
    )
    assert report["after_task"][-1]["shared_validation"]["tokens"] == 2
    assert report["test"]["shared_test"]["tokens"] == 3
    for index, (path, updates) in enumerate(
        zip(report["checkpoints"], (2, 3, 4, 4), strict=True)
    ):
        # These checkpoints were just written by the fixture.
        weights, state, metadata = load_checkpoint(writer.dir / path, trusted=True)
        expected_params, expected_state = outgoing[updates - 1]
        jax.tree.map(
            np.testing.assert_array_equal, unpack_head_params(weights), expected_params
        )
        jax.tree.map(np.testing.assert_array_equal, state, expected_state)
        assert metadata["updates"] == updates
        assert metadata["data_mode"] == "sequential-aligned"
        assert metadata["training_state_source"] == TRAINING_STATE_SOURCE
        assert metadata["reset_optimizer"] is reset_optimizer
        assert metadata["bank_snapshot_sha256"] == tiny_run.bank.input_snapshot_sha256
        assert metadata["frozen_residual_sha256"] == report["frozen_residual_sha256"]
        assert metadata["task_order"] == data.provenance["task_order"]
        assert metadata["source_fingerprints"] == data.provenance["source_fingerprints"]
        assert metadata["alignment_counts"] == data.provenance["alignment_counts"]
        assert metadata["loss_weights"] == {"ce": 1.0, "key": 0.6, "value": 1.4}
        if index < len(data.tasks):
            name = data.tasks[index].name
            assert (metadata["task_index"], metadata["task"]) == (index, name)
            assert metadata["task_metrics"] == report["task_metrics"][name]
            assert metadata["task_metrics"]["examples_processed"] == len(
                data.tasks[index].train_blocks
            )
        else:
            assert metadata["task_metrics"] == report["task_metrics"]


def test_sequential_aligned_cli_runs_trainer_with_mocked_native_dependencies(
    tmp_path, monkeypatch, tiny_run, prepared_tasks
):
    from runtime import data_loader
    from stages.stage_A import a3
    from stages.stage_A.a4 import checkpoint, config, data
    from symbolic_memory import bank, native_bridge

    args, prepared = prepared_tasks
    continuous = tiny_run.model.continuous
    resolve = Mock(return_value=continuous.sub)
    load_alignment = Mock(wraps=data.load_data)
    restore = Mock(return_value=tiny_run.frozen)
    native = Mock(return_value=nullcontext(tiny_run.retrieve))
    train_sequential = Mock(wraps=training.train_sequential)
    unexpected = Mock(side_effect=AssertionError("unexpected training path"))
    monkeypatch.setattr(runner, "_resolve_substrate", resolve)
    monkeypatch.setattr(a3, "A3Model", Mock(return_value=continuous))
    monkeypatch.setattr("residual.ResidualAdapter", Mock(return_value=continuous))
    monkeypatch.setattr(config, "load_config", Mock(return_value=tiny_run.config))
    monkeypatch.setattr(checkpoint, "restore_a3_weights", restore)
    monkeypatch.setattr(
        "stages.results.load_params", Mock(return_value=tiny_run.frozen)
    )
    monkeypatch.setattr(bank, "load_bank", Mock(return_value=tiny_run.bank))
    monkeypatch.setattr(data, "load_data", load_alignment)
    monkeypatch.setattr(native_bridge.NativeJaxRetriever, "from_mork", native)
    monkeypatch.setattr(data_loader, "load_runtime_data", unexpected)
    monkeypatch.setattr(training, "train", unexpected)
    monkeypatch.setattr(training, "train_sequential", train_sequential)

    files = {}
    for flag, name in {
        "warm-start": "a3.npz",
        "patterns": "patterns.jsonl",
        "embeddings": "embeddings.jsonl",
        "native-library": "retrieval.so",
    }.items():
        files[flag] = tmp_path / name
        files[flag].touch()
    (tmp_path / "retrieval.so.json").write_text("{}", encoding="utf-8")
    files.update(
        {
            "train-sources": args.train_sources,
            "validation-sources": args.validation_sources,
            "test-sources": args.test_sources,
            "alignment-labels": args.alignment_labels,
        }
    )
    argv = [
        "--data-mode",
        "sequential-aligned",
        "--run-name",
        "cli",
        "--output-dir",
        str(tmp_path),
        "--minimum-examples",
        str(args.minimum_examples),
        "--max-eval-rows",
        str(args.max_eval_rows),
        "--max-eval-blocks",
        str(args.max_eval_blocks),
    ]
    for flag, path in files.items():
        argv.extend((f"--{flag}", str(path)))

    assert runner.main(argv) == 0
    unexpected.assert_not_called()
    resolve.assert_called_once_with("tiny", None, revision=None)
    restore.assert_called_once_with(continuous, tiny_run.frozen)
    load_alignment.assert_called_once()
    assert load_alignment.call_args.args[0].data_mode == "sequential-aligned"
    assert load_alignment.call_args.args[0].task_eval_sources is None
    assert load_alignment.call_args.args[1] is continuous.sub.tokenizer
    assert load_alignment.call_args.args[2] is tiny_run.bank
    native.assert_called_once_with(
        tiny_run.bank,
        server_url="http://127.0.0.1:18000",
        library_path=files["native-library"],
        top_m=2,
        max_queries=1024,
    )
    train_sequential.assert_called_once()
    assert train_sequential.call_args.args[2].provenance == prepared.provenance
    output = tmp_path / "cli"
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    log_text = (output / "training.log").read_text()
    assert log_text.count("Validation after Task") == 3
    assert log_text.count("Continual Learning Metrics:") == 1
    assert json.loads((output / "data.json").read_text()) == prepared.provenance
    meta = json.loads((output / "meta.json").read_text())
    assert meta["data_mode"] == "sequential-aligned"
    assert meta["retrieval_backend"] == "native"
    assert meta["retrieval"] == tiny_run.retrieve.info
    assert meta["loss_weights"] == {"ce": 1.0, "key": 0.6, "value": 1.4}
    assert meta["task_order"] == prepared.provenance["task_order"]
    assert meta["continual_evaluation"]["scope"] == "fixed_held_out_groups"
    report = json.loads((output / "results.json").read_text())
    assert report["updates"] == 4
    assert report["initial"]["shared_validation"]["tokens"] == 2
    assert meta["continual_evaluation"]["group_order"] == list(ROCSTORIES_EVAL_GROUPS)
    assert all(report["initial"][name]["tokens"] > 0 for name in ROCSTORIES_EVAL_GROUPS)
    assert np.asarray(report["loss_matrix"]).shape == (3, 3)
    assert all((output / path).is_file() for path in report["checkpoints"])
