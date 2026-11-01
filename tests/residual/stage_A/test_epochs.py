"""Unit and integration tests for epoch-based training semantics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig, layer_key
from stages.results import RunWriter, load_checkpoint
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_base import (
    Domain,
    DomainConfig,
    RunConfig,
    TrainingConfig,
    batches,
    make_eval_set,
    run_sequential,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, SEQ, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def runner() -> Any:
    path = REPO_ROOT / "experiments" / "run_stage_a1.py"
    spec = importlib.util.spec_from_file_location("run_stage_a1_epochs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _make_domain(name: str, token: int, epochs: int = 1, num_blocks: int = 8) -> Domain:
    rng = np.random.default_rng(token)
    blocks = np.full((num_blocks, SEQ), token, dtype=np.int32)
    noise = rng.random(blocks.shape) < 0.1
    blocks[noise] = rng.integers(0, NUM_TOKENS, int(noise.sum()))
    return Domain(
        name=name,
        train_blocks=blocks,
        eval_set=make_eval_set(blocks, BATCH, 1),
        epochs=epochs,
    )


# ── Requirement 1 & 14: One epoch consumes every batch once, reproducible ────


def test_batches_consumes_every_complete_batch_once() -> None:
    # 10 blocks numbered 0..9, batch_size=2 -> exactly 5 batches
    blocks = np.arange(10)[:, None]
    collected = list(batches(blocks, batch_size=2, seed=42))
    assert len(collected) == 5

    seen = np.concatenate([np.asarray(b).ravel() for b in collected])
    assert set(seen) == set(range(10))
    assert len(seen) == 10


def test_batches_reproducibility_same_seed() -> None:
    blocks = np.arange(20)[:, None]
    run1 = [np.asarray(b) for b in batches(blocks, batch_size=4, seed=123)]
    run2 = [np.asarray(b) for b in batches(blocks, batch_size=4, seed=123)]
    run3 = [np.asarray(b) for b in batches(blocks, batch_size=4, seed=999)]

    for b1, b2 in zip(run1, run2, strict=True):
        np.testing.assert_array_equal(b1, b2)

    # Different seed produces a different permutation
    assert not all(np.array_equal(b1, b3) for b1, b3 in zip(run1, run3, strict=True))


# ── Requirement 2: Two epochs consume dataset twice ──────────────────────────


def test_batches_two_epochs_consume_dataset_twice() -> None:
    blocks = np.arange(8)[:, None]
    epoch1 = list(batches(blocks, batch_size=2, seed=100))
    epoch2 = list(batches(blocks, batch_size=2, seed=101))

    assert len(epoch1) == 4
    assert len(epoch2) == 4

    seen1 = np.concatenate([np.asarray(b).ravel() for b in epoch1])
    seen2 = np.concatenate([np.asarray(b).ravel() for b in epoch2])
    assert set(seen1) == set(range(8))
    assert set(seen2) == set(range(8))


# ── Requirement 3: Trailing incomplete batch is dropped ──────────────────────


def test_batches_drops_incomplete_tail_by_default() -> None:
    # 11 blocks, batch_size=3 -> 3 complete batches (9 blocks), 2 dropped
    blocks = np.arange(11)[:, None]
    collected = list(batches(blocks, batch_size=3, seed=42))
    assert len(collected) == 3
    for b in collected:
        assert b.shape == (3, 1)


# ── Requirement 4: Invalid dataset size raises ValueError ────────────────────


def test_batches_raises_on_invalid_dataset_size() -> None:
    # 3 blocks with batch_size=4
    blocks = np.arange(3)[:, None]
    with pytest.raises(ValueError, match="cannot fill batch_size"):
        list(batches(blocks, batch_size=4, seed=42))


# ── Requirement 5: Zero epochs perform zero optimizer updates ────────────────


def test_zero_epochs_performs_no_updates(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    init_params = adapter.init_params()
    domains = [_make_domain("empty", 5, epochs=0, num_blocks=8)]
    training = TrainingConfig(batch_size=BATCH, epochs=0)

    with RunWriter(tmp_path, "run") as writer:
        params, summary = run_sequential(
            substrate,
            adapter,
            init_params,
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    assert summary["steps"] == 0
    assert summary["tokens"] == 0
    assert summary["epochs"] == {"empty": 0}

    # Initial evaluation is still performed
    assert len(summary["base_loss"]) == 1
    assert summary["loss"][0][0] == pytest.approx(summary["base_loss"][0], abs=1e-5)

    # Checkpoint saved with initial params
    ckpt_path = tmp_path / "run" / "checkpoints" / "task_0_empty.npz"
    assert ckpt_path.exists()
    _, _, meta = load_checkpoint(ckpt_path, trusted=True)
    assert meta["global_step"] == 0
    assert meta["epoch"] == 0

    # No train events logged
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert not any(e["event"] == "train" for e in events)
    assert any(e["event"] == "eval" for e in events)


# ── Requirement 6 & 12: Global update count and checkpoint metadata ──────────


def test_global_update_count_and_checkpoint_metadata(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    # Domain A: 8 blocks, batch 2 -> 4 batches/epoch * 2 epochs = 8 updates
    # Domain B: 6 blocks, batch 2 -> 3 batches/epoch * 1 epoch = 3 updates
    domains = [
        _make_domain("dom_a", 5, epochs=2, num_blocks=8),
        _make_domain("dom_b", 9, epochs=1, num_blocks=6),
    ]
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)

    with RunWriter(tmp_path, "run") as writer:
        _, summary = run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    expected_steps = (4 * 2) + (3 * 1)  # 11
    assert summary["steps"] == expected_steps
    assert summary["tokens"] == expected_steps * BATCH * SEQ
    assert summary["epochs"] == {"dom_a": 2, "dom_b": 1}

    run_dir = tmp_path / "run"
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    train_events = [e for e in events if e["event"] == "train"]
    epoch_events = [e for e in events if e["event"] == "epoch"]

    assert len(train_events) == 11
    assert [e["global_step"] for e in train_events] == list(range(1, 12))
    assert [e["epoch"] for e in train_events] == [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]
    assert [e["batch"] for e in train_events] == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]

    # Exactly 3 epoch summary events (2 for dom_a, 1 for dom_b)
    assert len(epoch_events) == 3
    assert [(e["domain"], e["epoch"], e["total_epochs"]) for e in epoch_events] == [
        ("dom_a", 1, 2),
        ("dom_a", 2, 2),
        ("dom_b", 1, 1),
    ]
    for ep in epoch_events:
        assert ep["loss_total"] > 0
        assert ep["batches"] > 0
        assert ep["epoch_time"] >= 0

    # Verify checkpoint metadata
    _, _, meta_a = load_checkpoint(
        run_dir / "checkpoints" / "task_0_dom_a.npz", trusted=True
    )
    assert meta_a["domain"] == "dom_a"
    assert meta_a["epoch"] == 2
    assert meta_a["global_step"] == 8

    _, _, meta_b = load_checkpoint(
        run_dir / "checkpoints" / "task_1_dom_b.npz", trusted=True
    )
    assert meta_b["domain"] == "dom_b"
    assert meta_b["epoch"] == 1
    assert meta_b["global_step"] == 11


# ── Requirement 7: Domain-specific epoch overrides ───────────────────────────


def test_domain_specific_epoch_overrides() -> None:
    training = TrainingConfig(epochs=3)
    domain_default = DomainConfig(name="d1", dataset="x", epochs=None)
    domain_override = DomainConfig(name="d2", dataset="y", epochs=5)

    assert (domain_default.epochs or training.epochs) == 3
    assert (domain_override.epochs or training.epochs) == 5


# ── Requirement 8: CLI --epochs override applies consistently ────────────────


def test_cli_epochs_override_applies_consistently(runner: Any) -> None:
    cfg = RunConfig(
        adapter=ResidualConfig(),
        objective=A1Config(),
        training=TrainingConfig(epochs=1),
        domains=(
            DomainConfig(name="d1", dataset="x", epochs=2),
            DomainConfig(name="d2", dataset="y", epochs=5),
        ),
    )
    args = argparse.Namespace(
        config=None,
        model=None,
        rank=None,
        epochs=4,
        steps=None,
        output_dir=None,
        run_name=None,
        eval_checkpoint=None,
        split="test",
        eval_batches=None,
    )
    out = runner.apply_overrides(cfg, args)
    assert out.training.epochs == 4
    assert all(d.epochs == 4 for d in out.domains)


# ── Requirement 9 & 10: Optimizer state persistence and reset ────────────────


def test_optimizer_state_persists_across_epochs(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    # Train 2 epochs on one domain with reset_optimizer=False
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    domains = [_make_domain("a", 5, epochs=2, num_blocks=8)]
    training = TrainingConfig(
        batch_size=BATCH, learning_rate=1e-2, reset_optimizer=False
    )

    with RunWriter(tmp_path, "run") as writer:
        run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    _, opt_state, meta = load_checkpoint(
        tmp_path / "run" / "checkpoints" / "task_0_a.npz", trusted=True
    )
    assert meta["global_step"] == 8
    assert opt_state is not None
    # Adam count leaf equals total steps (8)
    adam_state = opt_state[0]
    assert int(adam_state.count) == 8


def test_optimizer_state_resets_between_domains_when_configured(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    domains = [
        _make_domain("a", 5, epochs=1, num_blocks=8),  # 4 steps
        _make_domain("b", 9, epochs=1, num_blocks=6),  # 3 steps
    ]
    training = TrainingConfig(
        batch_size=BATCH, learning_rate=1e-2, reset_optimizer=True
    )

    with RunWriter(tmp_path, "run") as writer:
        run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    # In task 1 (domain b), since reset_optimizer=True, Adam count reflects only domain b steps (3)
    _, opt_state_b, meta_b = load_checkpoint(
        tmp_path / "run" / "checkpoints" / "task_1_b.npz", trusted=True
    )
    assert meta_b["global_step"] == 7
    assert opt_state_b is not None
    adam_state_b = opt_state_b[0]
    assert int(adam_state_b.count) == 3


# ── Requirement 11: Adapter updated while substrate remains frozen ───────────


def test_adapter_updated_while_substrate_remains_frozen(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    init_params = adapter.init_params()
    domains = [_make_domain("a", 5, epochs=1, num_blocks=8)]
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2)

    with RunWriter(tmp_path, "run") as writer:
        final_params, _ = run_sequential(
            substrate,
            adapter,
            init_params,
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    # Adapter params have moved from init
    for idx in adapter.layers:
        key = layer_key(idx)
        assert float(jnp.abs(final_params[key]["B"]).max()) > 0.0

    # Substrate parameters strictly unchanged
    assert substrate.params_unchanged()
    assert substrate.verify_frozen()["params_unchanged"]


# ── Requirement 13: Continual learning evaluation metrics intact ─────────────


def test_continual_learning_summaries_intact(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    domains = [
        _make_domain("a", 5, epochs=1, num_blocks=8),
        _make_domain("b", 9, epochs=1, num_blocks=8),
    ]
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2)

    with RunWriter(tmp_path, "run") as writer:
        _, summary = run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    loss_mat = np.asarray(summary["loss"])
    assert loss_mat.shape == (2, 2)
    assert np.all(np.isfinite(loss_mat))
    assert len(summary["base_loss"]) == 2
    assert len(summary["mean_seen_loss"]) == 2
    assert summary["mean_forgetting"][0] is None
    assert summary["mean_forgetting"][1] is not None
    assert len(summary["improvement"]) == 2
