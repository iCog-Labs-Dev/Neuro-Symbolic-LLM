"""Stage A1 sequential runner: configs, training loop and run artifacts."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from metrics.performance import cross_entropy_loss
from residual import AdapterParams, ResidualAdapter, ResidualConfig, layer_key
from stages.results import RunWriter, load_params
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_base import (
    Domain,
    TrainingConfig,
    load_configs,
    make_eval_set,
    run_sequential,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, SEQ, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def runner() -> Any:
    """Load the A1 runner as a script module."""
    path = REPO_ROOT / "experiments" / "run_stage_a1.py"
    spec = importlib.util.spec_from_file_location("run_stage_a1_seq", path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    yield module

    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


def _domain(
    name: str,
    token: int,
    epochs: int = 1,
    num_blocks: int = 8,
) -> Domain:
    """Create a learnable synthetic sequential task."""
    rng = np.random.default_rng(token)

    blocks = np.full(
        (num_blocks, SEQ),
        token,
        dtype=np.int32,
    )

    noise = rng.random(blocks.shape) < 0.1
    blocks[noise] = rng.integers(
        0,
        NUM_TOKENS,
        int(noise.sum()),
    )

    return Domain(
        name=name,
        train_blocks=blocks,
        eval_set=make_eval_set(blocks, BATCH, 1),
        epochs=epochs,
    )


# ---------------------------------------------------------------------
# Sequential training and artifacts
# ---------------------------------------------------------------------


def test_run_sequential_writes_matrix_and_artifacts(
    substrate: FrozenSubstrate,
    tmp_path: Path,
) -> None:
    adapter = ResidualAdapter(
        ResidualConfig(rank=4),
        substrate.architecture,
    )

    domains = [
        _domain("a", 5, epochs=1, num_blocks=8),
        _domain("b", 9, epochs=1, num_blocks=6),
    ]

    training = TrainingConfig(
        batch_size=BATCH,
        learning_rate=1e-2,
        log_every=1,
    )

    with RunWriter(tmp_path, "run") as writer:
        params, summary = run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            domains,
            a1_objective,
            A1Config(),
            training,
            writer,
        )

    assert summary["domains"] == ["a", "b"]

    loss = np.asarray(summary["loss"])

    assert loss.shape == (2, 2)
    assert np.all(np.isfinite(loss))

    assert loss[0, 0] < summary["base_loss"][0]
    assert summary["improvement"][0][0] > 0

    assert summary["mean_seen_loss"][0] == pytest.approx(loss[0, 0])

    assert summary["mean_seen_loss"][1] == pytest.approx(loss[1].mean())

    assert summary["mean_forgetting"][0] is None

    assert summary["mean_forgetting"][1] == pytest.approx(loss[1, 0] - loss[0, 0])

    assert summary["steps"] == 7
    assert summary["tokens"] == 7 * BATCH * SEQ
    assert summary["epochs"] == {"a": 1, "b": 1}

    run = tmp_path / "run"

    events = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]

    train = [event for event in events if event["event"] == "train"]

    evals = [event for event in events if event["event"] == "eval"]

    epoch_events = [event for event in events if event["event"] == "epoch"]

    assert len(epoch_events) == 2

    assert [event["global_step"] for event in train] == list(range(1, 8))

    assert [event["epoch"] for event in train] == [1, 1, 1, 1, 1, 1, 1]

    assert [event["batch"] for event in train] == [1, 2, 3, 4, 1, 2, 3]

    assert {
        "loss_total",
        "loss_task",
        "loss_wd",
        "step_time",
        "ppl",
    } <= set(train[0])

    # KL is omitted from training when lambda_kl=0,
    # but evaluation continues reporting KL.
    assert all("loss_kl" not in event for event in train)
    assert all("kl" in event for event in evals)

    assert [(event["after_task"], event["domain"]) for event in evals] == [
        (task, domain) for task in (-1, 0, 1) for domain in ("a", "b")
    ]

    # Initial adapted model equals the base model.
    for event in evals[:2]:
        assert event["loss"] == pytest.approx(
            event["base_loss"],
            abs=1e-5,
        )

    saved = load_params(run / "params" / "task_1_b.npz")

    for idx in adapter.layers:
        key = layer_key(idx)

        np.testing.assert_allclose(
            saved[key]["B"],
            np.asarray(params[key]["B"]),
        )

    assert (run / "params" / "task_0_a.npz").exists()
    assert substrate.params_unchanged()


def test_run_sequential_logs_only_returned_terms(
    substrate: FrozenSubstrate,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The training loop must not assume optional KL or WD terms."""

    def task_only(
        params: AdapterParams,
        *,
        substrate: FrozenSubstrate,
        adapter: ResidualAdapter,
        config: Any,
        input_ids: jax.Array,
        labels: jax.Array | None = None,
        base: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits = substrate.run_with_interception(
            input_ids,
            modify_fn=adapter.modify_fn(params),
            intercept_layers=adapter.layers,
        ).logits

        task = cross_entropy_loss(logits, input_ids)

        return task, {
            "total": task,
            "task": task,
        }

    adapter = ResidualAdapter(
        ResidualConfig(rank=4),
        substrate.architecture,
    )

    training = TrainingConfig(
        batch_size=BATCH,
        log_every=1,
    )

    with (
        caplog.at_level(logging.INFO),
        RunWriter(tmp_path, "run") as writer,
    ):
        run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            [_domain("a", 5, epochs=1, num_blocks=4)],
            task_only,
            None,
            training,
            writer,
        )

    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]

    train = [event for event in events if event["event"] == "train"]

    assert len(train) == 2

    assert {key for key in train[0] if key.startswith("loss_")} == {
        "loss_total",
        "loss_task",
    }

    batch_lines = [
        record.getMessage()
        for record in caplog.records
        if "Batch " in record.getMessage()
    ]

    assert batch_lines
    assert all("task_loss=" in message for message in batch_lines)

    assert not any("step=" in message for message in batch_lines)
    assert not any("kl=" in message or "wd=" in message for message in batch_lines)


# ---------------------------------------------------------------------
# Configuration serialization
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter",
    [
        ResidualConfig(),
        ResidualConfig(rank=4, layers=[2, 7, 11]),
    ],
)
def test_config_round_trips_through_writer(
    tmp_path: Path,
    adapter: ResidualConfig,
) -> None:
    """The saved config must remain loadable by the stage config loader."""

    shipped = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        A1Config,
    )

    # Dataset definitions now belong to runtime_datasets.yaml.
    assert shipped.domains == ()

    cfg = replace(
        shipped,
        adapter=adapter,
        objective=A1Config(lambda_kl=0.5),
    )

    with RunWriter(tmp_path, "run") as writer:
        path = writer.write_config(cfg.to_dict())

    assert load_configs(path, A1Config) == cfg


def test_config_without_layers_key_still_loads(
    tmp_path: Path,
) -> None:
    """Older saved configs without the layers key remain supported."""

    path = tmp_path / "old_config.yaml"

    path.write_text(
        "adapter:\n"
        "  rank: 8\n"
        "  late_start: 8\n"
        "  late_end: 9\n"
        "objective: {lambda_kl: 0.0, lambda_wd: 0.0}\n"
        "domains:\n"
        "  - {name: a, dataset: x}\n"
    )

    cfg = load_configs(path, A1Config)

    assert cfg.adapter == ResidualConfig(
        rank=8,
        late_start=8,
        late_end=9,
    )

    assert cfg.adapter.layers is None


def test_epochs_override_applies_to_training_config(
    runner: Any,
) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        A1Config,
    )

    args = runner.parse_args(["--epochs", "2", "--rank", "4"])
    out = runner.apply_overrides(cfg, args)

    assert out.training.epochs == 2
    assert out.domains == ()
    assert out.adapter.rank == 4


def test_deprecated_steps_arg_raises_error(
    runner: Any,
) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        A1Config,
    )

    args = runner.parse_args(["--steps", "2"])

    with pytest.raises(ValueError, match="Use --epochs"):
        runner.apply_overrides(cfg, args)

    args_conflict = runner.parse_args(["--steps", "2", "--epochs", "3"])

    with pytest.raises(ValueError, match="Cannot specify both"):
        runner.apply_overrides(cfg, args_conflict)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "training: {steps: 1}\n",
            "Unknown TrainingConfig",
        ),
        (
            "domains:\n" "  - {name: a, dataset: x}\n" "  - {name: a, dataset: y}\n",
            "unique",
        ),
        (
            "domains:\n" "  - {name: a/b, dataset: x}\n",
            "Domain name",
        ),
        (
            "domains:\n" "  - {name: a, dataset: x, split: train}\n",
            "Unknown DomainConfig",
        ),
        (
            "training: {dataset: x}\n" "domains:\n" "  - {name: a, dataset: x}\n",
            "Unknown TrainingConfig",
        ),
    ],
)
def test_invalid_domain_configs_raise(
    tmp_path: Path,
    body: str,
    match: str,
) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(body)

    with pytest.raises(ValueError, match=match):
        load_configs(path, A1Config)


def test_empty_domains_are_allowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("domains: []\n")

    cfg = load_configs(path, A1Config)

    assert cfg.domains == ()


# ---------------------------------------------------------------------
# Training overrides and evaluation configuration
# ---------------------------------------------------------------------


def test_eval_batches_override_applies(
    runner: Any,
) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        A1Config,
    )

    args = runner.parse_args(["--eval-batches", "4"])
    out = runner.apply_overrides(cfg, args)

    assert out.training.eval_batches == 4

    args_none = runner.parse_args([])
    out_none = runner.apply_overrides(cfg, args_none)

    assert out_none.training.eval_batches == cfg.training.eval_batches


@pytest.mark.parametrize("val", [0, -1, -5])
def test_eval_batches_rejects_values_below_one(
    runner: Any,
    val: int,
) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        A1Config,
    )

    args = runner.parse_args(["--eval-batches", str(val)])

    with pytest.raises(ValueError, match="eval_batches must be >= 1"):
        runner.apply_overrides(cfg, args)


@pytest.mark.parametrize("val", [0, -1, -5])
def test_make_eval_set_rejects_values_below_one(
    val: int,
) -> None:
    blocks = np.zeros((10, 128), dtype=np.int32)

    with pytest.raises(ValueError, match="eval_batches must be >= 1"):
        make_eval_set(
            blocks,
            batch_size=2,
            eval_batches=val,
        )
