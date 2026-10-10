"""Stage A1 sequential runner: configs, the multi-domain loop and run artifacts.

The configs and the loop live in ``stages/stage_base.py``; the runner script only
parses arguments. Runs ``run_sequential`` on the 12-layer GPT-2 fixture with synthetic token
blocks, so no dataset download is needed.
"""

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
    # experiments/ is a script directory, not a package; load the runner by path.
    # It must be in sys.modules while executing so its dataclasses resolve.
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


def _domain(name: str, token: int, steps: int) -> Domain:
    # A constant-token domain is learnable in a few steps, so training on it
    # must lower its loss; two domains with different tokens compete.
    rng = np.random.default_rng(token)
    blocks = np.full((8, SEQ), token, dtype=np.int32)
    noise = rng.random(blocks.shape) < 0.1
    blocks[noise] = rng.integers(0, NUM_TOKENS, int(noise.sum()))
    return Domain(
        name=name,
        train_blocks=blocks,
        eval_set=make_eval_set(blocks, BATCH, 1),
        steps=steps,
    )


def test_run_sequential_writes_matrix_and_artifacts(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    domains = [_domain("a", 5, 4), _domain("b", 9, 3)]
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)

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
    # Training on a domain improves it over the base.
    assert loss[0, 0] < summary["base_loss"][0]
    assert summary["improvement"][0][0] > 0
    assert summary["mean_seen_loss"][0] == pytest.approx(loss[0, 0])
    assert summary["mean_seen_loss"][1] == pytest.approx(loss[1].mean())
    assert summary["mean_forgetting"][0] is None
    assert summary["mean_forgetting"][1] == pytest.approx(loss[1, 0] - loss[0, 0])
    assert summary["steps"] == 7
    assert summary["tokens"] == 7 * BATCH * SEQ

    run = tmp_path / "run"
    events = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    train = [e for e in events if e["event"] == "train"]
    evals = [e for e in events if e["event"] == "eval"]
    assert [e["global_step"] for e in train] == list(range(1, 8))
    assert {"loss_total", "loss_task", "loss_wd", "step_time", "ppl"} <= set(train[0])
    # lambda_kl = 0: the train objective omits KL, but evaluation still reports it.
    assert all("loss_kl" not in e for e in train)
    assert all("kl" in e for e in evals)
    # Every domain after init and after each of the 2 tasks.
    assert [(e["after_task"], e["domain"]) for e in evals] == [
        (t, d) for t in (-1, 0, 1) for d in ("a", "b")
    ]
    # At init the adapted model is the base model.
    for e in evals[:2]:
        assert e["loss"] == pytest.approx(e["base_loss"], abs=1e-5)

    saved = load_params(run / "params" / "task_1_b.npz")
    for idx in adapter.layers:
        key = layer_key(idx)
        np.testing.assert_allclose(saved[key]["B"], np.asarray(params[key]["B"]))
    assert (run / "params" / "task_0_a.npz").exists()
    assert substrate.params_unchanged()


def test_run_sequential_logs_only_returned_terms(
    substrate: FrozenSubstrate, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Another stage's objective need not return kl/wd; the loop must not assume them.
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
        return task, {"total": task, "task": task}

    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    training = TrainingConfig(batch_size=BATCH, log_every=1)
    with caplog.at_level(logging.INFO), RunWriter(tmp_path, "run") as writer:
        run_sequential(
            substrate,
            adapter,
            adapter.init_params(),
            [_domain("a", 5, 2)],
            task_only,
            None,
            training,
            writer,
        )

    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    train = [e for e in events if e["event"] == "train"]
    assert len(train) == 2
    assert {k for k in train[0] if k.startswith("loss_")} == {"loss_total", "loss_task"}
    step_lines = [r.getMessage() for r in caplog.records if " step=" in r.getMessage()]
    assert step_lines and all("task_loss=" in m for m in step_lines)
    assert not any("kl=" in m or "wd=" in m for m in step_lines)


@pytest.mark.parametrize(
    "adapter", [ResidualConfig(), ResidualConfig(rank=4, layers=[2, 7, 11])]
)
def test_config_round_trips_through_writer(
    tmp_path: Path, adapter: ResidualConfig
) -> None:
    # results.py promises config.yaml can be passed back to the runner.
    shipped = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml", A1Config
    )
    assert [d.name for d in shipped.domains] == ["eurlex", "ledgar", "scotus"]
    cfg = replace(shipped, adapter=adapter, objective=A1Config(lambda_kl=0.5))
    with RunWriter(tmp_path, "run") as writer:
        path = writer.write_config(cfg.to_dict())
    assert load_configs(path, A1Config) == cfg


def test_config_without_layers_key_still_loads(tmp_path: Path) -> None:
    # config.yaml files written before ``layers`` existed have no such key.
    path = tmp_path / "old_config.yaml"
    path.write_text(
        "adapter:\n  rank: 8\n  late_start: 8\n  late_end: 9\n"
        "objective: {lambda_kl: 0.0, lambda_wd: 0.0}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    cfg = load_configs(path, A1Config)
    assert cfg.adapter == ResidualConfig(rank=8, late_start=8, late_end=9)
    assert cfg.adapter.layers is None


def test_steps_override_applies_to_every_domain(runner: Any) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml", A1Config
    )
    args = runner.parse_args(["--steps", "2", "--rank", "4"])
    out = runner.apply_overrides(cfg, args)
    assert out.training.steps == 2
    assert all(d.steps == 2 for d in out.domains)
    assert out.adapter.rank == 4


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("training: {steps: 1}\n", "non-empty list"),
        ("domains: []\n", "non-empty list"),
        ("domains:\n  - {name: a, dataset: x}\n  - {name: a, dataset: y}\n", "unique"),
        ("domains:\n  - {name: a/b, dataset: x}\n", "Domain name"),
        ("domains:\n  - {name: a, dataset: x, split: train}\n", "Unknown DomainConfig"),
        (
            "training: {dataset: x}\ndomains:\n  - {name: a, dataset: x}\n",
            "Unknown TrainingConfig",
        ),
    ],
)
def test_invalid_domain_configs_raise(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=match):
        load_configs(path, A1Config)
