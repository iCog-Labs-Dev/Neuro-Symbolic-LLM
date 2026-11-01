"""Tests for configurable JAX JIT compilation in stage training and evaluation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig, layer_key
from stages.results import RunWriter
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_base import (
    Domain,
    TrainingConfig,
    evaluate,
    load_configs,
    make_eval_set,
    run_sequential,
)
from tests.frozenllm.conftest import NUM_TOKENS, SEQ, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def runner() -> Any:
    path = REPO_ROOT / "experiments" / "run_stage_a1.py"
    spec = importlib.util.spec_from_file_location("run_stage_a1_jit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


# ── 1. TrainingConfig validation ──────────────────────────────────────────────


def test_training_config_jit_validation() -> None:
    # Default is True
    cfg = TrainingConfig()
    assert cfg.jit is True

    # Explicit True and False
    assert TrainingConfig(jit=True).jit is True
    assert TrainingConfig(jit=False).jit is False

    # Non-boolean raises ValueError
    with pytest.raises(ValueError, match="jit must be a boolean"):
        TrainingConfig(jit="true")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="jit must be a boolean"):
        TrainingConfig(jit=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="jit must be a boolean"):
        TrainingConfig(jit=None)  # type: ignore[arg-type]

    # from_dict deserialisation
    assert TrainingConfig.from_dict({"jit": False}).jit is False
    assert TrainingConfig.from_dict({"jit": True}).jit is True
    with pytest.raises(ValueError, match="jit must be a boolean"):
        TrainingConfig.from_dict({"jit": "false"})


# ── 2. CLI --no-jit override ──────────────────────────────────────────────────


def test_cli_no_jit_flag(runner: Any) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml", A1Config
    )
    assert cfg.training.jit is True

    # Default without flag keeps jit=True
    args_default = runner.parse_args([])
    assert args_default.no_jit is False
    overridden_default = runner.apply_overrides(cfg, args_default)
    assert overridden_default.training.jit is True

    # Passing --no-jit sets jit=False
    args_no_jit = runner.parse_args(["--no-jit"])
    assert args_no_jit.no_jit is True
    overridden_no_jit = runner.apply_overrides(cfg, args_no_jit)
    assert overridden_no_jit.training.jit is False


# ── 3. Numerical Equivalence: evaluate() jit=True vs jit=False ───────────────


def test_evaluate_numerical_equivalence(substrate: FrozenSubstrate) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    params = adapter.init_params()

    rng = np.random.default_rng(42)
    blocks = rng.integers(0, NUM_TOKENS, (4, SEQ), dtype=np.int32)
    eval_batches = make_eval_set(blocks, batch_size=2, eval_batches=2)

    res_jit = evaluate(substrate, adapter, params, eval_batches, jit=True)
    res_nojit = evaluate(substrate, adapter, params, eval_batches, jit=False)

    for key in ("base_loss", "loss", "kl", "base_ppl", "ppl"):
        assert res_jit[key] == pytest.approx(
            res_nojit[key], rel=1e-5, abs=1e-5
        ), f"Mismatch on {key}: jit={res_jit[key]} vs nojit={res_nojit[key]}"


# ── 4. Numerical Equivalence: run_sequential() jit=True vs jit=False ─────────


def test_run_sequential_numerical_equivalence(
    substrate: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
    init_params = adapter.init_params()

    rng = np.random.default_rng(123)
    blocks = rng.integers(0, NUM_TOKENS, (4, SEQ), dtype=np.int32)
    eval_set = make_eval_set(blocks, batch_size=2, eval_batches=1)

    domain_jit = Domain("d", blocks, eval_set, epochs=1)
    domain_nojit = Domain("d", blocks, eval_set, epochs=1)

    cfg_jit = TrainingConfig(
        batch_size=2, learning_rate=1e-2, log_every=1, seed=0, jit=True
    )
    cfg_nojit = TrainingConfig(
        batch_size=2, learning_rate=1e-2, log_every=1, seed=0, jit=False
    )

    with RunWriter(tmp_path / "jit", "run") as w_jit:
        params_jit, summary_jit = run_sequential(
            substrate,
            adapter,
            init_params,
            [domain_jit],
            a1_objective,
            A1Config(),
            cfg_jit,
            w_jit,
        )

    with RunWriter(tmp_path / "nojit", "run") as w_nojit:
        params_nojit, summary_nojit = run_sequential(
            substrate,
            adapter,
            init_params,
            [domain_nojit],
            a1_objective,
            A1Config(),
            cfg_nojit,
            w_nojit,
        )

    # Check loss summaries match
    np.testing.assert_allclose(
        np.asarray(summary_jit["loss"]),
        np.asarray(summary_nojit["loss"]),
        rtol=1e-4,
        atol=1e-4,
    )

    # Check updated adapter parameters match
    for idx in adapter.layers:
        key = layer_key(idx)
        for p_name in ("A", "B"):
            np.testing.assert_allclose(
                np.asarray(params_jit[key][p_name]),
                np.asarray(params_nojit[key][p_name]),
                rtol=1e-4,
                atol=1e-4,
            )
