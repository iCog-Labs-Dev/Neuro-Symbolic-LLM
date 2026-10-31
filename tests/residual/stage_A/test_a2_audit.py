"""The A2 audit: the checkpoints pass on a correct objective, fail when it is broken, and
are reachable from the runner (``--audit`` / ``--audit-only``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stages.stage_A.a2 as a2
from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from stages.stage_A.a2 import (
    A2Config,
    Check,
    CPCConfig,
    HEPConfig,
    StabLossConfig,
    a2_audit,
    audit_ok,
    audit_to_dict,
    format_audit,
)
from tests.frozenllm.conftest import make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]
SEQ = 12
LAYERS = [5, 6]


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(
        ResidualConfig(rank=4, layers=LAYERS), substrate.architecture
    )


@pytest.fixture(scope="module")
def input_ids() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.integers(0, 64, (2, SEQ)), dtype=jnp.int32)


def full_cfg(**kw: Any) -> A2Config:
    return A2Config(
        lambda_kl=0.1,
        lambda_wd=0.01,
        cpc=CPCConfig(weight=0.05, dim=8),
        stab=StabLossConfig(
            weight=0.1, w_hep=0.1, hep=HEPConfig(kind="lowrank", rank=4)
        ),
        **kw,
    )


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_a_correct_objective_passes_every_checkpoint(
    substrate, adapter, input_ids
) -> None:
    checks = a2_audit(substrate, adapter, full_cfg(), input_ids)
    failed = [c for c in checks if c.ok is False]
    assert not failed, format_audit(checks)
    assert audit_ok(checks)
    names = set(by_name(checks))
    assert {
        "param_groups",
        "frozen_before",
        "init_equals_base",
        "finite_init",
        "finite_probe",
        "total_decomposition",
        "a1_parity",
        "grad_routing_task",
        "grad_routing_cpc",
        "grad_routing_stab_anch",
        "grad_routing_stab_hep",
        "jit_equals_eager",
        "finite_with_ignored_labels",
        "frozen_after",
    } <= names


def test_gradient_routing_values_match_the_design(
    substrate, adapter, input_ids
) -> None:
    c = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    task = c["grad_routing_task"].value
    assert task["adapter"] > 0 and task["cpc"] == 0 and task["hep"] == 0
    cpc = c["grad_routing_cpc"].value
    assert cpc["adapter"] > 0 and cpc["cpc"] > 0 and cpc["hep"] == 0
    anch = c["grad_routing_stab_anch"].value
    assert anch["adapter"] > 0 and anch["cpc"] == 0 and anch["hep"] == 0
    hep = c["grad_routing_stab_hep"].value
    assert hep["adapter"] == 0 and hep["hep"] > 0  # the HEP term trains only V


@pytest.mark.parametrize("kind", ["dense", "lowrank"])
def test_zero_alpha_highway_control_passes_audit(substrate, adapter, input_ids, kind):
    cfg = full_cfg()
    cfg = replace(
        cfg, stab=replace(cfg.stab, hep=HEPConfig(kind=kind, rank=4, alpha=0.0))
    )
    checks = a2_audit(substrate, adapter, cfg, input_ids)
    assert audit_ok(checks), format_audit(checks)
    check = by_name(checks)
    assert check["grad_finite_stab_hep"].ok
    assert all(v == 0 for v in check["grad_routing_stab_hep"].value.values())


def test_gradient_ratios_are_information_not_failures(
    substrate, adapter, input_ids
) -> None:
    c = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    for name in ("grad_ratio_cpc", "grad_ratio_stab_anch"):
        assert c[name].ok is None and c[name].value > 0


def test_only_the_enabled_terms_are_audited(substrate, adapter, input_ids) -> None:
    off = by_name(a2_audit(substrate, adapter, A2Config(lambda_kl=0.1), input_ids))
    assert not any(k.startswith("grad_ratio") for k in off)
    assert "grad_routing_cpc" not in off and "finite_with_ignored_labels" not in off
    assert audit_ok(list(off.values()))
    # A fixed highway is a metric: its HEP gradient reaches nothing.
    cfg = A2Config(stab=StabLossConfig(weight=0.1, hep=HEPConfig("fixed_random", 4)))
    fixed = by_name(a2_audit(substrate, adapter, cfg, input_ids))
    assert fixed["grad_routing_stab_hep"].ok is True
    assert fixed["grad_routing_stab_hep"].value["hep"] == 0
    assert "grad_routing_cpc" not in fixed


def test_a_wrong_total_is_caught(substrate, adapter, input_ids, monkeypatch) -> None:
    real = a2.a2_objective

    def off_by_one(params, **kw):
        total, terms = real(params, **kw)
        return total + 1.0, {**terms, "total": total + 1.0}

    monkeypatch.setattr(a2, "a2_objective", off_by_one)
    checks = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    assert checks["total_decomposition"].ok is False
    assert not audit_ok(list(checks.values()))


def test_a_leak_into_the_adapter_is_caught(
    substrate, adapter, input_ids, monkeypatch
) -> None:
    real = a2.a2_objective

    def leaky(params, **kw):
        total, terms = real(params, **kw)
        # The HEP term is made to depend on the adapter: it must no longer pass routing.
        leak = 1e-3 * sum(jnp.sum(v["B"]) for v in a2.adapter_params(params).values())
        if "stab_hep" in terms:
            terms = {**terms, "stab_hep": terms["stab_hep"] + leak}
        return total, terms

    monkeypatch.setattr(a2, "a2_objective", leaky)
    checks = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    assert checks["grad_routing_stab_hep"].ok is False


def test_non_finite_terms_are_caught(
    substrate, adapter, input_ids, monkeypatch
) -> None:
    real = a2.a2_objective

    def nan_cpc(params, **kw):
        total, terms = real(params, **kw)
        if "cpc" in terms:
            terms = {**terms, "cpc": terms["cpc"] * jnp.nan}
        return total, terms

    monkeypatch.setattr(a2, "a2_objective", nan_cpc)
    checks = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    assert checks["finite_init"].ok is False and "cpc" in checks["finite_init"].detail


def test_report_formats(substrate, adapter, input_ids) -> None:
    checks = a2_audit(substrate, adapter, full_cfg(), input_ids)
    text = format_audit(checks)
    assert "[PASS] a1_parity" in text and "[info] grad_ratio_cpc" in text
    data = audit_to_dict(checks)
    assert data["ok"] is True and len(data["checks"]) == len(checks)
    json.dumps(data)  # serialisable
    broken = [Check("x", False, 1.0, "why")]
    assert "[FAIL] x" in format_audit(broken) and "-> why" in format_audit(broken)
    assert audit_to_dict(broken)["ok"] is False


# --- runner --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner() -> Any:
    path = REPO_ROOT / "experiments" / "run_stage_a2.py"
    spec = importlib.util.spec_from_file_location("run_stage_a2_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        sys.modules.pop(spec.name, None)
        pytest.skip("experiments/run_stage_a2.py is not implemented yet (runner PR)")
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def setup(runner, substrate, tmp_path, monkeypatch) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "adapter: {rank: 4, layers: [5, 6]}\n"
        "objective:\n  lambda_kl: 0.1\n"
        "  cpc: {weight: 0.2, dim: 8}\n"
        "  stab: {weight: 0.2, hep: {kind: fixed_random, rank: 4}}\n"
        f"training: {{model: gpt2, seq_len: {SEQ}, batch_size: 2, epochs: 1, "
        "learning_rate: 1.0e-2, eval_batches: 1, log_every: 1}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    blocks = np.random.default_rng(0).integers(0, 64, (4, SEQ)).astype(np.int32)
    monkeypatch.setattr(runner, "FrozenSubstrate", lambda _name: substrate)
    monkeypatch.setattr(runner, "load_token_blocks", lambda *a, **k: blocks)
    return Path(cfg)


def test_audit_only_prints_json_and_does_not_train(
    runner, setup, tmp_path, capsys
) -> None:
    code = runner.main(
        ["--config", str(setup), "--audit-only", "--output-dir", str(tmp_path / "runs")]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and any(c["name"] == "a1_parity" for c in data["checks"])
    assert not (tmp_path / "runs").exists()  # no run directory, no training


def test_audit_before_training_writes_audit_json(runner, setup, tmp_path) -> None:
    code = runner.main(
        [
            "--config",
            str(setup),
            "--audit",
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 0
    run = tmp_path / "runs" / "r"
    assert json.loads((run / "audit.json").read_text())["ok"] is True
    assert (run / "results.json").exists()  # training ran after the audit


def test_a_failed_audit_stops_before_training(
    runner, setup, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runner, "a2_audit", lambda *a, **k: [Check("boom", False, 0.0, "bad")]
    )
    code = runner.main(
        [
            "--config",
            str(setup),
            "--audit",
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 1
    assert not (tmp_path / "runs" / "r").exists()
    assert runner.main(["--config", str(setup), "--audit-only"]) == 1


def test_no_audit_flags_means_no_audit(runner, setup, tmp_path, monkeypatch) -> None:
    def must_not_run(*a, **k):
        raise AssertionError("audit ran without a flag")

    monkeypatch.setattr(runner, "a2_audit", must_not_run)
    code = runner.main(
        [
            "--config",
            str(setup),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 0
    assert not (tmp_path / "runs" / "r" / "audit.json").exists()
