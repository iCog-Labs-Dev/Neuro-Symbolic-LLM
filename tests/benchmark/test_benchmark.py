"""Benchmark CLI, measurement, provenance and visualization regression tests."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from benchmark import benchmark as runner
from benchmark.benchmark import (
    DEFAULT_CONFIGS,
    DEFAULT_RUNTIME_CONFIG,
    EXTERNAL_METRIC_SPECS,
    INTERNAL_METRIC_SPECS,
    MEASURED_METRICS,
    REPORTED_METRICS,
    _a4_symbolic_metrics,
    _filter_adapter_params,
    _load_deployed_metrics,
    _materialize,
    _metric_samples,
    _select_headline_cohort,
    _write_json,
    aggregate_lm_rows,
    effective_condition,
    evaluate_manifest,
    expand_specs,
    finalize_attempt,
    make_manifest,
    measure_ablation_prediction,
    measurement_reason,
    metric_values_for_record,
    next_attempt_name,
    resolve_report_dir,
    select_a3_k,
    sha256_file,
    validate_complete_attempt,
)
from benchmark.summary import report_manifest
from benchmark.visualization import generate_visualizations
from metrics.adapter_audits import (
    measure_adapter_ablations,
    measure_cross_adapter_curvature,
)
from metrics.performance import (
    calibration_stats,
    ece_from_stats,
    expected_calibration_error,
)
from metrics.research_audits import (
    ablation_fidelity,
    accuracy_forgetting,
    calibration_error,
    classification_metrics,
    curvature_coupling,
    evaluate_audits,
    factor_organization,
    fingerprint_separability,
    fingerprint_stability,
    functional_commutator,
    off_support_leakage,
    squared_error,
)


@pytest.fixture
def data_loader():
    """Keep report/formula tests runnable without the model training stack."""
    for package in ("datasets", "torch", "optax"):
        pytest.importorskip(package)
    from runtime import data_loader

    return data_loader


def test_ece_finite_when_bins_are_empty() -> None:
    logits = jnp.zeros((2, 5, 10), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(50.0)
    labels = jnp.zeros((2, 5), dtype=jnp.int32)

    ece = float(expected_calibration_error(logits, labels, bins=15))
    assert math.isfinite(ece)
    assert not math.isnan(ece)
    assert ece == pytest.approx(0.0, abs=1e-3)


def test_ece_confident_wrong_predictions_near_one() -> None:
    logits = jnp.zeros((2, 5, 10), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(50.0)
    labels = jnp.ones((2, 5), dtype=jnp.int32)

    ece = float(expected_calibration_error(logits, labels, bins=15))
    assert math.isfinite(ece)
    assert ece == pytest.approx(1.0, abs=1e-3)


def test_ece_jits_cleanly() -> None:
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 16))
    labels = jnp.ones((2, 8), dtype=jnp.int32)

    jitted_ece = jax.jit(expected_calibration_error, static_argnames="bins")
    val1 = float(expected_calibration_error(logits, labels, bins=10))
    val2 = float(jitted_ece(logits, labels, bins=10))
    assert math.isfinite(val2)
    assert val1 == pytest.approx(val2, rel=1e-5)


def test_ece_ignores_masked_labels() -> None:
    logits = jnp.zeros((1, 4, 5), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(20.0)
    labels = jnp.array([[-100, -100, -100, -100]], dtype=jnp.int32)

    conf_sum, acc_sum, counts = calibration_stats(logits, labels, bins=5)
    assert float(counts.sum()) == 0.0
    ece = float(ece_from_stats(conf_sum, acc_sum, counts))
    assert ece == pytest.approx(0.0)


def test_pooled_ece_across_batches_equals_concatenated() -> None:
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    logits1 = jax.random.normal(k1, (2, 6, 8))
    labels1 = jax.random.randint(k2, (2, 6), 0, 8)
    logits2 = jax.random.normal(k3, (3, 6, 8))
    labels2 = jax.random.randint(k4, (3, 6), 0, 8)

    c1, a1, cnt1 = calibration_stats(logits1, labels1, bins=10)
    c2, a2, cnt2 = calibration_stats(logits2, labels2, bins=10)
    pooled_ece = float(ece_from_stats(c1 + c2, a1 + a2, cnt1 + cnt2))

    logits_cat = jnp.concatenate([logits1, logits2], axis=0)
    labels_cat = jnp.concatenate([labels1, labels2], axis=0)
    cat_ece = float(expected_calibration_error(logits_cat, labels_cat, bins=10))

    assert math.isfinite(pooled_ece)
    assert pooled_ece == pytest.approx(cat_ece, rel=1e-5)


def test_a3_k_sweep_curves_do_not_overwrite(tmp_path: Path) -> None:
    manifest = {
        "schema": 1,
        "runs": [
            {
                "run_id": "a3-seed0-r4-k1-sequential",
                "method": "a3",
                "seed": 0,
                "k": 1,
                "rank": 4,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r1.json"),
            },
            {
                "run_id": "a3-seed0-r16-k1-sequential",
                "method": "a3",
                "seed": 0,
                "k": 1,
                "rank": 16,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r2.json"),
            },
            {
                "run_id": "a3-seed0-r4-k4-sequential",
                "method": "a3",
                "seed": 0,
                "k": 4,
                "rank": 4,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r3.json"),
            },
            {
                "run_id": "a3-seed0-r16-k4-sequential",
                "method": "a3",
                "seed": 0,
                "k": 4,
                "rank": 16,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r4.json"),
            },
        ],
    }
    _write_json(tmp_path / "meta.json", {"runtime_data": {"task_order": ["d1", "d2"]}})
    _write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "cfg.yaml").write_text("adapter:\n  rank: 16\n", encoding="utf-8")

    _write_json(tmp_path / "r1.json", {"base_loss": [3.0], "loss": [[2.8]]})
    _write_json(tmp_path / "r2.json", {"base_loss": [3.0], "loss": [[2.6]]})
    _write_json(tmp_path / "r3.json", {"base_loss": [3.0], "loss": [[2.7]]})
    _write_json(tmp_path / "r4.json", {"base_loss": [3.0], "loss": [[2.4]]})

    evaluation = evaluate_manifest(tmp_path)
    records_by_id = {r["run_id"]: r for r in evaluation["records"]}

    rec_k1 = records_by_id["a3-seed0-r4-k1-sequential"]
    assert rec_k1["residual_compressibility"] is not None
    assert float(rec_k1["residual_compressibility"]["4"]) == pytest.approx(
        0.5, rel=1e-3
    )

    rec_k4 = records_by_id["a3-seed0-r4-k4-sequential"]
    assert rec_k4["residual_compressibility"] is not None
    assert float(rec_k4["residual_compressibility"]["4"]) == pytest.approx(
        0.5, rel=1e-3
    )


def test_nan_and_inf_sanitized_to_valid_json(tmp_path: Path) -> None:
    bad_data = {
        "metric_nan": float("nan"),
        "metric_inf": float("inf"),
        "metric_neg_inf": float("-inf"),
        "nested": {"val": float("nan")},
        "list": [1.0, float("nan"), 3.0],
    }
    out_file = tmp_path / "sanitized.json"
    _write_json(out_file, bad_data)

    raw_text = out_file.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    loaded = json.loads(raw_text)
    assert loaded["metric_nan"] is None
    assert loaded["metric_inf"] is None
    assert loaded["metric_neg_inf"] is None
    assert loaded["nested"]["val"] is None
    assert loaded["list"] == [1.0, None, 3.0]


def test_resume_rejects_unverified_result_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "results.json", {"test": {}})

    valid, reason = validate_complete_attempt(
        run_dir,
        condition_sha256="missing",
        config_sha256="missing",
    )
    assert not valid
    assert "missing artifacts" in str(reason)


def test_expand_specs_and_make_manifest(tmp_path: Path) -> None:
    manifest = make_manifest(
        tmp_path,
        seeds=[0, 1],
        methods=["a1", "a3"],
        k_values=[1, 4],
        rank_values=[16],
    )
    assert len(manifest["runs"]) == 6
    assert (tmp_path / "manifest.json").is_file()


def test_a4_command_uses_only_stage_a4_flags(tmp_path: Path) -> None:
    inputs = {}
    for name in (
        "a4_train_sources",
        "a4_validation_sources",
        "a4_alignment_labels",
        "a4_patterns",
        "a4_embeddings",
        "a4_test_sources",
        "a4_native_library",
    ):
        path = tmp_path / name
        path.write_text("input", encoding="utf-8")
        inputs[name] = str(path)
    (tmp_path / "a4_native_library.json").write_text("{}", encoding="utf-8")
    spec = expand_specs(
        tmp_path,
        seeds=[0],
        methods=["a4"],
        k_values=[1],
        a4_inputs=inputs,
    )[0]
    spec = replace(spec, a4_warm_start=str(tmp_path / "a3.npz"))
    command = runner._command(spec, Path(spec.config))
    assert command[1:4] == [str(runner.RUNNERS["a4"]), "--data-mode", "aligned"]
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--rank") + 1] == "16"
    assert "--runtime-config" not in command
    assert "--training-mode" not in command
    assert "--runtime-source" not in command
    assert "--task-order" not in command
    assert "--heldout-task" not in command
    assert "--arm" not in command


def test_a4_arms_plan_each_seed_and_rank_with_distinct_conditions(tmp_path):
    inputs = {"a4_retrieval_backend": "tcp", "a4_arms": list(runner.A4_ARM_CONFIGS)}
    for field in (*runner.A4_BANK_FIELDS, *runner.A4_ALIGNED_FIELDS):
        path = tmp_path / field
        path.write_text(field)
        inputs[field] = str(path)
    manifest = make_manifest(
        tmp_path,
        seeds=[3, 8],
        methods=["a3", "a4"],
        k_values=[1],
        rank_values=[4, 16],
        epochs=2,
        a4_inputs=inputs,
    )
    rows = [row for row in manifest["runs"] if row["method"] == "a4"]
    assert len(rows) == 16
    assert len({row["run_id"] for row in rows}) == 16
    assert len({row["condition_sha256"] for row in rows}) == 16
    for row in rows:
        assert row["a4_arm"] in row["run_id"]
        assert row["a4_data_mode"] == "aligned"
        assert row["a4_epochs"] == 2
        config = yaml.safe_load(Path(row["materialized_config"]).read_text())
        assert Path(config["residual_config"]).is_file()
        assert (
            tuple(config["objective"].values())
            == {
                "ce_only": (0, 0),
                "key_only": (1, 0),
                "value_only": (0, 1),
                "full": (1, 1),
            }[row["a4_arm"]]
        )


def test_runtime_a4_command_and_result(tmp_path):
    bank = {}
    for field in runner.A4_BANK_FIELDS:
        path = tmp_path / field
        path.touch()
        bank[field] = str(path)
    (spec,) = expand_specs(
        tmp_path,
        seeds=[8],
        methods=["a4"],
        k_values=[1],
        epochs=2,
        a4_inputs={**bank, "a4_data_mode": "runtime", "a4_retrieval_backend": "tcp"},
    )
    warm_start = tmp_path / "a3.npz"
    runtime = tmp_path / "materialized-runtime.yaml"
    command = runner._command(
        replace(spec, a4_warm_start=str(warm_start)), Path(spec.config), runtime
    )
    assert command[2:6] == ["--data-mode", "runtime", "--seed", "8"]
    assert command[command.index("--runtime-config") + 1] == str(runtime)
    assert command[command.index("--epochs") + 1] == "2"
    assert "--alignment-labels" not in command and "--train-sources" not in command

    validation = {"shared_validation": {"loss": 2, "ppl": 7, "tokens": 100}}
    record = runner._result_record(
        {
            "mode": "runtime_sequential",
            "seed": 8,
            "validation_scope": "shared",
            "task_order": ["stories_1"],
            "after_task": [validation],
            "test": {"stories": {"loss": 3, "ppl": 20}},
        },
        {"run_id": spec.run_id, "method": "a4", "seed": 8, "a4_arm": spec.a4_arm},
    )
    assert record["protocol"] == "sequential"
    assert record["task_order"] == ["stories_1"]
    assert record["final_mean_loss"] == 2
    assert record["final_mean_ppl"] == pytest.approx(math.exp(2))
    assert record["final_test_mean_loss"] == 3
    assert record["final_test_mean_ppl"] == pytest.approx(math.exp(3))
    assert record["mean_forgetting"] is None


@pytest.mark.parametrize("mode", ["aligned", "runtime"])
def test_a4_follows_fresh_a3_runs_and_resumes_per_seed(tmp_path, monkeypatch, mode):
    from experiments import run_stage_a3
    from experiments.run_stage_a4 import _apply_training_overrides, parse_args
    from runtime.data_loader import RuntimeData, runtime_pairing_fingerprint
    from stages.stage_A.a4.config import load_config

    monkeypatch.setattr(
        runner, "source_fingerprint", lambda _: {"source_sha256": "test"}
    )
    inputs = {"a4_retrieval_backend": "tcp", "a4_data_mode": mode}
    fields = runner.A4_BANK_FIELDS + (
        runner.A4_ALIGNED_FIELDS if mode == "aligned" else ()
    )
    for field in fields:
        path = tmp_path / field
        path.write_text(field)
        inputs[field] = str(path)
    output = tmp_path / "benchmark"
    make_manifest(
        output,
        seeds=[7, 19],
        methods=["a3", "a4"],
        k_values=[1],
        rank_values=[4],
        epochs=2,
        a4_inputs=inputs,
    )
    calls = []
    a3_checkpoints = {}

    order = ["first", "second"]
    frozen = {
        "revision": "immutable-commit",
        "architecture": {"hidden_size": 8, "num_layers": 1},
        "tokenizer_sha256": "a" * 64,
    }

    def load_runtime(**kwargs):
        blocks = np.arange(1024, dtype=np.int32).reshape(8, 128)
        tasks = tuple(
            _Task(name, blocks + index, (blocks + index + 2,), 2)
            for index, name in enumerate(order)
        )
        return RuntimeData(
            tasks,
            {"stories": blocks + 4},
            tuple(order),
            {
                "mode": kwargs["source"],
                "training_mode": kwargs["training_mode"],
                "sequence_length": kwargs["sequence_length"],
                "batch_size": kwargs["batch_size"],
                "task_order": order,
            },
        )

    def make_model(*args, rank, **kwargs):
        return SimpleNamespace(
            init_params=lambda key: {
                "layer_0": {
                    "A": jax.random.uniform(key, (8, rank)),
                    "B": np.zeros((rank, 8), dtype=np.float32),
                },
            },
            adapter_params=lambda params: params,
        )

    substrate = SimpleNamespace(
        architecture=SimpleNamespace(hidden_size=8, num_layers=1),
        params={},
        tokenizer=object(),
        pairing_identity=lambda: frozen,
        verify_frozen=lambda: {"params_unchanged": True},
    )
    monkeypatch.setattr(run_stage_a3, "FrozenSubstrate", lambda _: substrate)
    monkeypatch.setattr(
        run_stage_a3,
        "ResidualAdapter",
        lambda *_: SimpleNamespace(layers=(0,), scale=1),
    )
    monkeypatch.setattr(run_stage_a3, "A3Model", make_model)
    monkeypatch.setattr(run_stage_a3, "load_runtime_data", load_runtime)
    monkeypatch.setattr(
        run_stage_a3, "train", lambda **kwargs: (kwargs["params"], {"steps": 0})
    )
    monkeypatch.setattr(
        run_stage_a3, "evaluate_deployment", lambda *args, **kwargs: {"loss": 1.0}
    )
    monkeypatch.setattr(
        run_stage_a3, "log_experiment_header", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(run_stage_a3, "log_final_summary", lambda *args, **kwargs: None)

    real_run = subprocess.run

    def launch(command, **kwargs):
        if command[0] != sys.executable:
            return real_run(command, **kwargs)
        flags = command[2:]

        def option(name):
            return flags[flags.index(name) + 1]

        folder = Path(option("--output-dir")) / option("--run-name")
        config = Path(option("--config"))
        stage = Path(command[1]).stem
        calls.append(stage)
        if stage == "run_stage_a3":
            assert "--record-a4-pairing" in flags
            options = run_stage_a3.parse_args(flags)
            assert options.record_a4_pairing
            seed = yaml.safe_load(config.read_text())["training"]["seed"]
            assert run_stage_a3.main(flags) == 0
            a3_checkpoints[seed] = folder / "checkpoints/a3hybrid_final.npz"
            metadata = json.loads((folder / "meta.json").read_text())
            runtime_sha256 = sha256_file(options.runtime_config)
            data = load_runtime(
                source=options.runtime_source,
                training_mode=options.training_mode,
                sequence_length=128,
                batch_size=8,
            )
            assert metadata["a4_pairing"] == {
                "frozen": frozen,
                "runtime_sha256": runtime_pairing_fingerprint(
                    data,
                    runtime_config_sha256=runtime_sha256,
                ),
                "evaluation_split": "test",
            }
            return SimpleNamespace(returncode=0)
        else:
            checkpoints = folder / "checkpoints"
            checkpoints.mkdir(parents=True)
            meta = {"runtime_data": {"task_order": order}}
            args = parse_args(flags)
            effective = _apply_training_overrides(load_config(config), args)
            checkpoint = a3_checkpoints[effective.training.seed]
            assert args.warm_start == checkpoint
            assert effective.residual_config == checkpoint.parents[1] / "config.yaml"
            assert args.rank == 4
            assert effective.training.epochs == 2
            (folder / "config.yaml").write_text(yaml.safe_dump(effective.to_dict()))
            (checkpoints / "a4-final.npz").touch()
            pair = json.loads(args.benchmark_pair.read_text())
            assert pair["task_order"] == order
            assert args.runtime_config.is_file()
            meta.update(
                a3_pairing=pair,
                seed=effective.training.seed,
                rank=args.rank,
                warm_start_sha256=sha256_file(checkpoint),
                residual_config_sha256=sha256_file(effective.residual_config),
            )
            result = {
                "mode": "standalone_aligned"
                if mode == "aligned"
                else "runtime_sequential",
                "seed": effective.training.seed,
                "a3_pairing": pair,
                "task_order": order if mode == "runtime" else [],
                "checkpoints": ["checkpoints/a4-final.npz"],
            }
        _write_json(folder / "meta.json", meta)
        _write_json(folder / "results.json", result)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", launch)
    runner.train_manifest(output)
    manifest = runner._read_json(output / "manifest.json")
    a4_rows = [row for row in manifest["runs"] if row["method"] == "a4"]
    assert calls == ["run_stage_a3", "run_stage_a3", "run_stage_a4", "run_stage_a4"]
    assert all(row["status"] == "complete" for row in a4_rows)
    assert all(
        row["task_order"] == (["first", "second"] if mode == "runtime" else [])
        for row in a4_rows
    )
    assert a4_rows[0]["a4_warm_start_sha256"] != a4_rows[1]["a4_warm_start_sha256"]
    runner.train_manifest(output)
    assert len(calls) == 4

    a4 = a4_rows[0]
    a3 = manifest["runs"][0]
    pair = a4["a4_pairing"]
    assert pair["run_id"] == a3["run_id"]
    assert pair["checkpoint"] == str(a3_checkpoints[7])
    assert pair["checkpoint_sha256"] == sha256_file(a3_checkpoints[7])
    assert pair["common_sha256"] == runner.fingerprint(a4["a3_a4_common"])
    assert pair["frozen"] == frozen
    result = json.loads(Path(a4["results"]).read_text())
    assert result["a3_pairing"] == pair
    assert runner._result_record(result, a4)["a3_pairing"] == pair

    for key, incompatible in (
        ("runtime_config_sha256", "different-split"),
        ("task_order", ["second", "first"]),
        ("training_mode", "joint"),
        ("model", "different-model"),
        ("seq_len", 256),
        ("layer_selection", {"layers": [2]}),
    ):
        wrong = {**a3, "a3_a4_common": {**a3["a3_a4_common"], key: incompatible}}
        assert runner._resolve_a4_warm_start({"runs": [wrong, a3]}, a4) == pair
        with pytest.raises(RuntimeError, match=key):
            runner._resolve_a4_warm_start({"runs": [wrong]}, a4)

    for wrong in ({**a3, "rank": 16}, {**a3, "seed": 19}):
        with pytest.raises(RuntimeError, match="requested seed, rank"):
            runner._resolve_a4_warm_start({"runs": [wrong]}, a4)
    with pytest.raises(RuntimeError, match="incomplete run"):
        runner._resolve_a4_warm_start({"runs": [{**a3, "status": "running"}]}, a4)
    alternate = {**a3, "run_id": "another-a3", "planned_run_id": "another-plan", "k": 2}
    with pytest.raises(RuntimeError, match="ambiguous compatible A3 runs"):
        runner._resolve_a4_warm_start({"runs": [a3, alternate]}, a4)
    assert (
        runner._resolve_a4_warm_start(
            {"runs": [alternate, a3]}, {**a4, "a4_a3_run_id": a3["run_id"]}
        )
        == pair
    )
    assert (
        runner._resolve_a4_warm_start({"runs": [alternate, a3]}, {**a4, "a4_a3_k": 1})
        == pair
    )
    a3_dir = Path(a3["run_dir"])
    meta_path = a3_dir / "meta.json"
    meta = runner._read_json(meta_path)
    pairing_metadata = meta.pop("a4_pairing")
    completion_spec = runner._read_json(a3_dir / "spec.json")
    _write_json(meta_path, meta)
    runner.finalize_attempt(a3_dir, completion_spec)
    with pytest.raises(RuntimeError, match="missing A4 pairing metadata"):
        runner._resolve_a4_warm_start({"runs": [a3]}, a4)
    for key, invalid in (
        ("evaluation_split", "train"),
        ("runtime_sha256", "invalid"),
        ("frozen", None),
    ):
        _write_json(
            meta_path, {**meta, "a4_pairing": {**pairing_metadata, key: invalid}}
        )
        runner.finalize_attempt(a3_dir, completion_spec)
        with pytest.raises(
            RuntimeError, match="incompatible or incomplete A4 pairing metadata"
        ):
            runner._resolve_a4_warm_start({"runs": [a3]}, a4)
    meta["a4_pairing"] = pairing_metadata
    for changes in (
        {"num_layers": 2},
        {"hidden_size": 16},
        {"adapted_layers": [1]},
        {"runtime_data": {**meta["runtime_data"], "training_mode": "joint"}},
        {
            "runtime_data": {**meta["runtime_data"], "task_order": list(reversed(order))}
        },
    ):
        _write_json(meta_path, {**meta, **changes})
        runner.finalize_attempt(a3_dir, completion_spec)
        with pytest.raises(
            RuntimeError, match="incompatible or missing realized configuration"
        ):
            runner._resolve_a4_warm_start({"runs": [a3]}, a4)
    _write_json(meta_path, meta)
    config_path = a3_dir / "config.yaml"
    config_text = config_path.read_text()
    for section, key, invalid in (("training", "seed", 19), ("adapter", "rank", 16)):
        config = yaml.safe_load(config_text)
        config[section][key] = invalid
        config_path.write_text(yaml.safe_dump(config))
        runner.finalize_attempt(a3_dir, completion_spec)
        with pytest.raises(
            RuntimeError, match="incompatible or missing realized configuration"
        ):
            runner._resolve_a4_warm_start({"runs": [a3]}, a4)
    config_path.write_text(config_text)
    runner.finalize_attempt(a3_dir, completion_spec)
    checkpoint = a3_checkpoints[7]
    with checkpoint.open("ab") as stream:
        stream.write(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        runner._resolve_a4_warm_start({"runs": [a3]}, a4)
    runner.finalize_attempt(a3_dir, completion_spec)
    with pytest.raises(RuntimeError, match="checkpoint SHA-256 sidecar mismatch"):
        runner._resolve_a4_warm_start({"runs": [a3]}, a4)
    a3_checkpoints[7].unlink()
    with pytest.raises(RuntimeError, match="missing final checkpoint"):
        runner._resolve_a4_warm_start({"runs": [a3]}, a4)


def test_real_default_configs_materialize(tmp_path: Path) -> None:
    specs = expand_specs(
        tmp_path,
        seeds=[0],
        methods=["a1", "a2", "a3"],
        k_values=[4],
        rank_values=[4],
    )

    assert {Path(spec.config) for spec in specs} == {DEFAULT_CONFIGS[name] for name in ("a1", "a2", "a3")}
    for spec in specs:
        materialized = _materialize(spec, tmp_path, epochs=1)
        config = yaml.safe_load(materialized.read_text(encoding="utf-8"))
        assert "domains" not in config
        assert config["adapter"]["rank"] == 4
        assert config["training"]["epochs"] == 1


def test_runtime_source_override_ignores_yaml_default(
    monkeypatch: pytest.MonkeyPatch,
    data_loader,
) -> None:
    selected: list[str] = []

    def fake_separate(datasets, tokenizer, **kwargs):
        selected.extend(
            item["name"] for item in datasets if item.get("type") == "separate"
        )
        from types import SimpleNamespace

        return (SimpleNamespace(name="eurlex"),), {}, {}

    def fail_stories(*args, **kwargs):
        pytest.fail("stories loader should not be selected")

    monkeypatch.setattr(data_loader, "build_separate_tasks", fake_separate)
    monkeypatch.setattr(data_loader, "build_story_tasks", fail_stories)

    runtime_data = data_loader.load_runtime_data(
        DEFAULT_RUNTIME_CONFIG,
        tokenizer=object(),
        source="separate",
        sequence_length=2,
        batch_size=1,
        eval_batches=1,
    )

    assert runtime_data.metadata["mode"] == "separate"
    assert runtime_data.metadata["validation_scope"] == "task_specific"
    assert selected == ["eurlex", "ledgar", "scotus"]


def test_deployed_evaluation_filters_a2_auxiliary_params() -> None:
    layer_params = {"A": object(), "B": object()}
    full_a2_params = {
        "layer_5": layer_params,
        "cpc_q_5": {"weight": object()},
        "pred_layer_5": {"weight": object()},
    }

    assert _filter_adapter_params(full_a2_params) == {"layer_5": layer_params}


def test_compressibility_unreliable_when_reference_not_better_than_base(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema": 1,
        "runs": [
            {
                "run_id": f"a3-r{r}",
                "method": "a3",
                "seed": 0,
                "k": 1,
                "rank": r,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / f"r{r}.json"),
            }
            for r in (4, 16)
        ],
    }
    _write_json(tmp_path / "meta.json", {"runtime_data": {"task_order": ["d1", "d2"]}})
    _write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "cfg.yaml").write_text("adapter:\n  rank: 16\n", encoding="utf-8")
    _write_json(
        tmp_path / "r4.json",
        {"domains": ["d1", "d2"], "base_loss": [3.0, 3.0], "loss": [[2.5, 2.5]]},
    )
    _write_json(
        tmp_path / "r16.json",
        {"domains": ["d1", "d2"], "base_loss": [3.0, 3.0], "loss": [[2.5, 3.5]]},
    )
    ev = evaluate_manifest(tmp_path)
    rec = next(r for r in ev["records"] if r["rank"] == 4)
    assert rec["compressibility_status"] == "unreliable_reference"
    assert rec["residual_compressibility"] is None
    by_domain = rec["residual_compressibility_by_domain"]
    assert by_domain["d1"]["status"] == "ok"
    assert by_domain["d1"]["curve"]["4"] == pytest.approx(1.0)
    assert by_domain["d2"]["status"] == "unreliable_reference"


ROOT = Path(__file__).resolve().parents[2]


def test_lm_aggregation_is_token_weighted() -> None:
    result = aggregate_lm_rows(
        [
            {
                "tokens": 1,
                "loss_sum": 1.0,
                "base_loss_sum": 2.0,
                "kl_sum": 0.1,
                "correct": 1,
            },
            {
                "tokens": 3,
                "loss_sum": 9.0,
                "base_loss_sum": 12.0,
                "kl_sum": 0.9,
                "correct": 0,
            },
        ]
    )

    expected = {
        "tokens": 4,
        "loss": 2.5,
        "base_loss": 3.5,
        "perplexity": math.exp(2.5),
        "base_perplexity": math.exp(3.5),
        "kl": 0.25,
        "token_accuracy": 0.25,
    }
    assert result["status"] == "measured"
    assert {key: result[key] for key in expected} == pytest.approx(expected)


def test_lm_aggregation_reports_no_evaluated_tokens() -> None:
    assert aggregate_lm_rows([]) == {"status": "insufficient_data", "tokens": 0}
    assert aggregate_lm_rows(
        [
            {
                "tokens": 0,
                "loss_sum": 0.0,
                "base_loss_sum": 0.0,
                "kl_sum": 0.0,
                "correct": 0,
            }
        ]
    ) == {"status": "insufficient_data", "tokens": 0}


@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_lm_aggregation_rejects_invalid_token_counts(tokens: Any) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        aggregate_lm_rows(
            [
                {
                    "tokens": tokens,
                    "loss_sum": 0.0,
                    "base_loss_sum": 0.0,
                    "kl_sum": 0.0,
                    "correct": 0,
                }
            ]
        )


def test_shipped_runtime_config_keeps_external_lm_protocols(data_loader) -> None:
    config = data_loader.load_runtime_config(ROOT / "configs/runtime_datasets.yaml")

    assert config["evaluation"]["anchor"]["name"] == "general_language_anchor"
    assert len(config["evaluation"]["anchor"]["texts"]) == 5
    assert all(
        isinstance(dataset.get("revision"), str) and len(dataset["revision"]) == 40
        for dataset in config["datasets"]
    )


class _Dataset:
    column_names = ["body"]

    def __getitem__(self, field: str) -> list[str]:
        assert field == "body"
        return [" first ", "", "second"]


def test_lm_text_loading_forwards_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
    data_loader,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def load_dataset(path: str, **kwargs: Any) -> _Dataset:
        calls.append((path, kwargs))
        return _Dataset()

    monkeypatch.setattr(data_loader, "load_dataset", load_dataset)
    texts = data_loader.load_split_texts(
        {
            "dataset": "owner/corpus",
            "config": "subset",
            "revision": "a" * 40,
            "text_field": "body",
        },
        "train[:10]",
    )

    assert texts == ["first", "second"]
    assert calls == [
        (
            "owner/corpus",
            {
                "name": "subset",
                "split": "train[:10]",
                "revision": "a" * 40,
                "streaming": False,
            },
        )
    ]


def test_array_fingerprint_captures_layout_independent_values_shape_and_dtype(
    data_loader,
) -> None:
    blocks = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    fingerprint = data_loader.array_fingerprint(blocks)

    assert data_loader.array_fingerprint(np.asfortranarray(blocks)) == fingerprint
    assert data_loader.array_fingerprint(blocks.reshape(1, 4)) != fingerprint
    assert data_loader.array_fingerprint(blocks.astype(np.int64)) != fingerprint


@dataclass(frozen=True)
class _Task:
    name: str
    train_blocks: np.ndarray
    eval_set: tuple[Any, ...]
    epochs: int


def test_runtime_data_applies_task_order_heldout_and_joint_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_loader,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "runtime:\n"
        "  source: separate\n"
        "  task_order: [second, first]\n"
        "  training_mode: joint\n"
        "  heldout_tasks: [second]\n"
        "datasets:\n"
        "  - {name: first, type: separate}\n"
        "  - {name: second, type: separate}\n",
        encoding="utf-8",
    )
    first_blocks = np.arange(6, dtype=np.int32).reshape(3, 2)
    second_blocks = np.arange(8, dtype=np.int32).reshape(4, 2)

    def build_tasks(*_args: Any, **_kwargs: Any):
        tasks = (
            _Task("first", first_blocks, (), 2),
            _Task("second", second_blocks, (), 3),
        )
        return tasks, {"first": first_blocks, "second": second_blocks}, {}

    monkeypatch.setattr(data_loader, "build_separate_tasks", build_tasks)
    monkeypatch.setattr(data_loader, "Domain", _Task)

    runtime = data_loader.load_runtime_data(
        config_path,
        tokenizer=object(),
        sequence_length=2,
        batch_size=2,
        eval_batches=1,
    )

    assert [task.name for task in runtime.tasks] == ["first"]
    assert runtime.task_order == ("first",)
    assert runtime.heldout_tasks == ("second",)
    assert runtime.metadata["training_mode"] == "joint"
    assert runtime.training_tasks is not None
    assert [task.name for task in runtime.training_tasks] == ["joint"]
    np.testing.assert_array_equal(
        runtime.training_tasks[0].train_blocks,
        np.tile(first_blocks[:2], (2, 1)),
    )
    assert set(runtime.test_blocks) == {"first", "second"}

    sequential = data_loader.load_runtime_data(
        config_path,
        tokenizer=object(),
        task_order=("first", "second"),
        training_mode="sequential",
        heldout_tasks=(),
        sequence_length=2,
        batch_size=2,
        eval_batches=1,
        epochs_override=1,
    )
    assert sequential.training_tasks is sequential.tasks
    assert sequential.task_order == ("first", "second")
    assert [task.epochs for task in sequential.tasks] == [1, 1]


def test_runtime_pairing_checks_split_membership_and_realized_blocks(
    tmp_path, monkeypatch, data_loader
):
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "runtime: {source: separate}\n"
        "datasets:\n"
        "  - {name: example, type: separate, dataset: example, train_split: train, "
        "validation_split: validation, test_split: test, separator: ' '}\n"
    )
    texts = {"train": ["11 12"], "validation": ["23 24"], "test": ["35 36"]}
    monkeypatch.setattr(data_loader, "load_split_texts", lambda _, split: texts[split])

    def tokenize(text, **_):
        return {"input_ids": [[int(token) for token in text.split()]]}

    def load(sources=None):
        return data_loader.load_runtime_data(
            config,
            tokenize,
            sequence_length=2,
            batch_size=1,
            eval_batches=1,
            source_texts=sources,
        )

    def fingerprint(data):
        return data_loader.runtime_pairing_fingerprint(
            data, runtime_config_sha256=sha256_file(config)
        )

    source_files = {split: set(records) for split, records in texts.items()}
    data = load(source_files)
    original = fingerprint(data)
    assert original == fingerprint(load())
    with pytest.raises(ValueError, match="outside the runtime split"):
        load({**source_files, "validation": {"11 12"}})
    for split in texts:
        saved = texts[split]
        texts[split] = ["24 25"]
        assert original != fingerprint(load()), split
        texts[split] = saved
    assert original != fingerprint(replace(data, task_order=("changed",)))
    assert original != fingerprint(
        replace(data, metadata={**data.metadata, "seed": 123})
    )
    config.write_text(
        config.read_text().replace("dataset: example", "dataset: different")
    )
    assert original != fingerprint(load())


def _condition(**overrides):
    values = {
        "method": "a3",
        "seed": 0,
        "rank": 16,
        "k": 4,
        "protocol": "sequential",
        "task_order": ("eurlex", "ledgar", "scotus"),
        "heldout_tasks": (),
        "a2_arm": None,
        "stage_config_sha256": "a" * 64,
        "runtime_config_sha256": "b" * 64,
        "runtime_source": "separate",
        "epochs": 3,
        "source_sha256": "c" * 64,
    }
    values.update(overrides)
    return effective_condition(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"method": "a2", "a2_arm": "full"},
        {"seed": 1},
        {"rank": 8},
        {"k": 8},
        {"protocol": "joint"},
        {"task_order": ("scotus", "ledgar", "eurlex")},
        {"heldout_tasks": ("scotus",)},
        {"stage_config_sha256": "d" * 64},
        {"runtime_config_sha256": "e" * 64},
        {"runtime_source": "stories"},
        {"epochs": 4},
        {"source_sha256": "f" * 64},
    ],
)
def test_effective_condition_hashes_every_scientific_axis(override) -> None:
    base = _condition()
    assert base["condition_sha256"] != _condition(**override)["condition_sha256"]


def test_attempts_are_append_only(tmp_path: Path) -> None:
    assert next_attempt_name(tmp_path) == "attempt-0001"
    (tmp_path / "attempt-0001").mkdir()
    assert next_attempt_name(tmp_path) == "attempt-0002"


def test_completion_validation_rejects_mutation(tmp_path: Path) -> None:
    run = tmp_path / "attempt-0001"
    run.mkdir()
    (run / "results.json").write_text("{}\n", encoding="utf-8")
    (run / "config.yaml").write_text("adapter: {}\n", encoding="utf-8")
    provenance = run / "provenance"
    provenance.mkdir()
    (provenance / "materialized_stage.yaml").write_text(
        "adapter: {}\n", encoding="utf-8"
    )
    (provenance / "materialized_runtime.yaml").write_text(
        "runtime: {}\n", encoding="utf-8"
    )
    condition = _condition()
    spec = {
        **condition,
        "config_sha256": sha256_file(provenance / "materialized_stage.yaml"),
        "runtime_config_sha256": sha256_file(provenance / "materialized_runtime.yaml"),
    }
    finalize_attempt(run, spec)
    valid, reason = validate_complete_attempt(
        run,
        condition_sha256=condition["condition_sha256"],
        config_sha256=spec["config_sha256"],
    )
    assert valid and reason is None
    (run / "results.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    valid, reason = validate_complete_attempt(
        run,
        condition_sha256=condition["condition_sha256"],
        config_sha256=spec["config_sha256"],
    )
    assert not valid
    assert "hash mismatch" in str(reason)


def test_factor_fracture_and_overlap_match_paper_examples():
    pytest.importorskip("metrics.a4.representation")
    result = factor_organization([[1, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]])
    assert result["fracture"] == pytest.approx([0.25, 1.0, 0.25])
    assert result["overlap"][0][2] == 0
    partial = factor_organization([[0, 0], [1, 0]])
    assert partial["fracture"] == [None, 0.5]
    assert partial["zero_effect_factor_indices"] == [0]
    assert partial["mean_overlap"] is None
    with pytest.raises(ValueError, match="nonnegative"):
        factor_organization([[-1, 2]])


def test_factor_organization_reuses_shared_metrics(monkeypatch):
    contributions = [[1, 0], [0, 0]]
    shared = {
        "fracture": [0.5, None],
        "valid_factor_indices": [0],
        "zero_effect_factor_indices": [1],
        "overlap_matrix": [[1.0]],
        "mean_off_diagonal_overlap": None,
    }
    calls = []

    def shared_factor_organization(values):
        calls.append(values)
        return shared

    def load_module(name):
        assert name == "metrics.a4.representation"
        return SimpleNamespace(factor_organization=shared_factor_organization)

    monkeypatch.setattr("metrics.research_audits.import_module", load_module)
    result = factor_organization(contributions)
    assert calls == [contributions]
    assert result["mean_fracture"] == 0.5
    assert result["overlap"] == [[1.0]]
    assert result["zero_effect_factor_indices"] == [1]


def test_factor_organization_unavailable_does_not_block_other_audits(monkeypatch):
    def missing_module(name):
        raise ModuleNotFoundError("missing shared metrics", name="metrics.a4")

    monkeypatch.setattr("metrics.research_audits.import_module", missing_module)
    result = evaluate_audits(
        {
            "schema": "a1-a3-research-audit/v1",
            "factor_importance": [[1, 0]],
            "mixed_hessian": [[2]],
        }
    )
    assert result["factor_organization"]["status"] == "unavailable"
    assert result["factor_organization"]["mean_fracture"] is None
    assert result["mixed_curvature"] == 2


def test_audit_formulas_and_missing_information():
    assert curvature_coupling([[2]]) == 2
    assert (
        evaluate_audits(
            {
                "schema": "a1-a3-research-audit/v1",
                "cross_support_hessians": [[[1]], [[2]]],
            }
        )["cross_support_coupling"]
        == 2
    )
    assert off_support_leakage(
        [0.4, 0.5, 0.1, 0], [True, True, False, False]
    ) == pytest.approx(0.1)
    assert off_support_leakage([0, 0], [True, False]) is None
    assert fingerprint_separability(
        [[0], [2], [8], [10]], ["a", "a", "b", "b"]
    ) == pytest.approx(16)
    assert squared_error([[1, 0]], [[0, 0]], "naturality") == 1
    assert functional_commutator([[3, 4]], [[0, 0]]) == 5
    assert fingerprint_stability([[1, 2], [3, 4]], [[2, 4], [6, 8]]) == pytest.approx(1)
    assert fingerprint_stability([[1, 1]], [[2, 2]]) is None
    assert ablation_fidelity([0.2, 0.05, 0.3], [0.18, 0.06, 0.09])[
        "mae"
    ] == pytest.approx(0.08)
    with pytest.raises(ValueError):
        off_support_leakage([1, 2], [True])


def test_task_scores_and_accuracy_forgetting_are_distinct_from_token_accuracy():
    assert accuracy_forgetting(
        [[0.90, None, None], [0.82, 0.88, None], [0.75, 0.80, 0.91]]
    )["final"] == pytest.approx(0.115)
    result = classification_metrics([0, 1, 1], [0, 1, 0])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert classification_metrics([[1, 0], [0, 1]], [[1, 1], [0, 1]], multilabel=True)[
        "micro_f1"
    ] == pytest.approx(0.8)
    assert calibration_error([0.9, 0.9], [1, 0]) == pytest.approx(0.4)


def test_k_selection_uses_shared_seed_and_fingerprint():
    rows = []
    for k, seed, data, loss in [
        (1, 0, "shared", 1.0),
        (1, 1, "only-1", 0.01),
        (4, 0, "shared", 0.9),
        (4, 2, "only-4", 10.0),
    ]:
        rows.append(
            {
                "method": "a3",
                "protocol": "sequential",
                "rank": 4,
                "task_order": ["x"],
                "heldout_tasks": [],
                "k": k,
                "seed": seed,
                "runtime_data_sha256": data,
                "final_mean_loss": loss,
            }
        )
    result = select_a3_k(rows, protocol="sequential", rank=4, order=("x",), heldout=())
    assert result["selected_k"] == 4
    assert result["matched_seeds"] == [0]


def test_symbolic_audit_inputs_are_ignored(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("benchmark.benchmark._load_deployed_metrics", lambda *args: {})
    monkeypatch.setattr(
        "metrics.research_audits.factor_organization",
        lambda values: {"mean_fracture": 0.5},
    )
    checkpoint = tmp_path / "outputs" / "run" / "params" / "task_0_x.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("adapter: {}", encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(
        '{"domains": ["x"], "base_loss": [2], "loss": [[1]]}', encoding="utf-8"
    )
    run = {
        "run_id": "a1-0",
        "method": "a1",
        "seed": 0,
        "config": str(config),
        "results": str(result),
        "output_dir": str(tmp_path / "outputs"),
        "run_name": "run",
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"runs": [run]}), encoding="utf-8"
    )
    audit_dir = tmp_path / "measurement_inputs"
    audit_dir.mkdir()
    evidence = {
        "schema": "a1-a3-research-audit/v1",
        "run_id": "a1-0",
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol_sha256": "a" * 64,
        "factor_importance": [[1, 0]],
    }
    (audit_dir / "a1-0.json").write_text(json.dumps(evidence), encoding="utf-8")
    record = evaluate_manifest(tmp_path)["records"][0]
    assert "research_audits" not in record


def test_legacy_probe_metrics_are_not_reported():
    rows = []
    for method in ("a1", "a2", "a3"):
        rows.append(
            {
                "method": method,
                "seed": 0,
                "k": 1 if method == "a3" else None,
                "rank": 4,
                "protocol": "sequential",
                "task_order": ["x"],
                "heldout_tasks": [],
                "runtime_data_sha256": "data",
                "final_mean_loss": 1,
                "research_audits": {
                    "protocol_sha256": "other" if method == "a2" else "same",
                    "metrics": {"naturality_error": 0.1},
                },
            }
        )
    cohort = _select_headline_cohort(rows, 4)
    assert cohort is not None
    assert all(
        not values for values in _metric_samples(cohort, "naturality_error").values()
    )


def test_mixed_curvature_measures_known_cross_hessian():
    import jax.numpy as jnp

    from benchmark.benchmark import measure_mixed_curvature

    params = {"layer_0": {"A": jnp.ones((2,)), "B": jnp.ones((2,))}}
    result = measure_mixed_curvature(
        lambda p: 3 * jnp.sum(p["layer_0"]["A"] * p["layer_0"]["B"]),
        params,
        samples=3,
    )
    assert result["status"] == "measured"
    assert result["value"] == pytest.approx(3 * np.sqrt(2))
    independent = measure_mixed_curvature(
        lambda p: jnp.sum(p["layer_0"]["A"] ** 2) + jnp.sum(p["layer_0"]["B"] ** 2),
        params,
    )
    assert independent["value"] == 0
    with pytest.raises(ValueError, match="positive integer"):
        measure_mixed_curvature(lambda p: 0, params, samples=0)


def test_combined_configuration_covers_required_run_axes(tmp_path):
    manifest = make_manifest(
        tmp_path,
        seeds=[0],
        methods=["a1", "a2", "a3"],
        k_values=[4],
        rank_values=[4, 16],
        protocols=["sequential", "joint", "commutator"],
        task_order=["task_a", "task_b"],
        epochs=1,
    )
    assert len(manifest["runs"]) == 24
    for method in ("a1", "a2", "a3"):
        for rank in (4, 16):
            runs = [
                run
                for run in manifest["runs"]
                if run["method"] == method and run["rank"] == rank
            ]
            assert {run["protocol"] for run in runs} == {
                "sequential",
                "joint",
                "commutator",
            }
            orders = [
                tuple(run["task_order"])
                for run in runs
                if run["protocol"] == "commutator"
            ]
            assert len(orders) == 2 and orders[0] == orders[1][::-1]


def test_report_exact_metrics_and_missing_evidence(tmp_path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    _write_json(runs / "manifest.json", {"runtime_source": "stories"})
    _write_json(runs / "evaluation.json", {"records": []})
    path = report_manifest(runs, report_dir=reports)
    content = path.read_text()
    assert len(MEASURED_METRICS) == len(INTERNAL_METRIC_SPECS) + len(EXTERNAL_METRIC_SPECS)
    assert len(INTERNAL_METRIC_SPECS) == 16
    assert len(EXTERNAL_METRIC_SPECS) == 24
    for label, _, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS:
        assert label in content
    assert "No evaluated runs" in content
    assert "Residual direction redundancy" in content
    assert "Invariant redundancy" not in content
    assert "![" not in content
    assert not (runs / "REPORT.md").exists()
    assert (reports / "index.html").is_file()
    assert not any(
        word in content
        for word in (
            "Effective rank",
            "Peak memory",
            "Trainable parameters",
            "Curvature",
        )
    )


def test_report_dir_resolves_from_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_report_dir() == ROOT / "docs/stage_A_results"
    assert resolve_report_dir("custom") == ROOT / "custom"


def test_general_report_uses_methods_as_rows_and_one_evidence_table(tmp_path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    order = ["rocstories_task_1", "rocstories_task_2", "rocstories_task_3"]
    records = []
    for index, method in enumerate(("a1", "a2", "a3")):
        records.append(
            {
                "run_id": method,
                "method": method,
                "seed": 0,
                "rank": 4,
                "k": 4 if method == "a3" else None,
                "protocol": "sequential",
                "runtime_source": "stories",
                "runtime_data_sha256": "shared",
                "task_order": order,
                "heldout_tasks": [],
                "final_mean_loss": 3.0 + index / 10,
                "deployed_evaluation": {
                    "domains": {
                        "rocstories": {
                            "status": "measured",
                            "token_accuracy": 0.3 + index / 100,
                        }
                    }
                },
            }
        )
    records.append(
        {
            "run_id": "a4",
            "method": "a4",
            "seed": 0,
            "rank": 4,
            "k": None,
            "protocol": "a4",
            "runtime_source": "stories",
            "runtime_data_sha256": None,
            "task_order": [],
            "heldout_tasks": [],
            "final_mean_loss": 2.5,
            "deployed_evaluation": {
                "a4": {
                    "symbolic": {
                        "a4_retrieval_macro_precision_at_8": 0.8,
                        "a4_retrieval_macro_recall_at_8": 0.7,
                        "residual_direction_redundancy": 0.12,
                    }
                }
            },
        }
    )
    _write_json(runs / "manifest.json", {"runtime_source": "stories"})
    _write_json(runs / "evaluation.json", {"records": records})
    content = report_manifest(runs, report_rank=4, report_dir=reports).read_text()
    assert "Realized task order: rocstories task 1, 2, 3" in content
    assert "| Method | Residual compressibility |" in content
    assert "| A1 |" in content and "| A2 |" in content and "| A3 |" in content
    lines = content.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| Method | Validation loss"))
    columns = [cell.strip() for cell in lines[header].strip("|").split("|")]
    values = [cell.strip() for cell in lines[header + 5].strip("|").split("|")]
    assert dict(zip(columns, values, strict=True))["Retrieval macro precision @8"] == "0.8 (n=1)"
    assert "inherited from frozen A3 residual" in content
    assert "paired delta" not in content.lower()
    assert "| Run | Metric | Value |" not in content
    assert "## Per-dataset token accuracy" not in content
    assert "## Run conditions" not in content
    assert (
        "| Run | Method | Seed | Rank | K | Dataset/test split | Token accuracy | N/A metrics and reasons |"
        in content
    )
    assert "| a1 | A1 | 0 | 4 | None | rocstories | 0.3 (n=1) |" in content


def test_a4_probe_metrics_and_frozen_a3_redundancy(tmp_path):
    run_dir = tmp_path / "a4"
    _write_json(
        run_dir / "evaluation" / "representation" / "result.json",
        {
            "contract_version": "a4-representation-v1",
            "reports": [
                {
                    "condition": "trained",
                    "next_token_accuracy": 0.62,
                    "kl_base_to_model": 0.14,
                    "layers": [],
                }
            ],
        },
    )
    (run_dir / "params").mkdir(parents=True)
    np.savez(
        run_dir / "params" / "a3-frozen.npz",
        **{
            "layer_0/A": np.eye(2, dtype=np.float32),
            "layer_0/B": np.eye(2, dtype=np.float32),
        },
    )

    values, reason, _ = _a4_symbolic_metrics(run_dir)

    assert values["token_accuracy"] == pytest.approx(0.62)
    assert values["domain_kl"] == pytest.approx(0.14)
    assert values["residual_direction_redundancy"] == pytest.approx(0.0)
    assert reason is None


def test_effect_magnitudes_are_not_prediction_fidelity():
    record = {
        "deployed_evaluation": {
            "representation": {"adapter_ablations": {"mean_loss_increase": 1.5}}
        }
    }
    for key in ("ablation_mae", "ablation_spearman"):
        assert metric_values_for_record(record, key, 16) == []
        assert "predictor" in measurement_reason(record, key)
    for key in ("adapter_ablation", "causal_fingerprint", "cross_adapter_curvature"):
        assert metric_values_for_record(record, key, 16) == []


def test_reevaluation_replaces_old_metric_catalog_without_editing_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    _write_json(path, {"runs": [], "measured_metrics": ["invariant_redundancy"]})
    original = path.read_bytes()
    evaluation = evaluate_manifest(tmp_path)
    assert evaluation["measured_metrics"] == list(REPORTED_METRICS)
    assert "residual_direction_redundancy" in evaluation["measured_metrics"]
    assert "invariant_redundancy" not in evaluation["measured_metrics"]
    assert path.read_bytes() == original


def test_summary_has_no_plotting_dependency():
    import benchmark.summary as summary

    tree = ast.parse(Path(summary.__file__).read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        "matplotlib" not in ast.unparse(node)
        and "visualization" not in ast.unparse(node)
        for node in imports
    )


def test_empty_artifacts_generate_explicit_na_figures(tmp_path):
    pytest.importorskip("matplotlib")
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    reports.mkdir()
    sentinel = reports / "keep.txt"
    sentinel.write_text("keep")
    _write_json(runs / "manifest.json", {})
    _write_json(runs / "evaluation.json", {"records": []})
    generate_visualizations(runs, reports)
    assert sentinel.read_text() == "keep"
    for section in ("A1", "A2", "A3", "comparison"):
        folder = reports / "stage_A" / section
        assert len(list((folder / "figures").glob("*.png"))) == (
            len(INTERNAL_METRIC_SPECS)
            + len(EXTERNAL_METRIC_SPECS)
        )
        report = next(folder.glob("*.md")).read_text()
        assert "Figures are stored separately" in report
        assert "![" not in report
    assert "<img" not in (reports / "index.html").read_text()


def _audit_parameters():
    return {
        "layer_1": {"A": jnp.ones(2), "B": jnp.array([1.0, 2.0])},
        "layer_2": {"A": jnp.ones(2), "B": jnp.array([3.0, 4.0])},
    }


def test_ablations_record_signed_effects_without_mutating_checkpoint():
    params = _audit_parameters()
    audit = measure_adapter_ablations(
        {
            "helpful": lambda p: 10 - p["layer_1"]["B"].sum(),
            "harmful": lambda p: p["layer_2"]["B"].sum(),
        },
        params,
    )
    assert audit["effects"] == {
        "layer_1": {"helpful": 3.0, "harmful": 0.0},
        "layer_2": {"helpful": 0.0, "harmful": -7.0},
    }
    assert audit["mean_loss_increase"] == -1
    assert audit["fingerprint_rms"] == pytest.approx(np.sqrt(58 / 4))
    np.testing.assert_array_equal(params["layer_1"]["B"], [1, 2])
    np.testing.assert_array_equal(params["layer_2"]["B"], [3, 4])


def test_cross_adapter_norm_covers_both_factors_and_excludes_diagonal_blocks():
    def loss(p):
        a, b = p["layer_1"], p["layer_2"]
        return (
            3 * jnp.vdot(a["A"], b["B"])
            + 2 * jnp.vdot(a["B"], b["A"])
            + 100 * jnp.vdot(a["A"], a["B"])
        )

    audit = measure_cross_adapter_curvature(loss, _audit_parameters(), samples=3)
    assert audit["value"] == pytest.approx(np.sqrt(26), rel=0.002)
    assert list(audit["pairs"]) == ["layer_1:layer_2"]
    independent = measure_cross_adapter_curvature(
        lambda p: sum(jnp.sum(block["B"] ** 2) for block in p.values()),
        _audit_parameters(),
    )
    assert independent["value"] == 0


def test_audits_reject_undefined_or_nonfinite_measurements():
    with pytest.raises(ValueError, match="at least two"):
        measure_cross_adapter_curvature(
            lambda p: 0, {"layer_1": _audit_parameters()["layer_1"]}
        )
    with pytest.raises(ValueError, match="positive integer"):
        measure_cross_adapter_curvature(lambda p: 0, _audit_parameters(), samples=0)
    with pytest.raises(ValueError, match="non-finite"):
        measure_adapter_ablations({"bad": lambda p: float("nan")}, _audit_parameters())


def test_quadratic_gate_predictor_and_true_ablations_are_independent():
    params = {"layer_0": {"A": jnp.eye(2), "B": jnp.array([[1.0], [2.0]])}}
    original = np.asarray(params["layer_0"]["B"]).copy()
    ids = jnp.array([[0, 1, 1], [1, 0, 0]])

    def loss(p, batch):
        return 10 + (1 + batch[0, 0]) * p["layer_0"]["B"].sum()

    audit = measure_ablation_prediction(loss, params, {"shared_test": (ids,)})
    np.testing.assert_allclose(
        [point["gradient"] for point in audit["pairs"]], [1.5, 3]
    )
    np.testing.assert_allclose(
        [point["fisher_diagonal"] for point in audit["pairs"]], [2.5, 10]
    )
    np.testing.assert_allclose(
        [point["predicted"] for point in audit["pairs"]], [-0.25, 2]
    )
    np.testing.assert_allclose(
        [point["actual"] for point in audit["pairs"]], [-1.5, -3]
    )
    assert audit["mae"] == pytest.approx(3.125)
    assert audit["spearman"] == pytest.approx(-1)
    split = measure_ablation_prediction(
        loss, params, {"shared_test": (ids[:1], ids[1:])}
    )
    assert split["pairs"] == audit["pairs"]
    np.testing.assert_array_equal(params["layer_0"]["B"], original)


def test_constant_ablation_effects_have_mae_but_no_spearman():
    params = {"layer_0": {"A": jnp.eye(2), "B": jnp.zeros((2, 2))}}
    audit = measure_ablation_prediction(
        lambda p, ids: jnp.sum(p["layer_0"]["B"]),
        params,
        {"test": (jnp.ones((1, 3), dtype=jnp.int32),)},
    )
    assert audit["mae"] == 0
    assert audit["spearman"] is None
    record = {"deployed_evaluation": {"ablation_fidelity": audit}}
    assert metric_values_for_record(record, "ablation_mae", 2) == [0]
    assert metric_values_for_record(record, "ablation_spearman", 2) == []
    assert "nonconstant" in measurement_reason(record, "ablation_spearman")


def test_ablation_sampling_and_missing_test_data():
    params = {"layer_0": {"A": jnp.ones((1, 4)), "B": jnp.arange(4.0)[:, None]}}

    def loss(p, ids):
        return jnp.sum(p["layer_0"]["B"])

    with pytest.raises(ValueError, match="held-out"):
        measure_ablation_prediction(loss, params, {})
    with pytest.raises(ValueError, match="finite"):
        measure_ablation_prediction(
            lambda p, ids: loss(p, ids) * jnp.nan,
            params,
            {"test": (jnp.ones((1, 3)),)},
        )
    audit = measure_ablation_prediction(
        loss, params, {"test": (jnp.ones((1, 3)),)}, max_components=2
    )
    assert [row["channel"] for row in audit["pairs"]] == [0, 3]


@pytest.mark.parametrize("method", ["a1", "a2", "a3"])
@pytest.mark.parametrize("source", ["stories", "separate"])
def test_checkpoint_evaluation_respects_loader_test_keys(
    monkeypatch, tmp_path, method, source
):
    def module(name, **members):
        value = ModuleType(name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, value)

    class Substrate:
        def __init__(self, model):
            self.tokenizer = object()
            self.architecture = object()

        def run_with_interception(self, ids, modify_fn, intercept_layers):
            hidden = jax.nn.one_hot(ids, 2)
            logits = modify_fn(hidden, 0)
            return SimpleNamespace(logits=logits, hidden_state=lambda layer: hidden)

        def compute_loss(self, logits, ids, losses):
            probabilities = jax.nn.log_softmax(logits[:, :-1], axis=-1)
            return -jnp.take_along_axis(probabilities, ids[:, 1:, None], axis=-1).mean()

    class Adapter:
        layers = (0,)
        scale = 1.0

        def __init__(self, config, architecture):
            pass

        def validate_params(self, params):
            assert params["layer_0"]["B"].shape == (2, 2)

        def encode(self, block, hidden):
            return hidden @ block["A"]

        def modify_fn(self, params):
            return (
                lambda hidden, layer: hidden
                + self.encode(params["layer_0"], hidden) @ params["layer_0"]["B"]
            )

    ids = jnp.array([[0, 1, 0], [1, 0, 1]])
    names = (
        ["rocstories_task_1", "rocstories_task_2"]
        if source == "stories"
        else ["left", "right"]
    )
    test_sets = (
        {"rocstories": (ids,)} if source == "stories" else dict.fromkeys(names, (ids,))
    )
    tasks = [
        SimpleNamespace(name=name, train_blocks=np.asarray([[index, index, 1 - index]]))
        for index, name in enumerate(names)
    ]
    runtime = SimpleNamespace(tasks=tasks, get_test_sets=lambda *args: test_sets)
    calls = []

    def load_runtime_data(**kwargs):
        calls.append(kwargs)
        return runtime

    module(
        "frozenllm.substrate",
        FrozenSubstrate=Substrate,
        identity_modify=lambda hidden, layer: hidden,
    )
    module(
        "residual",
        ResidualAdapter=Adapter,
        ResidualConfig=SimpleNamespace(from_dict=lambda config: config),
    )
    module(
        "runtime.data_loader",
        load_runtime_data=load_runtime_data,
        load_runtime_config=lambda path: {},
    )
    module(
        "stages.stage_base",
        TrainingConfig=SimpleNamespace(
            from_dict=lambda config: SimpleNamespace(
                model="tiny", seq_len=3, batch_size=2, eval_batches=1, epochs=1
            )
        ),
        make_eval_set=lambda blocks, *args: (jnp.asarray(blocks),),
    )
    module(
        "stages.results",
        load_params=lambda path: {
            "layer_0": {"A": jnp.eye(2), "B": jnp.array([[0.2, -0.1], [0.4, 0.3]])}
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text("adapter: {}\n")
    checkpoint = tmp_path / "attempt" / "params"
    checkpoint.mkdir(parents=True)
    (checkpoint / "task_1_finished.npz").touch()
    run = {
        "method": method,
        "config": str(config),
        "materialized_runtime_config": str(config),
        "runtime_source": source,
        "task_order": names,
        "output_dir": str(tmp_path),
        "run_name": "attempt",
    }
    measured = _load_deployed_metrics(run, {})
    assert measured["status"] == "measured"
    assert set(measured["domains"]) == set(test_sets)
    assert calls[0]["task_order"] == names
    conflict = measured["representation"]["task_gradient_conflict"]
    assert conflict["status"] == "measured"
    assert list(conflict["evaluation_batches"]) == names
    assert conflict["split"].startswith("train")
    redundancy = measured["representation"]["residual_direction_redundancy"]["layer_0"]
    assert redundancy["read"] == pytest.approx(0)
    assert redundancy["write"] == pytest.approx(1 / np.sqrt(5))
    assert measured["ablation_fidelity"]["status"] == "measured"
    assert measured["ablation_fidelity"]["pairs"]
    record = {"deployed_evaluation": measured}
    for key in (
        "token_accuracy",
        "calibration",
        "domain_kl",
        "residual_direction_redundancy",
        "task_gradient_conflict",
        "ablation_mae",
        "ablation_spearman",
    ):
        assert metric_values_for_record(record, key, 2), key
    assert metric_values_for_record(record, "invariant_redundancy", 2) == []


@pytest.mark.parametrize("source", ["stories", "separate"])
def test_runtime_source_and_optional_order(tmp_path, source):
    config = tmp_path / "runtime.yaml"
    config.write_text(yaml.safe_dump({"runtime": {"source": source}, "datasets": []}))
    manifest = runner.make_manifest(
        tmp_path / "runs",
        seeds=[0],
        methods=["a1"],
        k_values=[1],
        runtime_config=config,
    )
    row = manifest["runs"][0]
    assert row["runtime_source"] == source
    assert row["task_order"] == []
    assert source in row["run_id"]
    spec = runner.expand_specs(
        tmp_path, seeds=[0], methods=["a1"], k_values=[1], runtime_config=config
    )[0]
    command = runner._command(spec, Path(row["materialized_config"]))
    assert "--task-order" not in command
    assert "--heldout-task" not in command
    overridden = runner.expand_specs(
        tmp_path,
        seeds=[0],
        methods=["a1"],
        k_values=[1],
        runtime_config=config,
        runtime_source="custom",
        task_order=["arbitrary"],
    )[0]
    assert overridden.runtime_source == "custom"
    assert runner._command(overridden, Path(overridden.config))[-2:] == [
        "--task-order",
        "arbitrary",
    ]


@pytest.mark.parametrize(
    "protocol,flag", [("heldout", "--heldout-task"), ("commutator", "--task-order")]
)
def test_named_protocols_fail_before_materialization(tmp_path, protocol, flag):
    with pytest.raises(ValueError, match=flag):
        runner.make_manifest(
            tmp_path, seeds=[0], methods=["a1"], k_values=[1], protocols=[protocol]
        )
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("source", ["stories", "separate"])
def test_realized_identity_and_resume(tmp_path, monkeypatch, source):
    monkeypatch.setattr(
        runner, "source_fingerprint", lambda root: {"source_sha256": "source"}
    )
    output = tmp_path / "runs"
    runner.make_manifest(
        output, seeds=[0], methods=["a1"], k_values=[1], runtime_source=source
    )
    calls = []

    def subprocess_run(command, **kwargs):
        calls.append(command)
        folder = (
            Path(command[command.index("--output-dir") + 1])
            / command[command.index("--run-name") + 1]
        )
        folder.mkdir(parents=True)
        runtime = command[command.index("--runtime-config") + 1]
        config = command[command.index("--config") + 1]
        (folder / "config.yaml").write_bytes(Path(config).read_bytes())
        runner._write_json(
            folder / "meta.json",
            {"runtime_data": {"task_order": ["generated_task"], "source": source}},
        )
        runner._write_json(
            folder / "results.json",
            {"domains": ["generated_task"], "loss": [[2.0]], "base_loss": [3.0]},
        )
        assert yaml.safe_load(Path(runtime).read_text())["runtime"]["source"] == source
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", subprocess_run)
    runner.train_manifest(output)
    row = runner._read_json(output / "manifest.json")["runs"][0]
    assert row["task_order"] == ["generated_task"]
    assert row["run_id"] != row["planned_run_id"]
    spec = runner._read_json(Path(row["run_dir"]) / "spec.json")
    assert spec["condition_sha256"] == row["condition_sha256"]
    assert spec["run_id"] == row["run_id"]
    runner.train_manifest(output)
    assert len(calls) == 1

    def deployed(run, result):
        assert run["task_order"] == ["generated_task"]
        return {"status": "unavailable", "reason": "test stub"}

    monkeypatch.setattr(runner, "_load_deployed_metrics", deployed)
    result = runner.evaluate_manifest(output)
    assert result["records"][0]["task_order"] == ["generated_task"]
    runner.train_manifest(output, epochs=2)
    assert len(calls) == 2
    updated = runner._read_json(output / "manifest.json")["runs"][0]
    assert updated["run_name"] == "attempt-0002"


def test_default_cli_runs_plan_train_eval(tmp_path, monkeypatch):
    calls = []

    def plan(*args, **kwargs):
        calls.append("plan")
        return {"runs": []}

    monkeypatch.setattr(runner, "make_manifest", plan)
    monkeypatch.setattr(
        runner, "train_manifest", lambda *args, **kwargs: calls.append("train")
    )
    monkeypatch.setattr(
        runner,
        "evaluate_manifest",
        lambda *args: calls.append("eval") or {"completed_runs": 0, "planned_runs": 0},
    )
    assert runner.main(["--output-dir", str(tmp_path)]) == 0
    assert calls == ["plan", "train", "eval"]


def test_generated_reports_excluded_from_tracked_fingerprint(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    reports = tmp_path / "docs/stage_A_results"
    reports.mkdir(parents=True)
    file = reports / "report.md"
    file.write_text("first")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    before = runner.source_fingerprint(tmp_path)
    file.write_text("updated")
    assert runner.source_fingerprint(tmp_path) == before
    (reports / "extra.json").write_text(json.dumps({}))
    assert runner.source_fingerprint(tmp_path) == before
