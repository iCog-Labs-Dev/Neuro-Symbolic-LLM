"""Benchmark CLI, measurement, provenance and visualization regression tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from benchmark.benchmark import (
    DEFAULT_CONFIGS,
    DEFAULT_RUNTIME_CONFIG,
    _filter_adapter_params,
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
    next_attempt_name,
    select_a3_k,
    sha256_file,
    validate_complete_attempt,
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


def test_real_default_configs_materialize(tmp_path: Path) -> None:
    specs = expand_specs(
        tmp_path,
        seeds=[0],
        methods=["a1", "a2", "a3"],
        k_values=[4],
        rank_values=[4],
    )

    assert {Path(spec.config) for spec in specs} == set(DEFAULT_CONFIGS.values())
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
    )
    assert sequential.training_tasks is sequential.tasks
    assert sequential.task_order == ("first", "second")


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
