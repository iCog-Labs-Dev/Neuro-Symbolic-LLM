"""Consolidated test suite for the Stage A A1/A2/A3 benchmark harness and metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

from benchmark.core import (
    BENCHMARK_RUNTIME_SOURCE,
    DEFAULT_CONFIGS,
    DEFAULT_RUNTIME_CONFIG,
    _command,
    _filter_adapter_params,
    _materialize,
    _spearman_corr,
    _write_json,
    compute_domain_overlap,
    compute_semantic_overlap,
    evaluate_manifest,
    expand_specs,
    make_manifest,
    report_manifest,
)
from benchmark.provenance import validate_complete_attempt
from metrics.performance import (
    calibration_stats,
    ece_from_stats,
    expected_calibration_error,
)
from metrics.representation import (
    pairwise_ablation_interaction,
    synergy,
    synergy_proxy,
)
from runtime import data_loader
from stages.stage_base import summarize


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


def test_report_requires_a_matched_three_stage_cohort(tmp_path: Path) -> None:
    evaluation = {
        "completed_runs": 2,
        "planned_runs": 2,
        "domains": ["d1"],
        "records": [
            {
                "method": "a1",
                "run_id": "a1-0",
                "seed": 0,
                "rank": 16,
                "protocol": "sequential",
                "task_order": ["eurlex", "ledgar", "scotus"],
                "final_mean_loss": 2.0,
                "residual_compressibility": {"8": 0.8, "16": 1.0},
            },
            {
                "method": "a1",
                "run_id": "a1-1",
                "seed": 1,
                "rank": 16,
                "protocol": "sequential",
                "task_order": ["eurlex", "ledgar", "scotus"],
                "final_mean_loss": 2.0,
                "residual_compressibility": {"8": 0.8, "16": 1.0},
            },
        ],
    }
    _write_json(tmp_path / "evaluation.json", evaluation)
    report_path = report_manifest(tmp_path, report_rank=4)
    report = report_path.read_text(encoding="utf-8")
    assert (
        "| Metric | A1 | A2 | A3 | A2-A1 paired delta | A3-A1 paired delta |" in report
    )
    assert "No complete matched A1/A2/A3 cohort is available" in report
    for metric in (
        "Residual compressibility",
        "Semantic-factor fracture",
        "Semantic-factor overlap",
        "Semantic-factor gradient conflict",
        "Mixed curvature coupling proxy",
        "Off-support leakage",
        "Update commutator",
        "Causal fingerprint separability",
        "Lowercase category naturality proxy",
        "Quadratic ablation fidelity Spearman",
        "Validation loss",
        "Perplexity",
        "Task accuracy",
        "Task primary score",
        "Task macro-F1",
        "Token calibration ECE",
        "Validation-loss forgetting",
        "Held-out transfer",
        "Sequential versus joint gap",
        "Anchor KL drift",
        "Symbolic retrieval F1@4",
        "Long-range coherence accuracy",
    ):
        assert f"| {metric} |" in report


def test_report_groups_by_configuration_and_shows_na_for_unimplemented(
    tmp_path: Path,
) -> None:
    evaluation = {
        "completed_runs": 4,
        "planned_runs": 4,
        "domains": ["d1"],
        "records": [
            {
                "run_id": "a1",
                "method": "a1",
                "seed": 0,
                "protocol": "sequential",
                "task_order": ["eurlex", "ledgar", "scotus"],
                "rank": 16,
                "final_mean_loss": 2.0,
            },
            {
                "run_id": "a2",
                "method": "a2",
                "seed": 0,
                "protocol": "sequential",
                "task_order": ["eurlex", "ledgar", "scotus"],
                "rank": 16,
                "final_mean_loss": 1.9,
            },
            {
                "run_id": "a3-k1",
                "method": "a3",
                "seed": 0,
                "protocol": "sequential",
                "task_order": ["eurlex", "ledgar", "scotus"],
                "rank": 16,
                "k": 1,
                "final_mean_loss": 1.8,
            },
            {
                "run_id": "a3-k4",
                "method": "a3",
                "seed": 0,
                "protocol": "sequential",
                "task_order": ["eurlex", "ledgar", "scotus"],
                "rank": 16,
                "k": 4,
                "final_mean_loss": 1.7,
            },
        ],
    }
    _write_json(tmp_path / "evaluation.json", evaluation)
    report_path = report_manifest(tmp_path, report_rank=16)
    report = report_path.read_text(encoding="utf-8")
    assert "A3 K=1" in report
    assert "A3 K=4" in report
    assert (
        "| Metric | A1 | A2 | A3 | A2-A1 paired delta | A3-A1 paired delta |" in report
    )
    assert (
        "| Mixed curvature coupling proxy | N/A | N/A | N/A | N/A | N/A | lower | 0 |"
        in report
    )
    assert "| Validation loss | 2 (n=1) | 1.9 (n=1) | 1.8 (n=1)" in report


def test_summarize_mixed_task_availability_nan_aware() -> None:
    names = ["eurlex", "ledgar", "scotus"]
    init = {
        "eurlex": {"base_loss": 3.0},
        "ledgar": {"base_loss": 3.0},
        "scotus": {"base_loss": 3.0},
    }
    rows = [
        {
            "eurlex": {
                "loss": 2.5,
                "ppl": 1.0,
                "kl": 0.0,
                "task_skip_reason": "label field 'label' not in dataset columns ['text', 'labels']",
            },
            "ledgar": {
                "loss": 2.4,
                "ppl": 1.0,
                "kl": 0.0,
                "task_accuracy": 0.20,
                "task_f1": 0.15,
            },
            "scotus": {
                "loss": 2.6,
                "ppl": 1.0,
                "kl": 0.0,
                "task_accuracy": 0.10,
                "task_f1": 0.08,
            },
        },
        {
            "eurlex": {"loss": 2.6, "ppl": 1.0, "kl": 0.0},
            "ledgar": {
                "loss": 1.8,
                "ppl": 1.0,
                "kl": 0.0,
                "task_accuracy": 0.80,
                "task_f1": 0.75,
            },
            "scotus": {
                "loss": 2.5,
                "ppl": 1.0,
                "kl": 0.0,
                "task_accuracy": 0.15,
                "task_f1": 0.12,
            },
        },
        {
            "eurlex": {"loss": 2.7, "ppl": 1.0, "kl": 0.0},
            "ledgar": {
                "loss": 1.9,
                "ppl": 1.0,
                "kl": 0.0,
                "task_accuracy": 0.75,
                "task_f1": 0.70,
            },
            "scotus": {
                "loss": 1.7,
                "ppl": 1.0,
                "kl": 0.0,
                "task_accuracy": 0.90,
                "task_f1": 0.85,
            },
        },
    ]
    summary = summarize(names, init, rows)
    assert "eurlex" in summary["skipped"]
    assert "label" in summary["skipped"]["eurlex"]
    assert "ledgar" not in summary["skipped"]
    assert "scotus" not in summary["skipped"]
    assert summary["accuracy_forgetting"][0] is None
    assert summary["accuracy_forgetting"][1] is None
    assert summary["accuracy_forgetting"][2] == pytest.approx(0.05)


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


def test_domain_overlap_within_layer() -> None:
    domain_effects = {
        "domain_A": [1.0, 0.0, 0.0, 0.0],
        "domain_B": [0.5, 0.5, 0.0, 0.0],
        "domain_C": [0.0, 0.0, 1.0, 1.0],
    }
    matrix = compute_domain_overlap(domain_effects)

    assert matrix["domain_A:domain_B"] == pytest.approx(0.5)
    assert matrix["domain_A:domain_C"] == pytest.approx(0.0)
    assert matrix["domain_B:domain_C"] == pytest.approx(0.0)


def test_domain_overlap_identical_domains_give_one() -> None:
    domain_effects = {
        "d1": [0.2, 0.4, 0.6, 0.8],
        "d2": [0.2, 0.4, 0.6, 0.8],
    }
    matrix = compute_domain_overlap(domain_effects)
    assert matrix["d1:d2"] == pytest.approx(1.0)


def test_domain_overlap_ignores_negative_contributions() -> None:
    domain_effects = {
        "d1": [1.0, -2.0, 0.0],
        "d2": [1.0, -1.0, 0.0],
    }
    matrix = compute_domain_overlap(domain_effects)
    assert matrix["d1:d2"] == pytest.approx(1.0)


def test_semantic_category_overlap() -> None:
    cat_effects = {
        "concept_syntax": [0.8, 0.2, 0.0, 0.0],
        "concept_semantics": [0.0, 0.0, 0.5, 0.5],
        "concept_discourse": [0.4, 0.1, 0.25, 0.25],
    }
    cat_matrix = compute_semantic_overlap(cat_effects)
    assert "concept_discourse:concept_semantics" in cat_matrix
    assert "concept_semantics:concept_syntax" in cat_matrix
    assert cat_matrix["concept_semantics:concept_syntax"] == pytest.approx(0.0)
    assert cat_matrix["concept_discourse:concept_semantics"] == pytest.approx(0.5)


def test_synergy_proxy_diagonal_metric() -> None:
    delta_i = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    delta_j = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    g_diag = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)

    val = float(synergy_proxy(delta_i, delta_j, g_diag))
    assert val == pytest.approx(48.0)


def test_synergy_proxy_matrix_metric() -> None:
    delta_i = jnp.array([1.0, 2.0], dtype=jnp.float32)
    delta_j = jnp.array([3.0, 4.0], dtype=jnp.float32)
    g_matrix = jnp.array([[2.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)
    val = float(synergy_proxy(delta_i, delta_j, g_matrix))
    assert val == pytest.approx(19.0)


def test_synergy_backward_compat() -> None:
    d1 = jnp.array([1.0, 2.0])
    d2 = jnp.array([3.0, 4.0])
    fisher = jnp.array([0.5, 1.5])
    assert float(synergy(d1, d2, fisher)) == float(synergy_proxy(d1, d2, fisher))


def test_measured_pairwise_ablation_interaction_additive() -> None:
    s_ij = float(
        pairwise_ablation_interaction(
            loss_full=10.0,
            loss_ablate_i=12.0,
            loss_ablate_j=13.0,
            loss_ablate_ij=15.0,
        )
    )
    assert s_ij == pytest.approx(0.0)


def test_measured_pairwise_ablation_interaction_synergistic() -> None:
    s_ij = float(
        pairwise_ablation_interaction(
            loss_full=10.0,
            loss_ablate_i=12.0,
            loss_ablate_j=13.0,
            loss_ablate_ij=17.0,
        )
    )
    assert s_ij == pytest.approx(2.0)


def test_spearman_correlation_validation() -> None:
    proxy_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    measured_vals = [2.0, 4.0, 6.0, 8.0, 10.0]
    corr = _spearman_corr(proxy_vals, measured_vals)
    assert corr == pytest.approx(1.0)

    rev_vals = [5.0, 4.0, 3.0, 2.0, 1.0]
    corr_rev = _spearman_corr(proxy_vals, rev_vals)
    assert corr_rev == pytest.approx(-1.0)


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


def test_benchmark_commands_select_legal_runtime_datasets(tmp_path: Path) -> None:
    specs = expand_specs(
        tmp_path,
        seeds=[0],
        methods=["a1", "a2", "a3"],
        k_values=[4],
    )

    for spec in specs:
        command = _command(spec, tmp_path / "config.yaml")
        source_index = command.index("--runtime-source") + 1
        assert command[source_index] == BENCHMARK_RUNTIME_SOURCE == "separate"


def test_runtime_source_override_ignores_yaml_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[str] = []

    def fake_separate(datasets, tokenizer, **kwargs):
        selected.extend(
            item["name"] for item in datasets if item.get("type") == "separate"
        )
        return (), {}, {}

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
