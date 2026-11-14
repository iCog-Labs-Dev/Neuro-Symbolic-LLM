"""Reference examples for the research formulas and report pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.core import evaluate_manifest, select_a3_k
from benchmark.provenance import sha256_file
from benchmark.visualization import (
    _metric_samples,
    _select_headline_cohort,
    generate_reports_and_visualizations,
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


def test_audit_inputs_are_checkpoint_bound(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("benchmark.core._load_deployed_metrics", lambda *args: {})
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
    audit = evaluate_manifest(tmp_path)["records"][0]["research_audits"]
    assert audit["metrics"]["factor_organization"]["mean_fracture"] == 0.5
    evidence["checkpoint_sha256"] = "wrong"
    (audit_dir / "a1-0.json").write_text(json.dumps(evidence), encoding="utf-8")
    assert (
        evaluate_manifest(tmp_path)["records"][0]["research_audits"]["status"]
        == "invalid"
    )


def test_cross_method_audits_require_same_protocol():
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


def test_report_writes_png_and_two_tables_in_four_folders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    benchmark = tmp_path / "runs"
    benchmark.mkdir()
    records = [
        {
            "method": method,
            "run_id": method,
            "seed": 0,
            "rank": 4,
            "k": 1 if method == "a3" else None,
            "protocol": "sequential",
            "task_order": ["eurlex", "ledgar", "scotus"],
            "heldout_tasks": [],
            "runtime_data_sha256": "same",
            "final_mean_loss": 2.0 + i / 10,
        }
        for i, method in enumerate(("a1", "a2", "a3"))
    ]
    (benchmark / "manifest.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    (benchmark / "evaluation.json").write_text(
        json.dumps({"records": records, "completed_runs": 3, "planned_runs": 3}),
        encoding="utf-8",
    )
    result = generate_reports_and_visualizations(
        benchmark, output_dir=tmp_path / "reports", repository_root=tmp_path / "repo"
    )
    for name in ("A1", "A2", "A3", "comparison"):
        folder = tmp_path / "reports" / "stage_A" / name
        report = next(folder.glob("*.md")).read_text(encoding="utf-8")
        assert "Internal Metrics" in report and "External Metrics" in report
        images = list((folder / "figures").glob("*.png"))
        assert images and all(
            path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in images
        )
    assert result["figures_enabled"]
