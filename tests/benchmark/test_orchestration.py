from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.provenance import (
    effective_condition,
    finalize_attempt,
    next_attempt_name,
    sha256_file,
    validate_complete_attempt,
)
from training.eval import evaluate_all_stages


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


def _manifest(tmp_path: Path, *, complete: bool = True) -> Path:
    runs = []
    for method in ("a1", "a2", "a3"):
        result = tmp_path / method / "results.json"
        if complete:
            result.parent.mkdir()
            result.write_text("{}\n", encoding="utf-8")
        runs.append(
            {
                "run_id": method,
                "method": method,
                "status": "complete" if complete else "pending",
                "results": str(result),
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"runs": runs}), encoding="utf-8"
    )
    return tmp_path


def test_global_eval_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        evaluate_all_stages(tmp_path)


def test_global_eval_rejects_incomplete_runs(tmp_path: Path) -> None:
    root = _manifest(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluate_all_stages(root)


def test_global_eval_delegates_to_common_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _manifest(tmp_path)
    evaluation = {"completed_runs": 3, "planned_runs": 3, "records": []}
    report = root / "REPORT.md"

    monkeypatch.setattr("benchmark.core.evaluate_manifest", lambda path: evaluation)

    def write_report(path, *, report_rank):
        assert path == root
        assert report_rank == 16
        report.write_text("# Report\n", encoding="utf-8")
        return report

    monkeypatch.setattr("benchmark.core.report_manifest", write_report)
    result = evaluate_all_stages(root, report_rank=16)
    assert result["global_evaluation"]["methods"] == ["a1", "a2", "a3"]
    assert result["global_evaluation"]["report_path"] == str(report)


def test_global_eval_can_report_partial_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _manifest(tmp_path, complete=False)
    evaluation = {"completed_runs": 1, "planned_runs": 3, "records": []}
    monkeypatch.setattr("benchmark.core.evaluate_manifest", lambda path: evaluation)
    result = evaluate_all_stages(
        root,
        generate_report=False,
        require_complete=False,
    )
    assert len(result["global_evaluation"]["incomplete_runs"]) == 3
    assert result["global_evaluation"]["report_path"] is None
