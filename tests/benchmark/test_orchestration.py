"""Dataset-independent orchestration and realized identity regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmark import benchmark as runner


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
    """Stub subprocess output to test the contract, not model training quality."""
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
    monkeypatch.setattr(runner, "train_manifest", lambda *a, **k: calls.append("train"))
    monkeypatch.setattr(
        runner,
        "evaluate_manifest",
        lambda *a: calls.append("eval") or {"completed_runs": 0, "planned_runs": 0},
    )
    assert runner.main(["--output-dir", str(tmp_path)]) == 0
    assert calls == ["plan", "train", "eval"]


def test_generated_reports_excluded_from_tracked_fingerprint(tmp_path):
    import subprocess

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
