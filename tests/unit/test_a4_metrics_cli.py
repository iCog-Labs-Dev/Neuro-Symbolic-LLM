"""A4-only recorded-metric CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from metrics.a4.cli import (
    inspect_latency,
    inspect_run,
    latest_completed_run,
    list_runs,
    main,
    score_probes,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_incomplete_run_is_not_presented_as_completed(tmp_path):
    write_json(tmp_path / "status.json", {"stage": "A4", "status": "running"})
    result = inspect_run(tmp_path)
    assert result["status"]["status"] == "running"
    assert not result["results_available"]
    assert result["results"] == {}


def test_show_reports_recorded_losses_without_creating_artifacts(tmp_path, capsys):
    write_json(tmp_path / "status.json", {"stage": "A4", "status": "completed"})
    write_json(tmp_path / "results.json", {"stage": "A4", "validation": {"loss": 3}})
    records = [
        {"event": "eval", "split": "validation", "epoch": 0, "loss": 2, "ppl": 7.389},
        {"event": "train", "step": 5, "task": 1, "key": 2, "value": 3},
        {"event": "eval", "split": "validation", "epoch": 1, "loss": 3, "ppl": 20.086},
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main(["show", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "CE=2.000000" in printed and "CE=3.000000" in printed
    assert "Last update 5" in printed
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_other_stages_are_rejected(tmp_path):
    write_json(tmp_path / "status.json", {"stage": "A3", "status": "completed"})
    with pytest.raises(ValueError, match="Stage A4"):
        inspect_run(tmp_path)


@pytest.mark.parametrize("kind", ["semantic_firing", "exact_neighbor"])
def test_probe_origin_distinguishes_semantics_from_index_agreement(tmp_path, kind):
    path = write_json(
        tmp_path / "probes.json",
        {
            "label_kind": kind,
            "split": "validation",
            "probes": [{"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]}],
        },
    )
    result = score_probes(path)
    assert result["measurement"] == (
        "semantic_retrieval" if kind == "semantic_firing" else "ann_agreement"
    )
    assert result["metrics"]["macro_precision"] == 0.5
    assert result["metrics"]["macro_recall"] == 1
    assert "not new model inference" in result["scope"]


def test_probe_origin_is_required(tmp_path):
    path = write_json(tmp_path / "probes.json", {"probes": []})
    with pytest.raises(ValueError, match="label_kind"):
        score_probes(path)


@pytest.fixture
def timing_file(tmp_path):
    rows = [
        {
            "mode": "deployment",
            "sample": 0,
            "batch": 0,
            "step_seconds": 0.01,
            "full_round_trip_seconds": 0.004,
            "cpu_index_seconds": 0.003,
        }
    ]
    path = write_json(
        tmp_path / "result.json",
        {
            "status": "completed",
            "checkpoint_sha256": "checkpoint",
            "samples": rows,
            "reports": [{"communication_mean_below_five_percent": True}],
        },
    )
    write_json(
        tmp_path / "cuda-transfers-deployment.json",
        {
            "samples": [
                {
                    "sample": 0,
                    "batch": 0,
                    "unprofiled_step_seconds": 0.01,
                    "profiled_model_seconds": 0.02,
                    "d2h_gpu_seconds": 0.00005,
                    "h2d_gpu_seconds": 0.00035,
                }
            ]
        },
    )
    return path


def test_latency_recomputes_raw_samples_not_saved_pass_flags(timing_file):
    report = inspect_latency(timing_file)["reports"][0]
    assert not report["communication_mean_below_five_percent"]
    assert report["communication_ratio_of_means_percent"] == pytest.approx(10)
    assert report["cuda_transfers"]["transfer_only_mean_below_five_percent"]


def test_transfer_samples_must_match_primary_denominators(timing_file):
    path = timing_file.parent / "cuda-transfers-deployment.json"
    payload = json.loads(path.read_text())
    payload["samples"][0]["unprofiled_step_seconds"] = 0.1
    write_json(path, payload)
    with pytest.raises(ValueError, match="denominator"):
        inspect_latency(timing_file)


def test_missing_raw_transfers_are_not_fabricated(timing_file):
    (timing_file.parent / "cuda-transfers-deployment.json").unlink()
    assert "cuda_transfers" not in inspect_latency(timing_file)["reports"][0]


def test_json_output(tmp_path, capsys):
    write_json(tmp_path / "status.json", {"stage": "A4", "status": "running"})
    assert main(["show", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "A4"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_nonfinite_artifacts_are_rejected(tmp_path, value):
    (tmp_path / "status.json").write_text(
        '{"stage":"A4","status":"running","invalid":' + value + "}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        inspect_run(tmp_path)


def test_malformed_input_exits_with_cli_error(tmp_path, capsys):
    write_json(tmp_path / "status.json", {"stage": "A4"})
    with pytest.raises(SystemExit) as error:
        main(["show", str(tmp_path)])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["", "measure-latency", "measure-retrieval"])
def test_module_cli_needs_no_numeric_runtime(command):
    code = (
        "import runpy,sys\n"
        "class Guard:\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'jax','torch','numpy','optax','faiss'}:\n"
        "   raise AssertionError(fullname)\n"
        "sys.meta_path.insert(0,Guard())\n"
        f"sys.argv=['metrics.a4',*({command!r}.split()),'--help']\n"
        "runpy.run_module('metrics.a4',run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (command or "measure-latency") in result.stdout


def test_run_discovery_excludes_other_stages_and_incomplete_checkpoints(tmp_path):
    for name, stage, status in (
        ("a4-completed", "A4", "completed"),
        ("a4-running", "A4", "running"),
        ("a3", "A3", "completed"),
    ):
        path = tmp_path / name
        path.mkdir()
        write_json(path / "status.json", {"stage": stage, "status": status})
        (path / "checkpoints").mkdir()
        (path / "checkpoints/a4-final.npz").write_bytes(b"test fixture")
    assert len(list_runs(tmp_path)["runs"]) == 2
    assert latest_completed_run(tmp_path).name == "a4-completed"


def test_latest_requires_an_existing_completed_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="No completed A4 checkpoint"):
        latest_completed_run(tmp_path)


@pytest.mark.parametrize(
    "arguments", [["measure-latency"], ["measure-latency", "run", "--latest"]]
)
def test_measurement_run_selection_is_explicit(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_measurement_command_dispatches_without_retraining(
    monkeypatch, timing_file, capsys
):
    from stages.stage_A.a4 import benchmark

    calls = []

    def measure(run, **settings):
        calls.append((run, settings))
        return timing_file.parent

    monkeypatch.setattr(benchmark, "measure_latency", measure)
    assert main(["measure-latency", str(timing_file.parent), "--samples", "20"]) == 0
    assert calls[0][1]["samples"] == 20
    assert calls[0][1]["profile_transfers"]
    assert "CUDA copy intervals only" in capsys.readouterr().out


def test_retrieval_measurement_dispatches_to_checkpoint_evaluator(
    tmp_path, monkeypatch, capsys
):
    from stages.stage_A.a4 import evaluation

    calls = []
    write_json(
        tmp_path / "result.json", {"reports": [], "scope": "reference evaluation"}
    )

    def measure(run, probes, **settings):
        calls.append((run, probes, settings))
        return tmp_path

    monkeypatch.setattr(evaluation, "measure_retrieval", measure)
    assert main(["measure-retrieval", str(tmp_path), "--probes", "probes.json"]) == 0
    assert calls[0][1] == Path("probes.json")
    assert not calls[0][2]["allow_source_change"]
    assert "reference evaluation" in capsys.readouterr().out
