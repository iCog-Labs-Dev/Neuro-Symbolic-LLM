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
        {
            "event": "train",
            "step": 5,
            "task": 1,
            "key": 2,
            "value": 3,
            "total": 6,
            "gradient_norm": 10,
            "step_time": 0.03,
        },
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
    assert "Total loss=6.000000" in printed
    assert "before clipping)=10.000000" in printed
    assert "30.000 ms" in printed
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


@pytest.mark.parametrize(
    "command", ["", "measure-latency", "measure-retrieval", "report"]
)
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
        cwd=Path(__file__).resolve().parents[4],
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
        tmp_path / "result.json",
        {
            "status": "completed",
            "contract_version": "a4-retrieval-evaluation-v1",
            "reports": [],
            "scope": "reference evaluation",
        },
    )

    def measure(run, probes, **settings):
        calls.append((run, probes, settings))
        return tmp_path

    monkeypatch.setattr(evaluation, "measure_retrieval", measure)
    assert main(["measure-retrieval", str(tmp_path), "--probes", "probes.json"]) == 0
    assert calls[0][1] == Path("probes.json")
    assert not calls[0][2]["allow_source_change"]
    assert "reference evaluation" in capsys.readouterr().out


def test_representation_report_prints_alignment_and_undefined_factors(tmp_path, capsys):
    write_json(
        tmp_path / "result.json",
        {
            "status": "completed",
            "contract_version": "a4-representation-v1",
            "reports": [
                {
                    "condition": "trained",
                    "ce": 2,
                    "ppl": 7.389,
                    "kl_base_to_model": 0.1,
                    "next_token_accuracy": 0.5,
                    "next_token_targets": 20,
                    "layers": [
                        {
                            "layer": 6,
                            "reference_pair_key_nll": 4,
                            "reference_pair_value_squared_error": 5,
                            "alignment_reference_pairs": 10,
                            "retrieval": [],
                            "query_clusters": {"silhouette": -0.1},
                            "writeback_to_hidden_norm_mean": 0.2,
                            "writeback_nonzero_fraction": 1,
                            "factor_organization": {
                                "fracture": [0.5, 1, None],
                                "mean_off_diagonal_overlap": 0.6,
                                "factors": 3,
                                "components": 2,
                                "zero_effect_factor_indices": [2],
                            },
                        }
                    ],
                }
            ],
        },
    )
    write_json(
        tmp_path / "capacity.json",
        {
            "6": {
                "controls": [
                    {
                        "dimensions": 16,
                        "bank_size": 100,
                        "available_reference_pair_fraction": 0.8,
                        "scores": [
                            {
                                "top_k": 8,
                                "all_probe_precision": 0.25,
                                "positive_macro_recall": 0.5,
                            }
                        ],
                    }
                ],
                "ann_exact_id_agreement_at_8": 0.75,
            }
        },
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main(["report", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "key NLL=4.000000" in text
    assert "value squared error=5.000000" in text
    assert "mean fracture=0.750000" in text
    assert "mean pair overlap=0.600000" in text
    assert "zero-effect factors=1" in text
    assert "Query silhouette=-0.100000" in text
    assert "dimensions=16 bank=100" in text
    assert "reference coverage=80.00%" in text
    assert "ID agreement=75.00%" in text
    assert "no new inference" in text
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_saved_retrieval_report_preserves_positive_and_negative_denominators(
    tmp_path, capsys
):
    path = write_json(
        tmp_path / "result.json",
        {
            "status": "completed",
            "contract_version": "a4-retrieval-evaluation-v1",
            "reports": [
                {
                    "condition": "trained",
                    "causal_window_ce": 2,
                    "next_token_targets": 20,
                    "layers": [
                        {
                            "layer": 6,
                            "scores": [
                                {
                                    "top_k": 8,
                                    "positive_probes": 3,
                                    "known_negative_probes": 1,
                                    "all_probe_micro_precision": 0.25,
                                    "positive_only": {
                                        "macro_recall": 0.5,
                                        "hit_rate": 1,
                                        "mean_reciprocal_rank": 0.75,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    assert main(["report", str(path)]) == 0
    text = capsys.readouterr().out
    assert "precision(all)=25.00%" in text
    assert "recall(positive)=50.00%" in text
    assert "3 positive, 1 known negative" in text


def test_saved_latency_report_reuses_sample_validation(timing_file, capsys):
    assert main(["report", str(timing_file)]) == 0
    text = capsys.readouterr().out
    assert "CUDA copy intervals only" in text
    assert "no new inference" in text


@pytest.mark.parametrize("kind", ["latency", "retrieval", "representation"])
def test_fresh_and_saved_evaluations_share_console_summary(
    kind, timing_file, monkeypatch, capsys
):
    from importlib import import_module

    module = import_module(
        "stages.stage_A.a4."
        + {
            "latency": "benchmark",
            "retrieval": "evaluation",
            "representation": "diagnostics",
        }[kind]
    )
    if kind != "latency":
        write_json(
            timing_file,
            {
                "status": "completed",
                "contract_version": {
                    "retrieval": "a4-retrieval-evaluation-v1",
                    "representation": "a4-representation-v1",
                }[kind],
                "reports": [],
                "scope": "Shared evaluation summary",
            },
        )
    monkeypatch.setattr(
        module, f"measure_{kind}", lambda *args, **kwargs: timing_file.parent
    )
    before = {path.name: path.read_bytes() for path in timing_file.parent.iterdir()}
    arguments = [f"measure-{kind}", str(timing_file.parent)]
    if kind != "latency":
        arguments.extend(["--probes", "probes.json"])
    assert main(arguments) == 0
    fresh = capsys.readouterr().out
    fresh = "\n".join(
        line for line in fresh.splitlines() if not line.startswith("Selected A4 run:")
    ).strip()
    assert main(["report", str(timing_file)]) == 0
    saved = capsys.readouterr().out.split("Saved evaluation:")[0].strip()
    assert fresh == saved
    assert {
        path.name: path.read_bytes() for path in timing_file.parent.iterdir()
    } == before


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "running", "contract_version": "a4-representation-v1"},
        {"status": "completed", "contract_version": "unrelated"},
    ],
)
def test_saved_report_rejects_incomplete_or_unknown_measurements(tmp_path, payload):
    path = write_json(tmp_path / "result.json", payload)
    with pytest.raises(SystemExit) as error:
        main(["report", str(path)])
    assert error.value.code == 2


def test_representation_cli_dispatches_without_loading_numeric_runtime(
    tmp_path, monkeypatch
):
    from types import ModuleType

    calls = []
    module = ModuleType("stages.stage_A.a4.diagnostics")

    def measure(run, probes, **kwargs):
        calls.append((run, probes, kwargs))
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "contract_version": "a4-representation-v1",
                    "reports": [],
                }
            )
        )
        return tmp_path

    module.measure_representation = measure
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert (
        main(
            [
                "measure-representation",
                str(tmp_path),
                "--probes",
                str(tmp_path / "probes.json"),
            ]
        )
        == 0
    )
    assert calls[0][2]["maximum_factors"] == 32
