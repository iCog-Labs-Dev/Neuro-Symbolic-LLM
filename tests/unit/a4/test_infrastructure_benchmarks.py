"""Infrastructure protocol checks without MORK, CUDA or model weights."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stages.stage_A.a4.benchmarks.cli import main, parse_args
from stages.stage_A.a4.benchmarks.common import (
    CpuBackend,
    digest,
    paired_samples,
    query_batch,
    summarize,
    verify_outputs,
    write_artifacts,
)
from stages.stage_A.a4.benchmarks.search import neighbor_agreement


def arguments(*extra):
    return [
        "search",
        "--patterns",
        "patterns.jsonl",
        "--embeddings",
        "embeddings.jsonl",
        "--output",
        "runs/uncreated-benchmark-test",
        *extra,
    ]


def test_cli_help_has_no_hardware_dependencies():
    completed = subprocess.run(
        [sys.executable, "-m", "stages.stage_A.a4.benchmarks", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "native-prototype" in completed.stdout
    assert "--mork-url" in completed.stdout


def test_defaults_and_overrides():
    args = parse_args(arguments())
    assert (args.samples, args.warmups, args.threads, args.query_count, args.top_m) == (
        100,
        10,
        2,
        256,
        8,
    )
    assert args.ef_search == [16, 50, 100]
    assert parse_args(arguments("--samples", "3")).samples == 3


@pytest.mark.parametrize(
    "option",
    ["--samples", "--warmups", "--threads", "--query-count", "--top-m", "--ef-search"],
)
def test_rejects_nonpositive_settings(option):
    with pytest.raises(SystemExit):
        parse_args(arguments(option, "0"))


def test_rejects_existing_output(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(arguments("--output", str(tmp_path)))


def test_query_control_and_loaded_shape(tmp_path):
    records = [
        SimpleNamespace(key_vector=np.array([i + 1, 1], dtype=np.float32))
        for i in range(4)
    ]
    control = query_batch(records, 3, None)
    np.testing.assert_array_equal(control, [[1, 1], [2, 1], [4, 1]])
    path = tmp_path / "queries.npy"
    np.save(path, control)
    np.testing.assert_array_equal(query_batch(records, 3, path), control)
    with pytest.raises(ValueError, match="float32"):
        query_batch(records, 2, path)
    np.save(path, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="nonzero"):
        query_batch(records, 3, path)


def test_sampling_excludes_warmups_and_rotates_order():
    order = []

    def call(name):
        order.append(name)
        return {"cpu_index_seconds": 0.0}

    samples = paired_samples(
        {name: lambda name=name: call(name) for name in "abc"}, 3, 2
    )
    assert order[:6] == ["a", "a", "b", "b", "c", "c"]
    assert order[6:] == ["a", "b", "c", "a", "c", "b", "c", "a", "b"]
    assert len(samples) == 9
    assert all(
        row["search_excluded_seconds"] == row["total_seconds"] for row in samples
    )


@pytest.mark.parametrize(
    "details",
    [
        {"cpu_index_seconds": -1},
        {"cpu_index_seconds": 1e9},
        {"x_seconds": float("nan")},
        {"total_seconds": 0},
    ],
)
def test_invalid_timing_rejected(details):
    with pytest.raises(ValueError):
        paired_samples({"invalid": lambda: details}, 1, 1)


def test_summary_has_milliseconds_and_empirical_percentiles():
    rows = summarize(
        [{"variant": "a", "total_seconds": t} for t in (0.001, 0.002, 0.003)]
    )
    assert rows[0]["samples"] == 3
    assert rows[0]["mean_ms"] == pytest.approx(2)
    assert rows[0]["p50_ms"] == pytest.approx(2)
    assert rows[0]["p95_ms"] == pytest.approx(2.9)


def test_equivalence_includes_indices_and_dtypes():
    expected = (
        np.ones((1, 2, 3), np.float32),
        np.ones((1, 2, 3), np.float32),
        np.ones((1, 2), np.float32),
        np.array([[0, 1]], np.int32),
    )
    verify_outputs(expected, expected)
    with pytest.raises(AssertionError):
        verify_outputs(expected, (*expected[:3], np.array([[1, 0]], np.int32)))
    with pytest.raises(ValueError, match="dtypes"):
        verify_outputs(expected, (*expected[:3], expected[3].astype(np.int64)))


def test_tie_agreement_is_not_strict_id_agreement():
    exact = SimpleNamespace(scores=np.array([[1.0, 0.5]]), template_ids=[["a", "b"]])
    actual = SimpleNamespace(scores=np.array([[1.0, 0.5]]), template_ids=[["a", "c"]])
    result = neighbor_agreement(actual, exact)
    assert result["strict_id_agreement"] == 0.5
    assert result["tie_aware_score_agreement"] == 1


def test_cpu_threads_set_in_executing_thread_and_restored(monkeypatch):
    threads = [7]
    fake = SimpleNamespace(
        omp_get_max_threads=lambda: threads[0],
        omp_set_num_threads=lambda n: threads.__setitem__(0, n),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake)

    def query(q, m):
        assert threads[0] == 2
        return q

    backend = CpuBackend(SimpleNamespace(query_top_k=query), {}, 2)
    assert backend.query_top_k("query", 8) == "query"
    assert backend.calls == 1
    assert threads[0] == 7


def test_result_artifacts_do_not_claim_a_model_gate(tmp_path):
    result = {
        "status": "completed",
        "under_five_percent_verified": False,
        "model_time_denominator": None,
    }
    write_artifacts(
        tmp_path, result, [{"variant": "a", "trial": 0, "total_seconds": 0.001}]
    )
    saved = json.loads((tmp_path / "result.json").read_text())
    assert not saved["under_five_percent_verified"]
    assert saved["model_time_denominator"] is None
    assert saved["rows"][0]["p95_ms"] == 1
    assert (tmp_path / "samples.csv").is_file()
    assert len(digest(tmp_path / "result.json")) == 64


def test_benchmark_code_has_no_local_experiment_paths():
    directory = Path(__file__).resolve().parents[3] / "stages/stage_A/a4/benchmarks"
    assert directory.is_dir()
    for source in directory.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "data/check" not in text
        assert "/home/jovyan" not in text


@pytest.mark.parametrize("fail", [False, True])
def test_entrypoint_reads_existing_bank_and_records_failure(
    tmp_path, monkeypatch, fail
):
    from stages.stage_A.a4.benchmarks import common, search

    record = SimpleNamespace(key_vector=np.array([1, 1], dtype=np.float32))
    bank = SimpleNamespace(records=[record])
    reads = []

    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load_index(self, supplied):
            assert supplied is bank
            reads.append(supplied)
            return "index"

    def measure(args, index, queries, result, samples):
        assert index == "index"
        samples.append({"variant": "a", "trial": 0, "total_seconds": 0.001})
        if fail:
            raise RuntimeError("measurement failed")

    monkeypatch.setitem(
        sys.modules,
        "symbolic_memory.bank",
        SimpleNamespace(load_bank=lambda *args: bank),
    )
    monkeypatch.setitem(
        sys.modules, "symbolic_memory.storage", SimpleNamespace(MorkTemplateStore=Store)
    )
    monkeypatch.setattr(common, "metadata", lambda *args: {"status": "running"})
    monkeypatch.setattr(search, "measure", measure)
    output = tmp_path / "output"
    argv = arguments("--output", str(output), "--top-m", "1")
    if fail:
        with pytest.raises(RuntimeError, match="measurement failed"):
            main(argv)
    else:
        assert main(argv) == 0
    assert len(reads) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == ("failed" if fail else "completed")
    assert result["rows"][0]["samples"] == 1


def test_completed_samples_survive_a_later_variant_failure():
    rows = []
    calls = [0]

    def broken():
        calls[0] += 1
        if calls[0] > 1:
            raise RuntimeError("second call failed")
        return {}

    with pytest.raises(RuntimeError, match="second call"):
        paired_samples({"good": lambda: {}, "broken": broken}, 2, 1, rows)
    assert len(rows) == 1
    assert rows[0]["variant"] == "good"


def test_search_sweep_with_real_faiss_when_available():
    pytest.importorskip("faiss")
    pytest.importorskip("jax")  # Imported by the repository's residual package.
    from residual.symbolic_head.contracts.template_record import TemplateRecord
    from residual.symbolic_head.retrieval.faiss_index import FaissTemplateIndex
    from stages.stage_A.a4.benchmarks.search import measure

    records = [
        TemplateRecord(
            str(i),
            f"(Fact {i})",
            np.array(vector, np.float32),
            np.array(vector, np.float32),
        )
        for i, vector in enumerate(([1, 0], [0, 1], [-1, 0], [0, -1]))
    ]
    index = FaissTemplateIndex(key_dim=2)
    index.rebuild(records)
    effort = index._faiss_index.hnsw.efSearch
    args = SimpleNamespace(threads=1, top_m=2, ef_search=[4, 8], samples=2, warmups=1)
    result, samples = {}, []
    measure(args, index, np.array([[1, 0]], np.float32), result, samples)
    assert len(samples) == 6
    assert len(result["ann_quality"]) == 2
    assert all(row["tie_aware_score_agreement"] == 1 for row in result["ann_quality"])
    assert index._faiss_index.hnsw.efSearch == effort
