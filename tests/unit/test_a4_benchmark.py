"""Host-side A4 replay admission and timing-schedule tests; not GPU qualification."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from stages.stage_A.a4.benchmark import (
    _paired_samples,
    measure_latency,
    require_idle_gpu,
)
from stages.stage_A.a4.replay import digest, source_changes, training_arguments


def test_recorded_inputs_resolve_against_training_directory(tmp_path):
    meta = {
        "working_directory": str(tmp_path),
        "training_arguments": {
            "patterns": "inputs/patterns.jsonl",
            "native_library": "native/retrieval.so",
            "test_sources": None,
            "max_grad_norm": 1,
        },
    }
    args = training_arguments(meta)
    assert args.patterns == tmp_path / "inputs/patterns.jsonl"
    assert args.native_library == tmp_path / "native/retrieval.so"
    assert args.test_sources is None
    assert meta["training_arguments"]["patterns"] == "inputs/patterns.jsonl"


def test_changed_source_is_detected(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("original", encoding="utf-8")
    meta = {"source_sha256": {"model.py": digest(path)}}
    assert source_changes(meta, root=tmp_path) == []
    path.write_text("changed", encoding="utf-8")
    assert source_changes(meta, root=tmp_path) == ["model.py"]


def test_recorded_source_cannot_escape_checkout(tmp_path):
    with pytest.raises(ValueError, match="leaves the checkout"):
        source_changes({"source_sha256": {"../outside.py": "unused"}}, root=tmp_path)


@pytest.mark.parametrize(
    "pids,allowed", [("", True), (str(os.getpid()), True), ("99999999", False)]
)
def test_only_own_gpu_process_is_allowed(monkeypatch, pids, allowed):
    monkeypatch.setattr(
        "stages.stage_A.a4.benchmark.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=pids),
    )
    if allowed:
        require_idle_gpu()
    else:
        with pytest.raises(RuntimeError, match="Another GPU process"):
            require_idle_gpu()


@pytest.mark.parametrize(
    "settings",
    [
        {"samples": 0},
        {"samples": 2, "batches": 3},
        {"warmups": 0},
        {"mode": "invalid"},
    ],
)
def test_invalid_settings_fail_before_loading_models(tmp_path, settings):
    with pytest.raises(ValueError):
        measure_latency(tmp_path, **settings)


class HostSchedule:
    def __init__(self):
        self.values = dict.fromkeys(
            (
                "calls",
                "d2h_seconds",
                "h2d_seconds",
                "host_search_seconds",
                "cpu_index_seconds",
            ),
            0.0,
        )
        self.info = {"return_mode": "sync", "max_top_m": 2, "record_count": 10}
        self.query = SimpleNamespace(size=8, shape=(2, 4))
        self.captured = None
        self.order = []

    def stats(self):
        return dict(self.values)

    @contextmanager
    def capture_queries(self):
        self.captured = []
        try:
            yield self.captured
        finally:
            self.captured = None

    def call(self, query):
        self.values["calls"] += 1
        for field, seconds in (
            ("d2h_seconds", 0.0001),
            ("h2d_seconds", 0.0001),
            ("host_search_seconds", 0.00025),
            ("cpu_index_seconds", 0.0002),
        ):
            self.values[field] += seconds
        if self.captured is not None:
            self.captured.append(query)

    def invoke(self, mode, batch):
        self.order.append("step")
        self.call(self.query)
        self.call(self.query)

    def bridge(self, query):
        self.order.append("bridge")
        self.call(query)


def test_paired_schedule_excludes_warmups_and_alternates_call_order():
    retriever = HostSchedule()
    ticks = iter(index * 0.001 for index in range(8))
    rows = _paired_samples(
        "deployment",
        retriever.invoke,
        retriever.bridge,
        retriever,
        [object()],
        2,
        1,
        lambda value: value,
        lambda value: value,
        expected_calls=2,
        clock=lambda: next(ticks),
    )
    assert len(rows) == 2
    assert retriever.order[-6:] == [
        "step",
        "bridge",
        "bridge",
        "bridge",
        "bridge",
        "step",
    ]
    assert all(row["step_seconds"] == pytest.approx(0.001) for row in rows)
    assert all(row["retrieval_calls_per_step"] == 2 for row in rows)
    assert rows[0]["query_bytes"] == 64
    assert rows[0]["result_bytes"] == 320


def test_capture_must_cover_all_attached_layers():
    retriever = HostSchedule()
    with pytest.raises(RuntimeError, match="attached layers"):
        _paired_samples(
            "deployment",
            retriever.invoke,
            retriever.bridge,
            retriever,
            [object()],
            2,
            1,
            lambda value: value,
            lambda value: value,
            expected_calls=3,
        )
