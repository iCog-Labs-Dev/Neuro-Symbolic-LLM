"""Reference retrieval, replay admission and timing schedules; no GPU qualification."""

import copy
import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from stages.stage_A.a4.benchmark import (
    _paired_samples,
    measure_latency,
    require_idle_gpu,
    training_state_comparison,
)
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE
from stages.stage_A.a4.evaluation import (
    score_rankings,
    score_support_prior,
    validate_probes,
)
from stages.stage_A.a4.replay import (
    digest,
    recorded_training_state_source,
    source_changes,
    training_arguments,
)


def test_known_negative_probes_are_not_hidden_from_precision():
    result = score_rankings([["a", "b"], ["x", "y"]], [["a", "z"], []], [1, 2])
    assert result[0]["all_probe_micro_precision"] == 0.5
    assert result[1]["all_probe_micro_precision"] == 0.25
    assert result[1]["positive_only"]["macro_recall"] == 0.5
    assert result[1]["known_negative_probes"] == 1
    assert result[1]["negative_probes_with_returns"] == 1


def test_rejects_unmeasured_retrieval_depth_and_empty_positives():
    with pytest.raises(ValueError, match="depth"):
        score_rankings([["a"]], [["a"]], [8])
    with pytest.raises(ValueError, match="No positive"):
        score_rankings([["a"]], [[]], [1])


def test_support_prior_uses_bank_support_and_deterministic_ties():
    patterns = [
        SimpleNamespace(id=name, expensive_trace={"binding_types": [None] * count})
        for name, count in [("z", 2), ("b", 3), ("a", 3)]
    ]
    left = score_support_prior(patterns, [["z"]], [1, 2])
    right = score_support_prior(patterns, [["b"]], [1, 2])
    assert left["top_ids"] == right["top_ids"] == ["a", "b"]
    assert left["top_support_counts"] == [3, 3]


@pytest.fixture
def probe_input(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"bank_source_text_sha256": []}))
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"document_id": "train", "source_text": "training"}))
    replay = SimpleNamespace(
        arguments=SimpleNamespace(alignment_labels=labels, train_sources=train),
        bank=SimpleNamespace(
            input_snapshot_sha256="bank",
            records=[SimpleNamespace(template_id="a")],
            patterns=[
                SimpleNamespace(
                    provenance=SimpleNamespace(source_document_ids=["bank-source"])
                )
            ],
        ),
        config=SimpleNamespace(training=SimpleNamespace(seq_len=4, model="model")),
    )
    text = "new"
    sha = hashlib.sha256(text.encode()).hexdigest()
    payload = {
        "contract_version": "a4-retrieval-probes-v1",
        "split": "validation",
        "label_kind": "reference_graph_firing",
        "review_status": "reference annotations",
        "bank_snapshot": "bank",
        "seq_len": 4,
        "model": "model",
        "inputs": [],
        "documents": [
            {
                "corpus_id": "reference",
                "document_id": "new",
                "source_text": text,
                "text_sha256": sha,
            }
        ],
        "probes": [
            {
                "corpus_id": "reference",
                "document_id": "new",
                "window_start": 0,
                "window_end": 3,
                "token_index": 2,
                "input_ids": [1, 2, 3],
                "source_text_sha256": sha,
                "relevant_ids": ["a"],
            }
        ],
    }

    def tokenizer(text):
        return {"input_ids": list(range(1, len(text) + 1))}

    return payload, replay, tokenizer


def test_probe_data_retokenizes_and_checks_relevance(probe_input):
    payload, replay, tokenizer = probe_input
    assert len(validate_probes(payload, replay, tokenizer)) == 1
    payload["probes"][0]["relevant_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="relevance"):
        validate_probes(payload, replay, tokenizer)


@pytest.mark.parametrize("name", ["train", "bank-source"])
def test_identity_overlap_rejected_even_when_text_differs(probe_input, name):
    payload, replay, tokenizer = probe_input
    payload["documents"][0]["document_id"] = name
    with pytest.raises(ValueError, match="overlaps"):
        validate_probes(payload, replay, tokenizer)


@pytest.mark.parametrize("change", ["future", "tokens", "duplicate"])
def test_future_or_modified_or_repeated_windows_rejected(probe_input, change):
    payload, replay, tokenizer = probe_input
    if change == "future":
        payload["probes"][0]["token_index"] = 1
    elif change == "tokens":
        payload["probes"][0]["input_ids"] = [9, 2, 3]
    else:
        payload["probes"].append(copy.deepcopy(payload["probes"][0]))
    with pytest.raises(ValueError):
        validate_probes(payload, replay, tokenizer)


def test_unmarked_runs_do_not_acquire_the_new_training_protocol():
    assert recorded_training_state_source({}, {}) == "unspecified-legacy"
    assert training_state_comparison(
        {}, {}, mode="deployment", allow_source_change=False
    ) == {
        "recorded": "unspecified-legacy",
        "runtime": TRAINING_STATE_SOURCE,
        "matches": False,
    }


@pytest.mark.parametrize("mode", ["training", "both"])
def test_legacy_training_comparison_requires_explicit_opt_in(mode):
    with pytest.raises(ValueError, match="Recorded training state source differs"):
        training_state_comparison({}, {}, mode=mode, allow_source_change=False)
    assert not training_state_comparison({}, {}, mode=mode, allow_source_change=True)[
        "matches"
    ]


def test_new_training_protocol_is_recorded_consistently():
    record = {"training_state_source": TRAINING_STATE_SOURCE}
    assert training_state_comparison(
        record, record, mode="both", allow_source_change=False
    )["matches"]


@pytest.mark.parametrize(
    "meta,result",
    [
        ({"training_state_source": TRAINING_STATE_SOURCE}, {}),
        ({}, {"training_state_source": TRAINING_STATE_SOURCE}),
        ({"training_state_source": ""}, {"training_state_source": ""}),
        ({"training_state_source": 1}, {"training_state_source": 1}),
    ],
)
def test_inconsistent_training_protocol_is_rejected(meta, result):
    with pytest.raises(ValueError, match="Training state source"):
        recorded_training_state_source(meta, result)


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
