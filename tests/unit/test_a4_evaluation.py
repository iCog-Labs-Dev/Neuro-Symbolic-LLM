"""Reference retrieval scoring and checkpoint-evaluation input checks."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from stages.stage_A.a4.evaluation import (
    score_rankings,
    score_support_prior,
    validate_probes,
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
