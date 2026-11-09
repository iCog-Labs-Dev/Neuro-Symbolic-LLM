"""Alignment-label admission and training-block identity."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stages.source_data import source_blocks
from stages.stage_A.a4.config import A4Config, load_config
from stages.stage_A.a4.data import load_data


@dataclass(frozen=True)
class Training:
    seq_len: int = 2
    batch_size: int = 1


def tokenizer(text):
    return {"input_ids": [int(value) for value in text.split()]}


@pytest.fixture
def inputs(tmp_path):
    args = SimpleNamespace(
        train_sources=tmp_path / "train.jsonl",
        validation_sources=tmp_path / "validation.jsonl",
        test_sources=tmp_path / "not-read.jsonl",
        alignment_labels=tmp_path / "labels.json",
        skip_test=True,
        max_train_rows=100,
        max_eval_rows=100,
        max_train_blocks=100,
        max_eval_blocks=100,
    )
    args.train_sources.write_text(
        json.dumps({"source_text": "1 2 3 4"}) + "\n", encoding="utf-8"
    )
    args.validation_sources.write_text(
        json.dumps({"source_text": "5 6"}) + "\n", encoding="utf-8"
    )
    _, _, metadata = source_blocks(args.train_sources, tokenizer, 2, 100, 100)
    labels = {
        "contract_version": "alignment-labels-v1",
        "bank_snapshot_sha256": "bank",
        "token_blocks_sha256": metadata["token_blocks_sha256"],
        "labels": [
            {"block_index": 0, "token_index": 1, "template_id": "a"},
            {"block_index": 1, "token_index": 0, "template_id": "b"},
        ],
    }
    bank = SimpleNamespace(
        input_snapshot_sha256="bank",
        records=[SimpleNamespace(template_id="a"), SimpleNamespace(template_id="b")],
    )
    return args, bank, labels


def admit(inputs):
    args, bank, labels = inputs
    args.alignment_labels.write_text(json.dumps(labels), encoding="utf-8")
    return load_data(args, tokenizer, bank, Training())


def test_targets_follow_shuffled_block_indices(inputs):
    data = admit(inputs)
    positions, targets = data.batch_labels(np.array([1, 0]))
    np.testing.assert_array_equal(positions, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(targets, [1, 0])
    assert data.test is None
    assert data.provenance["alignment"]["labelled_blocks"] == 2


@pytest.mark.parametrize(
    "field", ["bank_snapshot_sha256", "token_blocks_sha256", "contract_version"]
)
def test_rejects_mismatched_manifest(inputs, field):
    inputs[2][field] = "wrong"
    with pytest.raises(ValueError):
        admit(inputs)


def test_rejects_missing_labels_instead_of_ce_fallback(inputs):
    inputs[2]["labels"].pop()
    with pytest.raises(ValueError, match="lack firing labels"):
        admit(inputs)


@pytest.mark.parametrize("payload", [[], {"labels": None}])
def test_rejects_malformed_manifest(inputs, payload):
    args, bank, _ = inputs
    args.alignment_labels.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="labels list"):
        load_data(args, tokenizer, bank, Training())


def test_rejects_incomplete_label_records(inputs):
    inputs[2]["labels"][0].pop("template_id")
    with pytest.raises(ValueError, match="require block"):
        admit(inputs)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "outside", "boolean"])
def test_rejects_invalid_targets(inputs, mutation):
    labels = inputs[2]["labels"]
    if mutation == "duplicate":
        labels.append(dict(labels[0]))
    elif mutation == "unknown":
        labels[0]["template_id"] = "absent"
    elif mutation == "outside":
        labels[0]["token_index"] = 2
    else:
        labels[0]["block_index"] = True
    with pytest.raises(ValueError):
        admit(inputs)


def test_rejects_split_overlap(inputs):
    inputs[0].validation_sources.write_text(
        inputs[0].train_sources.read_text(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="multiple splits"):
        admit(inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("key_weight", 0),
        ("value_weight", -1),
        ("key_weight", float("nan")),
        ("value_weight", True),
        ("top_m", 0),
        ("minimum_templates", True),
    ],
)
def test_alignment_configuration_requires_valid_weights(field, value):
    with pytest.raises(ValueError):
        A4Config(Path("a3.yaml"), Training(), **{field: value})


def test_configuration_serializes_objective():
    cfg = A4Config(Path("a3.yaml"), Training(), key_weight=0.5, value_weight=2)
    assert cfg.to_dict()["objective"] == {"key_weight": 0.5, "value_weight": 2}


def test_configuration_rejects_unused_schema_before_runtime_import(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("stage: A4\noptimization: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires sections"):
        load_config(path)
