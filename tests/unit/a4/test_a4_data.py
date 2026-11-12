"""Source preparation, alignment admission and A4 configuration."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.prepare_symbolic_corpus import file_digest, prepare_snapshot
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
        ("key_weight", -0.1),
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


@pytest.mark.parametrize("key_weight,value_weight", [(0, 1), (1, 0), (0, 0)])
def test_configuration_allows_alignment_ablation(key_weight, value_weight):
    cfg = A4Config(
        Path("a3.yaml"), Training(), key_weight=key_weight, value_weight=value_weight
    )
    assert cfg.to_dict()["objective"] == {
        "key_weight": key_weight,
        "value_weight": value_weight,
    }


def test_configuration_rejects_unused_schema_before_runtime_import(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("stage: A4\noptimization: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires sections"):
        load_config(path)


def prepare(tmp_path, records, **kwargs):
    snapshot = tmp_path / "raw.jsonl"
    snapshot.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return prepare_snapshot(
        snapshot,
        tmp_path / "out",
        name="example",
        dataset="owner/data",
        revision="pinned-revision",
        split="train",
        text_field="text",
        **kwargs,
    )


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_preserves_complete_theory_and_exact_whitespace(tmp_path):
    text = "Réa is blue.\n\nIf Réa is blue, Réa is kind.  "
    original = {"id": "theory1", "text": text, "answer": "True"}
    result = prepare(tmp_path, [original], parent_field="id")
    source = rows(tmp_path / "out/sources.jsonl")[0]
    assert set(source) == {"document_id", "corpus_id", "source_text"}
    assert source["source_text"] == text
    metadata = rows(tmp_path / "out/metadata.jsonl")[0]
    assert metadata["annotations"] == [{"snapshot_line": 1, "record": original}]
    assert result["policy"]["annotations_in_parser_input"] is False
    assert result["snapshot_sha256"] == file_digest(tmp_path / "raw.jsonl")
    for name, digest in result["files"].items():
        assert file_digest(tmp_path / "out" / name) == digest


def test_groups_contract_hypotheses_without_duplicating_source(tmp_path):
    records = [
        {"id": 34, "text": "The agreement survives.", "hypothesis": value}
        for value in ("A", "B")
    ]
    result = prepare(tmp_path, records, parent_field="id")
    assert result["input_rows"] == 2
    assert result["parent_count"] == result["included_parents"] == 1
    assert len(rows(tmp_path / "out/sources.jsonl")) == 1
    metadata = rows(tmp_path / "out/metadata.jsonl")[0]
    assert [entry["record"] for entry in metadata["annotations"]] == records


def test_preserves_idiom_group_identity(tmp_path):
    prepare(
        tmp_path,
        [
            {
                "text": "She broke the ice.",
                "idiom": "break the ice",
                "usage": "figurative",
            }
        ],
        split_group_field="idiom",
    )
    metadata = rows(tmp_path / "out/metadata.jsonl")[0]
    assert metadata["split_groups"] == ["break the ice"]
    assert metadata["annotations"][0]["record"]["usage"] == "figurative"


def test_excludes_long_parent_without_truncating_or_losing_annotations(tmp_path):
    text = "The complete source is longer than the configured limit."
    result = prepare(tmp_path, [{"text": text, "label": "neutral"}], max_chars=10)
    assert result["included_parents"] == 0
    assert result["excluded_parents"] == 1
    assert rows(tmp_path / "out/sources.jsonl") == []
    metadata = rows(tmp_path / "out/metadata.jsonl")[0]
    assert metadata["source_text"] == text
    assert metadata["included"] is False
    failure = rows(tmp_path / "out/excluded.jsonl")[0]
    assert failure["characters"] == len(text)


def test_rejects_inconsistent_text_for_one_parent_before_writing(tmp_path):
    with pytest.raises(ValueError, match="inconsistent source text"):
        prepare(
            tmp_path,
            [{"id": "a", "text": "First."}, {"id": "a", "text": "Second."}],
            parent_field="id",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("record", [{"text": ""}, {"text": 3}, {}, []])
def test_rejects_invalid_source_rows(tmp_path, record):
    with pytest.raises(ValueError):
        prepare(tmp_path, [record])


def test_missing_parent_id_is_not_replaced_silently(tmp_path):
    with pytest.raises(ValueError, match="parent identity"):
        prepare(tmp_path, [{"text": "A source."}], parent_field="missing")


def test_missing_group_is_not_replaced_silently(tmp_path):
    with pytest.raises(ValueError, match="split group"):
        prepare(tmp_path, [{"text": "A source."}], split_group_field="missing")


def test_output_directory_cannot_be_overwritten(tmp_path):
    (tmp_path / "out").mkdir()
    marker = tmp_path / "out/retained.txt"
    marker.write_text("retained")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, [{"text": "A source."}])
    assert marker.read_text() == "retained"


def test_identical_inputs_produce_identical_source_and_metadata_bytes(tmp_path):
    original = [{"id": "p1", "text": "A source.", "label": "positive"}]
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    a = prepare(first, original, parent_field="id")
    b = prepare(second, original, parent_field="id")
    assert a["files"] == b["files"]


def test_empty_snapshot_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no records"):
        prepare(tmp_path, [])
