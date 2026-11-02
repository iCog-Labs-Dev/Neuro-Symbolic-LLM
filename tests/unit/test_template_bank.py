"""Pattern/embedding ID association, vector integrity and input validation."""

import json

import numpy as np
import pytest

from residual.symbolic_head.contracts.template_record import TemplateRecord
from symbolic_memory.__main__ import main
from symbolic_memory.bank import load_bank, read_patterns
from symbolic_memory.storage import _decode, _encode


@pytest.fixture
def artifact(tmp_path):
    pattern = {
        "contract_version": "pattern-record-v1",
        "id": "pattern-agent",
        "canonical_term": "(Agent $v0 $v1)",
        "interface_variables": ["$v0", "$v1"],
        "existential_variables": [],
        "type_constraints": {"$v0": "Event", "$v1": "Entity"},
        "semantic_level": "extracted",
        "ontology_groundings": ["upper-v1:Agent"],
        "source_spans": [{"document_id": "doc-1", "start": 0, "end": 4}],
        "parser_confidences": [0.9],
        "provenance": {
            "corpus_id": "unit-fixture",
            "parser_version": "fixture-parser-v1",
            "ontology_versions": ["upper-v1"],
            "source_document_ids": ["doc-1"],
            "miner_version": "fixture-miner-v1",
        },
        "generator_history": ["fixture"],
        "cheap_features": {"clause_count": 1},
        "expensive_endpoint": {"observed_support": 1},
        "status": "candidate",
    }
    embedding = {
        "pattern_id": pattern["id"],
        "key": [3.0, 4.0],
        "value": [0.0, 2.0],
        "embedding_dim": 2,
        "encoder_model": "fixture-not-a-model",
        "pooling": "fixture",
    }
    _write(tmp_path / "patterns.jsonl", [pattern])
    _write(tmp_path / "embeddings.jsonl", [embedding])
    return tmp_path, pattern, embedding


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load(directory):
    return load_bank(directory / "patterns.jsonl", directory / "embeddings.jsonl")


def test_existing_storage_type_and_unchanged_vectors(artifact):
    directory, _, _ = artifact
    bank = _load(directory)
    record = bank.records[0]
    assert record.__class__.__module__.startswith("residual.symbolic_head.")
    assert record.template_id == "pattern-agent"
    assert record.metta_expr == "(Agent $v0 $v1)"
    assert bank.vector_dim == 2
    assert record.key_vector.dtype == np.float32
    np.testing.assert_array_equal(record.key_vector, [3, 4])
    np.testing.assert_array_equal(record.val_vector, [0, 2])
    assert not record.key_vector.flags.writeable
    assert not record.val_vector.flags.writeable


@pytest.mark.parametrize("dimension", [4, 32])
def test_storage_codec_preserves_float32(dimension):
    values = np.array(
        [
            0.05010432004928589,
            -0.15998779237270355,
            np.nextafter(np.float32(0), np.float32(1)),
            np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    key = np.resize(values, dimension)
    value = -key
    record = TemplateRecord("pattern-agent", "(Agent $v0 $v1)", key, value)
    restored = _decode(_encode(record, dimension), dimension)
    assert restored.template_id == "pattern-agent"
    assert restored.metta_expr == "(Agent $v0 $v1)"
    np.testing.assert_array_equal(restored.key_vector, key)
    np.testing.assert_array_equal(restored.val_vector, value)


def test_join_uses_ids_not_embedding_row_order(artifact):
    directory, pattern, embedding = artifact
    second_pattern = {**pattern, "id": "pattern-second"}
    second_embedding = {**embedding, "pattern_id": "pattern-second", "value": [9, 8]}
    _write(directory / "patterns.jsonl", [pattern, second_pattern])
    _write(directory / "embeddings.jsonl", [second_embedding, embedding])
    bank = _load(directory)
    assert [item.template_id for item in bank.records] == [
        "pattern-agent",
        "pattern-second",
    ]
    np.testing.assert_array_equal(bank.records[0].val_vector, [0, 2])
    np.testing.assert_array_equal(bank.records[1].val_vector, [9, 8])


def test_fingerprints_identify_actual_input_bytes(artifact):
    directory, _, embedding = artifact
    first = _load(directory)
    assert first.input_snapshot_sha256 == _load(directory).input_snapshot_sha256
    embedding["value"] = [7, 8]
    _write(directory / "embeddings.jsonl", [embedding])
    changed = _load(directory)
    assert first.patterns_sha256 == changed.patterns_sha256
    assert first.embeddings_sha256 != changed.embeddings_sha256
    assert first.input_snapshot_sha256 != changed.input_snapshot_sha256


def test_metadata_is_optional_and_changes_input_identity(artifact):
    directory, _, embedding = artifact
    before = _load(directory)
    embedding["producer_specific_setting"] = {"revision": "new"}
    _write(directory / "embeddings.jsonl", [embedding])
    assert _load(directory).input_snapshot_sha256 != before.input_snapshot_sha256
    _write(
        directory / "embeddings.jsonl",
        [{k: embedding[k] for k in ("pattern_id", "key", "value")}],
    )
    assert _load(directory).vector_dim == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"key": [0, 0]}, "non-zero"),
        ({"key": [float("nan"), 0]}, "finite"),
        ({"value": [0, float("inf")]}, "finite"),
        ({"key": [1e100, 0]}, "overflow"),
        ({"value": [1]}, "dim mismatch"),
        ({"key": []}, "nonempty"),
        ({"value": [[1, 2]]}, "numbers"),
        ({"value": ["1", "2"]}, "numbers"),
        ({"key": [True, False]}, "numbers"),
        ({"key": "not-a-vector"}, "nonempty"),
        ({"pattern_id": "unknown"}, "unknown embedding"),
        ({"pattern_id": " "}, "nonempty"),
        ({"embedding_dim": 3}, "embedding_dim"),
        ({"embedding_dim": True}, "embedding_dim"),
    ],
)
def test_incompatible_embeddings_report_line_number(artifact, changes, message):
    directory, _, embedding = artifact
    _write(directory / "embeddings.jsonl", [{**embedding, **changes}])
    with pytest.raises(ValueError, match=f"embeddings.jsonl:1:.*{message}"):
        _load(directory)


def test_dimension_is_consistent_across_rows(artifact):
    directory, pattern, embedding = artifact
    _write(directory / "patterns.jsonl", [pattern, {**pattern, "id": "second"}])
    other = {"pattern_id": "second", "key": [1, 2, 3], "value": [4, 5, 6]}
    _write(directory / "embeddings.jsonl", [embedding, other])
    with pytest.raises(ValueError, match="embeddings.jsonl:2:.*dim mismatch"):
        _load(directory)


@pytest.mark.parametrize("kind", ["patterns", "embeddings"])
def test_duplicate_ids_are_rejected(artifact, kind):
    directory, pattern, embedding = artifact
    row = pattern if kind == "patterns" else embedding
    _write(directory / f"{kind}.jsonl", [row, row])
    with pytest.raises(ValueError, match=f"{kind}.jsonl:2: duplicate"):
        _load(directory)


def test_missing_embeddings_are_not_synthesized(artifact):
    directory, _, _ = artifact
    _write(directory / "embeddings.jsonl", [])
    with pytest.raises(ValueError, match="missing embeddings"):
        _load(directory)


@pytest.mark.parametrize("payload", ["[]\n", "{bad json}\n", "null\n"])
def test_invalid_json_rows_are_rejected(artifact, payload):
    directory, _, _ = artifact
    (directory / "embeddings.jsonl").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="embeddings.jsonl:1:"):
        _load(directory)


def test_blank_lines_and_utf8_bom_are_supported(artifact):
    directory, _, _ = artifact
    for name in ("patterns.jsonl", "embeddings.jsonl"):
        path = directory / name
        original = path.read_text(encoding="utf-8")
        path.write_text("\ufeff\n" + original + "\n", encoding="utf-8")
    assert len(_load(directory).records) == 1


def test_empty_patterns_fail_clearly(artifact):
    directory, _, _ = artifact
    _write(directory / "patterns.jsonl", [])
    with pytest.raises(ValueError, match="no pattern records"):
        read_patterns(directory / "patterns.jsonl")


def test_cli_inspection_is_read_only_and_reports_real_dimensions(artifact, capsys):
    directory, _, _ = artifact
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert (
        main(
            [
                "inspect-bank",
                "--patterns",
                str(directory / "patterns.jsonl"),
                "--embeddings",
                str(directory / "embeddings.jsonl"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["record_count"] == 1
    assert output["vector_dimension"] == 2
    assert output["normalization_applied"] is False
    assert len(output["input_snapshot_sha256"]) == 64
    assert original == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_cli_patterns_does_not_claim_embeddings_exist(artifact, capsys):
    directory, _, _ = artifact
    assert main(["inspect-patterns", "--input", str(directory / "patterns.jsonl")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["contains_vectors"] is False
    assert output["statuses"] == ["candidate"]


def test_cli_missing_input_has_no_success_output(tmp_path, capsys):
    assert (
        main(
            [
                "inspect-bank",
                "--patterns",
                str(tmp_path / "missing"),
                "--embeddings",
                str(tmp_path / "absent"),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert not output.out
    assert "error:" in output.err
