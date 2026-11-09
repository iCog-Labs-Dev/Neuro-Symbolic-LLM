"""Lossless context grouping, annotation isolation and exclusion accounting."""

from __future__ import annotations

import json

import pytest

from scripts.prepare_symbolic_corpus import file_digest, prepare_snapshot


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
