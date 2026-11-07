"""Training and label preparation share source ordering and token fingerprints."""

import hashlib
import json

import numpy as np
import pytest

from stages.source_data import source_blocks


def tokenizer(text):
    return {"input_ids": [int(token) for token in text.split()]}


def sources(tmp_path, rows):
    path = tmp_path / "sources.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_independent_blocks_deduplication_and_fingerprint(tmp_path):
    path = sources(
        tmp_path,
        [
            {"document_id": "a", "source_text": "1 2 3 4 5"},
            {"document_id": "duplicate", "source_text": "1 2 3 4 5"},
            {"document_id": "short", "source_text": "8"},
            {"document_id": "b", "source_text": "6 7 8 9"},
        ],
    )
    blocks, hashes, metadata = source_blocks(path, tokenizer, 2, 10, 10)
    np.testing.assert_array_equal(blocks, [[1, 2], [3, 4], [6, 7], [8, 9]])
    assert len(hashes) == 2
    assert [row["block_start"] for row in metadata["selected"]] == [0, 2]
    assert [row["row"] for row in metadata["selected"]] == [1, 4]
    assert (
        metadata["token_blocks_sha256"]
        == hashlib.sha256(blocks.astype(np.int32).tobytes()).hexdigest()
    )


def test_limits_preserve_source_order(tmp_path):
    path = sources(tmp_path, [{"source_text": "1 2 3 4"}, {"source_text": "5 6"}])
    blocks, _, metadata = source_blocks(path, tokenizer, 2, 10, 1)
    np.testing.assert_array_equal(blocks, [[1, 2]])
    assert metadata["selected"][0]["blocks"] == 1


@pytest.mark.parametrize("text", ["", " ", "1"])
def test_rejects_empty_or_no_complete_blocks(tmp_path, text):
    path = sources(tmp_path, [{"source_text": text}])
    with pytest.raises(ValueError):
        source_blocks(path, tokenizer, 2, 10, 10)


@pytest.mark.parametrize("limits", [(1, 10, 10), (2, 0, 10), (2, 10, 0)])
def test_rejects_invalid_limits(tmp_path, limits):
    with pytest.raises(ValueError):
        source_blocks(tmp_path / "not-read.jsonl", tokenizer, *limits)
