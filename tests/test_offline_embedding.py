"""Tests for source-grounded offline template embeddings."""

import json

import numpy as np
import pytest

from offline.embedding.context_builder import (
    load_pattern_contexts,
    prepare_contexts,
)
from offline.embedding.template_embeddings import (
    create_random_projection,
    initialize_template_kv,
)


class DeterministicEncoder:
    """Small encoder for testing averaging without downloading GPT-2."""

    model_name = "deterministic-test-encoder"
    pooling = "attention_masked_mean"
    layer_idx = 11

    def encode(self, texts):
        return np.asarray(
            [
                [
                    float(len(text)),
                    float(sum(ord(char) for char in text) % 1000),
                    1.0,
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def make_files(tmp_path, examples):
    """Create minimal records accepted by the embedding context loader."""
    documents = []
    patterns = []

    for pattern_id, document_id, source_text, evidence in examples:
        documents.append(
            {
                "contract_version": "semantic-document-v1",
                "corpus_id": "test-corpus",
                "document_id": document_id,
                "source_text": source_text,
            }
        )

        spans = []
        for text in evidence:
            start = source_text.index(text)
            spans.append(
                {
                    "document_id": document_id,
                    "start": start,
                    "end": start + len(text),
                }
            )

        patterns.append(
            {
                "contract_version": "pattern-record-v1",
                "id": pattern_id,
                "provenance": {
                    "corpus_id": "test-corpus",
                    "source_document_ids": [document_id],
                },
                "source_spans": spans,
            }
        )

    documents_path = tmp_path / "documents.jsonl"
    patterns_path = tmp_path / "patterns.jsonl"

    write_jsonl(documents_path, documents)
    write_jsonl(patterns_path, patterns)

    return load_pattern_contexts(patterns_path, documents_path)


def test_concession_complete_source_span(tmp_path):
    templates = make_files(
        tmp_path,
        [
            (
                "concession",
                "doc1",
                "Although it snowed, Nora sang.",
                ["Although it snowed, Nora sang"],
            )
        ],
    )

    assert templates[0].contexts == ("Although it snowed, Nora sang",)


def test_conditional_complete_source_span(tmp_path):
    templates = make_files(
        tmp_path,
        [
            (
                "conditional",
                "doc1",
                "If it rains, Nora will leave.",
                ["If it rains, Nora will leave"],
            )
        ],
    )

    assert templates[0].contexts == ("If it rains, Nora will leave",)


def test_multiple_sentences_remain_one_context(tmp_path):
    source = "Although it snowed, Nora sang. " "She continued singing despite the cold."

    templates = make_files(
        tmp_path,
        [
            (
                "multi-sentence",
                "doc1",
                source,
                [source],
            )
        ],
    )

    assert templates[0].contexts == (source,)


def test_separate_spans_remain_separate_contexts(tmp_path):
    templates = make_files(
        tmp_path,
        [
            (
                "conditional-split",
                "doc1",
                "If it rains, Nora will leave.",
                ["it rains", "Nora will leave"],
            )
        ],
    )

    assert templates[0].contexts == (
        "it rains",
        "Nora will leave",
    )


def test_average_embeddings_and_initialize_kv():
    template = prepare_contexts(
        "pattern1",
        ["it rains", "Nora will leave"],
    )

    encoder = DeterministicEncoder()

    result = initialize_template_kv(
        template,
        encoder,
        symbolic_dim=2,
        projection_seed=42,
    )

    original_embeddings = encoder.encode(template.contexts)
    average = original_embeddings.mean(axis=0)

    projection = create_random_projection(
        hidden_dim=3,
        symbolic_dim=2,
        seed=42,
    )

    expected = projection @ average

    np.testing.assert_allclose(result["key"], expected)
    np.testing.assert_allclose(result["value"], expected)

    assert result["key"] == result["value"]
    assert result["num_contexts"] == 2
    assert result["embedding_dim"] == 2
    assert result["encoder_hidden_dim"] == 3
    assert result["projection_shape"] == [2, 3]
    assert result["projection_seed"] == 42


def test_duplicate_source_span_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate source span"):
        make_files(
            tmp_path,
            [
                (
                    "duplicate",
                    "doc1",
                    "Although it snowed, Nora sang.",
                    ["it snowed", "it snowed"],
                )
            ],
        )


def test_invalid_source_span_is_rejected(tmp_path):
    documents_path = tmp_path / "documents.jsonl"
    patterns_path = tmp_path / "patterns.jsonl"

    write_jsonl(
        documents_path,
        [
            {
                "contract_version": "semantic-document-v1",
                "corpus_id": "test-corpus",
                "document_id": "doc1",
                "source_text": "Hello world",
            }
        ],
    )

    write_jsonl(
        patterns_path,
        [
            {
                "contract_version": "pattern-record-v1",
                "id": "invalid",
                "provenance": {
                    "corpus_id": "test-corpus",
                    "source_document_ids": ["doc1"],
                },
                "source_spans": [
                    {
                        "document_id": "doc1",
                        "start": 0,
                        "end": 100,
                    }
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="exceeds"):
        load_pattern_contexts(patterns_path, documents_path)
