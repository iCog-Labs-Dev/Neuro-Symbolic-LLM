"""Tests for source-grounded offline template embeddings."""

import json

import numpy as np
import pytest

from offline.embedding.context_builder import (
    load_pattern_contexts,
    load_semantic_documents,
    prepare_contexts,
)
from offline.embedding.template_embeddings import (
    create_random_projection,
    initialize_template_kv,
)


@pytest.mark.parametrize("contract", ["semantic-document-v1", "semantic-document-v2"])
def test_projection_context_reader_preserves_versioned_source_text(tmp_path, contract):
    path = tmp_path / "documents.jsonl"
    text = "Nora did not repair the radio."
    path.write_text(
        json.dumps(
            {
                "contract_version": contract,
                "corpus_id": "corpus",
                "document_id": "doc",
                "source_text": text,
            }
        )
        + "\n"
    )
    assert load_semantic_documents(path) == {("corpus", "doc"): text}


def test_projection_retrieval_reports_boundary_ties():
    from offline.embedding.projection_training import retrieval_report

    scores = np.array([[0.8, 0.8, 0.1], [0.9, 0.5, 0.1]])
    targets = np.array([[False, True, False], [True, False, False]])
    result = retrieval_report(scores, targets, top_k=1)
    assert result["micro_precision"] == 0.5
    assert result["tie_precision_lower"] == 0.5
    assert result["tie_precision_upper"] == 1.0
    with pytest.raises(ValueError):
        retrieval_report(scores, np.zeros_like(targets), top_k=1)


def test_learned_projection_preserves_scale_and_frozen_features():
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from offline.embedding.projection_training import fit_projection

    templates = np.eye(4, dtype=np.float32)
    queries = np.repeat(templates, 2, axis=0)
    positives = np.repeat(np.eye(4, dtype=bool), 2, axis=0)
    original = templates.copy(), queries.copy()
    initial = create_random_projection(4, 2, 42)
    projection, report = fit_projection(
        initial,
        queries,
        templates,
        positives,
        np.ones_like(positives),
        templates,
        np.eye(4, dtype=bool),
        epochs=8,
        batch_size=3,
        learning_rate=0.03,
    )
    np.testing.assert_allclose(projection @ projection.T, np.eye(2), atol=2e-5)
    np.testing.assert_array_equal(templates, original[0])
    np.testing.assert_array_equal(queries, original[1])
    assert report["best_epoch"] > 0
    assert (
        min(row["validation_loss"] for row in report["history"])
        < report["history"][0]["validation_loss"]
    )
    assert not np.array_equal(projection, initial)


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
