"""Cross-repository tests for catalog-guided semantic-document validation."""

import json
from typing import Any

import pytest

pytest.importorskip("hybrid_miner.contracts.semantic_document")

from hybrid_miner.contracts.semantic_document import SemanticDocument  # noqa: E402

from parser.ontology.catalog import load_ontology_catalog  # noqa: E402
from parser.semantic.catalog_document_parser import (  # noqa: E402
    CATALOG_PARSER_VERSION,
    CatalogSemanticDocumentParser,
)
from parser.semantic.semantic_parser import SemanticParseError  # noqa: E402


class RecordingBackend:
    provider_name = "test-provider"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return json.dumps(self.payload)


def _source_and_payload() -> tuple[str, dict[str, Any]]:
    source = "🙂 Although Mira promised Sol, she delayed."
    clause_end = source.index(",")
    second_start = source.index("she")
    return source, {
        "entities": [
            {
                "id": "clause-a",
                "type": "Clause",
                "span": {"start": 2, "end": clause_end},
            },
            {
                "id": "clause-b",
                "type": "Clause",
                "span": {"start": second_start, "end": len(source)},
            },
        ],
        "assertions": [
            {
                "predicate": "Concession",
                "arguments": ["clause-a", "clause-b"],
                "argument_types": ["Clause", "Clause"],
                "status": "extracted-hypothesis",
                "confidence": 0.9,
                "source_spans": [{"start": 2, "end": len(source)}],
                "alternatives": [],
            }
        ],
    }


def _parse(payload: dict[str, Any], source: str) -> SemanticDocument:
    parser = CatalogSemanticDocumentParser(
        backend=RecordingBackend(payload), model_name="teacher-model"
    )
    return parser.parse_document(
        document_id="doc-1", corpus_id="corpus-1", source_text=source
    )


def test_returns_validated_document_with_runtime_provenance() -> None:
    source, payload = _source_and_payload()
    backend = RecordingBackend(payload)
    parser = CatalogSemanticDocumentParser(
        backend=backend,
        model_name="teacher-model",
        model_revision="revision-1",
    )

    document = parser.parse_document(
        document_id="doc-1", corpus_id="corpus-1", source_text=source
    )

    assert isinstance(document, SemanticDocument)
    assert document.source_text == source
    assert document.parser.parser_version == CATALOG_PARSER_VERSION
    assert document.parser.prompt_version == "catalog-document-v1"
    assert document.parser.ontology_versions == load_ontology_catalog().versions
    assert document.parser.provider == "test-provider"
    assert document.parser.model_revision == "revision-1"
    assert document.assertions[0].source_spans[0].end == len(source)
    assert f"<source_text>\n{source}\n</source_text>" in backend.calls[0][0]
    assert (
        document.canonical_json()
        == SemanticDocument.model_validate_json(
            document.canonical_json()
        ).canonical_json()
    )


def test_rejects_unknown_predicate_without_provisional_status() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["predicate"] = "InventedRelation"

    with pytest.raises(SemanticParseError, match="unknown predicate"):
        _parse(payload, source)


def test_accepts_explicit_provisional_relation() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["predicate"] = "ProvisionalRelation"
    payload["assertions"][0]["provisional_relation_id"] = "unclassified_contrast"

    document = _parse(payload, source)

    assert document.assertions[0].provisional_relation_id == "unclassified_contrast"


def test_rejects_confidence_outside_catalog_range() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["confidence"] = 0.3

    with pytest.raises(SemanticParseError, match="outside catalog range"):
        _parse(payload, source)


def test_rejects_undeclared_entity_type() -> None:
    source, payload = _source_and_payload()
    payload["entities"][0]["type"] = "Artifact"
    payload["assertions"] = []

    with pytest.raises(SemanticParseError, match="Undeclared ontology entity type"):
        _parse(payload, source)


def test_rejects_out_of_bounds_evidence() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["source_spans"][0]["end"] = len(source) + 1

    with pytest.raises(SemanticParseError, match="source span exceeds source text"):
        _parse(payload, source)
