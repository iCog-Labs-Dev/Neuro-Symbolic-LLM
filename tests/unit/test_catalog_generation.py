"""Tests for the untrusted catalog-guided model-generation step."""

import json

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.schema import SourceDocument
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)
from tests.semantic_fixtures import target


class RecordingBackend:
    provider_name = "test"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return self.response


def _parser(
    response: str | RecordingBackend, *, catalog=None
) -> ReferenceSemanticParser:
    backend = (
        response
        if isinstance(response, RecordingBackend)
        else RecordingBackend(response)
    )
    return ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        catalog=catalog,
    )


def test_parser_exposes_current_prompt_without_generation() -> None:
    parser = _parser('{"entities": [], "assertions": [], "rules": []}')
    prompt = ReferenceSemanticParser.build_prompt("Alice opened the door.")

    assert parser.prompt_version.startswith("catalog-document-v15+sha256.")
    assert "<worked_example>" in prompt
    assert "Shared source interpretation rules:" in prompt


def test_instance_prompt_and_generation_use_the_selected_catalog() -> None:
    catalog = load_ontology_catalog(modules=("upper",))
    source = "Mira left."
    backend = RecordingBackend(json.dumps(target(source)))
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="review"),
        catalog=catalog,
    )

    inspected = parser.build_document_prompt(source)
    result = parser.parse(source)

    assert inspected.text == backend.calls[0][0]
    assert "Ontology versions: upper-v1\n" in inspected.text
    assert result.provenance.prompt_version == inspected.version
    assert result.document.source_text == source


def test_reference_context_is_not_added_to_the_extraction_source() -> None:
    source = "She left."
    backend = RecordingBackend(json.dumps(target(source)))
    parser = _parser(backend)

    result = parser.parse(source, context="Nora arrived yesterday.")

    assert result.document.source_text == source
    assert (
        "<reference_context>\nNora arrived yesterday.\n</reference_context>"
        in backend.calls[0][0]
    )
    quote = result.extraction.assertions[0].evidence[0]
    assert result.locate(quote) == (0, len(source))

    backend.response = json.dumps(target("Nora arrived yesterday."))
    with pytest.raises(SemanticParseError, match="not in the source text"):
        parser.parse(source, context="Nora arrived yesterday.")


def test_generation_returns_extraction_and_prompt_version() -> None:
    source = "  Mira left.  "
    payload = target(source)
    backend = RecordingBackend(f"```json\n{json.dumps(payload)}\n```")
    catalog = load_ontology_catalog()
    parser = _parser(backend, catalog=catalog)
    document = SourceDocument(
        document_id="generation-test",
        corpus_id="unit-tests",
        source_text=source,
    )

    generated, prompt_version = parser.generate_extraction_payload(
        document, catalog=catalog
    )

    assert generated == payload
    assert (
        prompt_version == build_catalog_document_prompt(source, catalog=catalog).version
    )
    assert backend.calls[0][1] == "teacher-model"
    assert f"<source_text>\n{source}\n</source_text>" in backend.calls[0][0]


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        json.dumps(
            {"entities": [], "assertions": [], "rules": [], "source_text": "forged"}
        ),
        json.dumps({"entities": {}, "assertions": [], "rules": []}),
    ],
)
def test_parse_rejects_wrong_envelopes_and_model_supplied_metadata(
    response: str,
) -> None:
    with pytest.raises(SemanticParseError):
        _parser(response).parse("Mira left.")


def test_rejects_invalid_or_empty_model_output() -> None:
    for response in ("not JSON", "  "):
        with pytest.raises(SemanticParseError):
            _parser(response).parse("Mira left.")
