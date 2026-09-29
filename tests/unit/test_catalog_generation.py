"""Tests for the untrusted catalog-guided model-generation step."""

import json

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_generation import _generate_catalog_payload
from parser.semantic.semantic_parser import ModelGenerationError, SemanticParseError


class RecordingBackend:
    provider_name = "test"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return self.response


def test_uses_existing_backend_and_retains_prompt_version() -> None:
    backend = RecordingBackend('```json\n{"entities": [], "assertions": []}\n```')
    source = "  Mira left.  "

    output = _generate_catalog_payload(
        source,
        catalog=load_ontology_catalog(),
        backend=backend,
        model_name="teacher-model",
    )

    assert output.prompt_version == "catalog-document-v1"
    assert output.payload == {"entities": [], "assertions": []}
    assert backend.calls[0][1] == "teacher-model"
    assert f"<source_text>\n{source}\n</source_text>" in backend.calls[0][0]


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        json.dumps({"entities": [], "assertions": [], "source_text": "forged"}),
        json.dumps({"entities": {}, "assertions": []}),
    ],
)
def test_rejects_wrong_envelope_and_model_supplied_metadata(response: str) -> None:
    with pytest.raises(SemanticParseError, match="Catalog output|must be lists"):
        _generate_catalog_payload(
            "Mira left.",
            catalog=load_ontology_catalog(),
            backend=RecordingBackend(response),
            model_name="teacher-model",
        )


def test_rejects_invalid_or_empty_model_output() -> None:
    for response, error_type in (
        ("not JSON", SemanticParseError),
        ("  ", ModelGenerationError),
    ):
        with pytest.raises(error_type):
            _generate_catalog_payload(
                "Mira left.",
                catalog=load_ontology_catalog(),
                backend=RecordingBackend(response),
                model_name="teacher-model",
            )
