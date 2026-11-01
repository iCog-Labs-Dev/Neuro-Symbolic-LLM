"""Unit tests for the versioned semantic parser output contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticParseResult,
)


def valid_result_data() -> dict[str, Any]:
    """Return a minimal complete parse record with evidence in its source."""
    return {
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "document": {
            "document_id": "fixture-1",
            "corpus_id": "unit-tests",
            "source_text": "Mira left.",
        },
        "provenance": {
            "provider": "test",
            "model": "fixture-model",
            "parser_version": "1.0.0",
            "prompt_version": "test-v1",
            "ontology_versions": ["test-ontology-v1"],
        },
        "extraction": {
            "entities": [
                {
                    "id": "mira",
                    "entity_type": "Person",
                    "mention": {"text": "Mira"},
                }
            ],
            "assertions": [
                {
                    "predicate": "Evaluation",
                    "relation": "leave",
                    "arguments": [{"entity_id": "mira", "role": "agent"}],
                    "fallback": True,
                    "confidence": 0.98,
                    "evidence": [{"text": "Mira left."}],
                }
            ],
            "rules": [],
        },
    }


def test_accepts_the_current_structured_parse_contract() -> None:
    result = SemanticParseResult.model_validate(valid_result_data())

    assert result.contract_version == SEMANTIC_CONTRACT_VERSION
    assert result.document.document_id == "fixture-1"
    assert result.extraction.assertions[0].predicate == "Evaluation"
    assert result.extraction.assertions[0].arguments[0].entity_id == "mira"


def test_applies_optional_field_defaults() -> None:
    data = valid_result_data()
    assertion = data["extraction"]["assertions"][0]
    for field in ("fallback", "polarity", "factuality", "modality", "alternatives"):
        assertion.pop(field, None)

    result = SemanticParseResult.model_validate(data)
    parsed = result.extraction.assertions[0]

    assert parsed.fallback is False
    assert parsed.polarity == "positive"
    assert parsed.factuality == "asserted"
    assert parsed.modality == "none"
    assert parsed.alternatives == []
    assert result.provenance.repairs == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("polarity", "uncertain"), ("factuality", "possible"), ("modality", "required")],
)
def test_rejects_unsupported_assertion_classification(field: str, value: str) -> None:
    data = valid_result_data()
    data["extraction"]["assertions"][0][field] = value

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    data = valid_result_data()
    data["extraction"]["assertions"][0]["confidence"] = confidence

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


@pytest.mark.parametrize("level", ["result", "document", "assertion", "argument"])
def test_forbids_unrecognized_fields(level: str) -> None:
    data = valid_result_data()
    targets = {
        "result": data,
        "document": data["document"],
        "assertion": data["extraction"]["assertions"][0],
        "argument": data["extraction"]["assertions"][0]["arguments"][0],
    }
    targets[level]["unexpected"] = "value"

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


def test_requires_at_least_one_assertion_or_rule() -> None:
    data = valid_result_data()
    data["extraction"]["assertions"] = []

    with pytest.raises(ValidationError, match="at least one assertion or rule"):
        SemanticParseResult.model_validate(data)


def test_rejects_duplicate_entities_and_unresolved_arguments() -> None:
    duplicate = valid_result_data()
    duplicate["extraction"]["entities"].append(
        deepcopy(duplicate["extraction"]["entities"][0])
    )
    with pytest.raises(ValidationError, match="Entity IDs must be unique"):
        SemanticParseResult.model_validate(duplicate)

    unresolved = valid_result_data()
    unresolved["extraction"]["assertions"][0]["arguments"][0]["entity_id"] = "missing"
    with pytest.raises(ValidationError, match="unknown entity or rule variable"):
        SemanticParseResult.model_validate(unresolved)


def test_locates_repeated_source_quotes_by_occurrence() -> None:
    data = valid_result_data()
    data["document"]["source_text"] = "Mira saw Mira."
    data["extraction"]["assertions"][0]["evidence"] = [
        {"text": "Mira", "occurrence": 1}
    ]

    result = SemanticParseResult.model_validate(data)

    assert result.locate(result.extraction.assertions[0].evidence[0]) == (9, 13)


def test_rejects_source_quotes_that_do_not_resolve() -> None:
    data = valid_result_data()
    data["extraction"]["assertions"][0]["evidence"] = [
        {"text": "Mira", "occurrence": 1}
    ]

    with pytest.raises(ValidationError, match="occurrence 1 is not in the source text"):
        SemanticParseResult.model_validate(data)


def test_accepts_a_specific_rule_with_hypothetical_literals() -> None:
    data = valid_result_data()
    data["extraction"]["assertions"] = []
    literal = deepcopy(valid_result_data()["extraction"]["assertions"][0])
    literal["factuality"] = "hypothetical"
    data["extraction"]["rules"] = [
        {
            "scope": "specific",
            "antecedents": [literal],
            "consequents": [deepcopy(literal)],
            "evidence": [{"text": "Mira left."}],
        }
    ]

    result = SemanticParseResult.model_validate(data)

    assert result.extraction.rules[0].scope == "specific"
    assert result.extraction.rules[0].antecedents[0].factuality == "hypothetical"


def test_rejects_rule_with_non_hypothetical_literal() -> None:
    data = valid_result_data()
    data["extraction"]["assertions"] = []
    literal = deepcopy(valid_result_data()["extraction"]["assertions"][0])
    data["extraction"]["rules"] = [
        {
            "scope": "specific",
            "antecedents": [literal],
            "consequents": [deepcopy(literal)],
            "evidence": [{"text": "Mira left."}],
        }
    ]

    with pytest.raises(ValidationError, match="must have hypothetical factuality"):
        SemanticParseResult.model_validate(data)
