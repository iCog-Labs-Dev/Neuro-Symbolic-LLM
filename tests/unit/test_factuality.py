"""Regression tests for factuality, polarity, and modality in current parses."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from parser.semantic import (
    CallableBackend,
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from parser.semantic.schema import (
    SemanticAssertion,
    SemanticParseResult,
    SourceDocument,
)


def _parse_payload(source_text: str, payload: dict[str, Any]) -> SemanticParseResult:
    """Run a fixed structured answer through the current reference parser."""
    parser = ReferenceSemanticParser(
        backend=CallableBackend(lambda **_: json.dumps(payload), provider_name="test"),
        config=SemanticParserConfig(model_name="fixture"),
    )
    return parser.parse_document(
        SourceDocument(
            document_id="factuality-test",
            corpus_id="unit-tests",
            source_text=source_text,
        )
    )


def _claim_payload(
    source_text: str,
    *,
    factuality: str = "asserted",
    polarity: str = "positive",
    modality: str = "none",
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "entities": [
            {
                "id": "drug",
                "entity_type": "Entity",
                "mention": {"text": "drug"},
            }
        ],
        "assertions": [
            {
                "predicate": "Evaluation",
                "relation": "work",
                "arguments": [{"entity_id": "drug", "role": "agent"}],
                "fallback": True,
                "polarity": polarity,
                "factuality": factuality,
                "modality": modality,
                "confidence": confidence,
                "evidence": [{"text": source_text}],
            }
        ],
        "rules": [],
    }


def test_factuality_defaults_to_asserted() -> None:
    assertion = SemanticAssertion.model_validate(
        {
            "predicate": "Evaluation",
            "relation": "announce",
            "arguments": [{"entity_id": "company", "role": "agent"}],
            "fallback": True,
            "confidence": 0.99,
            "evidence": [{"text": "The company announced a product."}],
        }
    )

    assert assertion.factuality == "asserted"


@pytest.mark.parametrize(
    "factuality", ["asserted", "opinion", "speculative", "hypothetical"]
)
def test_schema_accepts_supported_factuality_values(factuality: str) -> None:
    assertion = SemanticAssertion.model_validate(
        {
            "predicate": "Evaluation",
            "relation": "work",
            "arguments": [{"entity_id": "drug", "role": "agent"}],
            "confidence": 0.99,
            "evidence": [{"text": "The drug works."}],
            "factuality": factuality,
        }
    )

    assert assertion.factuality == factuality


@pytest.mark.parametrize(
    "factuality", ["fact", "uncertain", "possible", "conditional", "subjective", ""]
)
def test_schema_rejects_unsupported_factuality_values(factuality: str) -> None:
    with pytest.raises(ValidationError):
        SemanticAssertion.model_validate(
            {
                "predicate": "Evaluation",
                "relation": "work",
                "arguments": [{"entity_id": "drug", "role": "agent"}],
                "confidence": 0.99,
                "evidence": [{"text": "The drug works."}],
                "factuality": factuality,
            }
        )


@pytest.mark.parametrize(
    "factuality", ["asserted", "opinion", "speculative", "hypothetical"]
)
def test_reference_parser_preserves_factuality(factuality: str) -> None:
    source = "The drug probably works."
    result = _parse_payload(source, _claim_payload(source, factuality=factuality))

    assert result.extraction.assertions[0].factuality == factuality


def test_factuality_is_independent_of_confidence_and_polarity() -> None:
    source = "The drug might not work."
    result = _parse_payload(
        source,
        _claim_payload(
            source,
            factuality="speculative",
            polarity="negative",
            confidence=0.99,
        ),
    )

    assertion = result.extraction.assertions[0]

    assert assertion.factuality == "speculative"
    assert assertion.polarity == "negative"
    assert assertion.confidence == 0.99


def test_rule_literals_remain_hypothetical_and_preserve_modality() -> None:
    source = "If the drug works, the patient may recover."
    payload = {
        "entities": [
            {
                "id": "drug",
                "entity_type": "Entity",
                "mention": {"text": "drug"},
            },
            {
                "id": "patient",
                "entity_type": "Entity",
                "mention": {"text": "patient"},
            },
        ],
        "assertions": [],
        "rules": [
            {
                "scope": "specific",
                "antecedents": [
                    {
                        "predicate": "Evaluation",
                        "relation": "work",
                        "arguments": [{"entity_id": "drug", "role": "agent"}],
                        "fallback": True,
                        "factuality": "hypothetical",
                        "confidence": 0.95,
                        "evidence": [{"text": "the drug works"}],
                    }
                ],
                "consequents": [
                    {
                        "predicate": "Evaluation",
                        "relation": "recover",
                        "arguments": [{"entity_id": "patient", "role": "agent"}],
                        "fallback": True,
                        "factuality": "hypothetical",
                        "modality": "permission",
                        "confidence": 0.95,
                        "evidence": [{"text": "the patient may recover"}],
                    }
                ],
                "evidence": [{"text": source}],
            }
        ],
    }

    result = _parse_payload(source, payload)
    rule = result.extraction.rules[0]

    assert rule.antecedents[0].factuality == "hypothetical"
    assert rule.consequents[0].factuality == "hypothetical"
    assert rule.consequents[0].modality == "permission"
