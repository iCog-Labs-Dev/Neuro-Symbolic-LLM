"""Tests for deterministic normalization of model extraction payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from parser.semantic.normalization import (
    SemanticNormalizationError,
    normalize_extraction_payload,
    normalize_symbol,
)


def extraction_payload() -> dict[str, Any]:
    """Return a raw extraction with names and enum values needing cleanup."""
    return {
        "entities": [
            {
                "id": " E-1 ",
                "kind": " Supplier ",
                "entity_type": " Entity ",
                "mention": {"text": "  the Supplier  ", "occurrence": 1},
            }
        ],
        "assertions": [
            {
                "predicate": " Evaluation ",
                "relation": " Deliver Goods ",
                "arguments": [
                    {"entity_id": " E-1 ", "role": " Agent "},
                ],
                "fallback": True,
                "polarity": " Negative ",
                "factuality": " Hypothetical ",
                "modality": " Obligation ",
                "confidence": 0.9,
                "evidence": [{"text": "  the Supplier delivers Goods  "}],
                "alternatives": ["  Another interpretation  "],
            }
        ],
        "rules": [],
    }


def test_normalizes_names_and_enums_but_preserves_source_evidence() -> None:
    normalized = normalize_extraction_payload(extraction_payload())
    entity = normalized["entities"][0]
    assertion = normalized["assertions"][0]

    assert entity["id"] == "e_1"
    assert entity["kind"] == "supplier"
    assert entity["entity_type"] == "Entity"
    assert entity["mention"] == {"text": "  the Supplier  ", "occurrence": 1}
    assert assertion["predicate"] == "Evaluation"
    assert assertion["relation"] == "deliver_goods"
    assert assertion["arguments"] == [{"entity_id": "e_1", "role": "agent"}]
    assert assertion["polarity"] == "negative"
    assert assertion["factuality"] == "hypothetical"
    assert assertion["modality"] == "obligation"
    assert assertion["evidence"] == [{"text": "  the Supplier delivers Goods  "}]
    assert assertion["alternatives"] == ["Another interpretation"]


def test_does_not_mutate_the_model_output() -> None:
    original = extraction_payload()
    before = deepcopy(original)

    normalize_extraction_payload(original)

    assert original == before


def test_applies_only_explicit_aliases_to_kinds_and_relations() -> None:
    payload = extraction_payload()
    normalized = normalize_extraction_payload(
        payload,
        aliases={"Supplier": "Vendor", "Deliver Goods": "Provide"},
    )

    assert normalized["entities"][0]["kind"] == "vendor"
    assert normalized["assertions"][0]["relation"] == "provide"


def test_normalizes_rule_variables_and_nested_literals() -> None:
    payload = {
        "entities": [],
        "assertions": [],
        "rules": [
            {
                "scope": " Specific ",
                "variables": [{"id": "$Some Person", "entity_type": " Entity "}],
                "antecedents": [
                    {
                        "predicate": " Evaluation ",
                        "relation": " Is Red ",
                        "arguments": [{"entity_id": "$Some Person", "role": " Agent "}],
                        "factuality": " Hypothetical ",
                        "confidence": 0.99,
                        "evidence": [{"text": "Some person is red."}],
                    }
                ],
                "consequents": [],
            }
        ],
    }

    normalized = normalize_extraction_payload(payload)
    rule = normalized["rules"][0]

    assert rule["scope"] == "specific"
    assert rule["variables"] == [{"id": "$some_person", "entity_type": "Entity"}]
    assert rule["antecedents"][0]["relation"] == "is_red"
    assert rule["antecedents"][0]["arguments"] == [
        {"entity_id": "$some_person", "role": "agent"}
    ]
    assert rule["antecedents"][0]["evidence"] == [{"text": "Some person is red."}]


def test_normalizes_compatible_unicode() -> None:
    assert normalize_symbol("Ａｐｐｌｅ Phone") == "Apple_Phone"


def test_rejects_whitespace_only_symbols() -> None:
    with pytest.raises(SemanticNormalizationError, match="cannot be empty"):
        normalize_symbol(" \t ")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Des's face", "Des_s_face"),
        ("God’s grace and help", "God_s_grace_and_help"),
        ("peek-and-seek", "peek_and_seek"),
        ("city-wide level", "city_wide_level"),
        ("La América Mexicana", "La_America_Mexicana"),
        ("BABA NAM: KEVALAM", "BABA_NAM_KEVALAM"),
        ("research/design", "research_design"),
        ("cats & dogs", "cats_and_dogs"),
        ("around $50", "around_50"),
        ("time 4:30pm", "time_4_30pm"),
    ],
)
def test_normalizes_punctuation_to_metta_safe_symbols(
    value: str, expected: str
) -> None:
    assert normalize_symbol(value) == expected


def test_preserves_metta_variables_and_normalizes_their_names() -> None:
    assert normalize_symbol("$x") == "$x"
    assert normalize_symbol("$person") == "$person"
    assert normalize_symbol("$Some Person") == "$Some_Person"
