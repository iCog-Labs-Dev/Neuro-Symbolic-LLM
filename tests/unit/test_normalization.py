"""Tests for deterministic normalization of model extraction payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from parser.semantic.normalization import (
    SemanticNormalizationError,
    normalize_extraction_payload,
    normalize_symbol,
    restore_quote_case,
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


def test_restore_quote_case_fixes_only_letter_case() -> None:
    source = "Students may not use phones. The students left."
    payload: dict[str, Any] = {
        "entities": [
            {"id": "students", "mention": {"text": "students"}},
            {"id": "phones", "mention": {"text": "phones"}},
        ],
        "assertions": [
            {"evidence": [{"text": "students may not use phones"}]},
            {"evidence": [{"text": "pupils"}]},
        ],
        "rules": [{"evidence": [{"text": "THE STUDENTS LEFT"}], "antecedents": []}],
    }

    repaired, changes = restore_quote_case(payload, source)

    # found exactly as written: untouched, even though "Students" also exists
    assert repaired["entities"][0] == payload["entities"][0]
    assert repaired["entities"][1] == payload["entities"][1]
    # case-only difference: the source's own words and their occurrence number
    assert repaired["assertions"][0]["evidence"] == [
        {"text": "Students may not use phones", "occurrence": 0}
    ]
    assert repaired["rules"][0]["evidence"] == [
        {"text": "The students left", "occurrence": 0}
    ]
    # different words stay wrong for validation to reject
    assert repaired["assertions"][1] == payload["assertions"][1]
    assert len(changes) == 2
    assert payload["assertions"][0]["evidence"] == [
        {"text": "students may not use phones"}
    ]


def test_restore_quote_case_recomputes_the_occurrence() -> None:
    source = "the Buyer met the buyer and the Buyer."
    payload = {"entities": [{"mention": {"text": "BUYER", "occurrence": 2}}]}

    repaired, changes = restore_quote_case(payload, source)

    # third "buyer" ignoring case is the second exact "Buyer"
    assert repaired["entities"][0]["mention"] == {"text": "Buyer", "occurrence": 1}
    assert changes == ["quote 'BUYER' restored to source spelling 'Buyer'"]
    assert restore_quote_case(
        {"entities": [{"mention": {"text": "buyer", "occurrence": 5}}]}, source
    ) == ({"entities": [{"mention": {"text": "buyer", "occurrence": 5}}]}, [])
