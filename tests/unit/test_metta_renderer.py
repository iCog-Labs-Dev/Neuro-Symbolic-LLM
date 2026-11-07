"""Tests for deterministic structured-semantics-to-MeTTa rendering."""

from __future__ import annotations

import pytest

from parser.semantic.metta_renderer import (
    MettaRenderError,
    render_metta,
    validate_rendered_metta,
)
from parser.semantic.schema import SemanticParseResult


def make_result(
    *,
    predicate: str,
    values: tuple[str, ...],
    roles: tuple[str, ...],
    relation: str | None = None,
    fallback: bool = False,
    polarity: str = "positive",
    factuality: str = "asserted",
    modality: str = "none",
) -> SemanticParseResult:
    """Build a complete parse result whose arguments refer to entity IDs."""
    assert len(values) == len(roles)
    source_text = " ".join((*values, "supporting text"))
    entity_ids = [f"entity_{index}" for index in range(len(values))]
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "renderer-test",
                "corpus_id": "unit-tests",
                "source_text": source_text,
            },
            "provenance": {
                "provider": "test",
                "model": "fixture",
                "parser_version": "test-v1",
                "prompt_version": "test-v1",
                "ontology_versions": ["test-v1"],
            },
            "extraction": {
                "entities": [
                    {
                        "id": entity_id,
                        "entity_type": "Entity",
                        "mention": {"text": value},
                    }
                    for entity_id, value in zip(entity_ids, values, strict=True)
                ],
                "assertions": [
                    {
                        "predicate": predicate,
                        "relation": relation,
                        "arguments": [
                            {"entity_id": entity_id, "role": role}
                            for entity_id, role in zip(entity_ids, roles, strict=True)
                        ],
                        "fallback": fallback,
                        "polarity": polarity,
                        "factuality": factuality,
                        "modality": modality,
                        "confidence": 0.99,
                        "evidence": [{"text": "supporting text"}],
                    }
                ],
                "rules": [],
            },
        }
    )


def test_renders_known_predicate_with_entity_ids() -> None:
    result = make_result(
        predicate="Has",
        values=("dog", "fur"),
        roles=("owner", "possessed"),
    )

    assert render_metta(result) == ["(Has entity_0 entity_1)"]


def test_renders_evaluation_with_argument_list() -> None:
    result = make_result(
        predicate="Evaluation",
        relation="buy",
        values=("Ben", "car"),
        roles=("agent", "patient"),
        fallback=True,
    )

    assert render_metta(result) == ["(Evaluation buy (List entity_0 entity_1))"]


def test_renders_negative_assertion() -> None:
    result = make_result(
        predicate="LocatedIn",
        values=("cat", "chair"),
        roles=("entity", "location"),
        polarity="negative",
    )

    assert render_metta(result) == ["(Not (LocatedIn entity_0 entity_1))"]


def test_preserves_symbol_case_in_entity_ids() -> None:
    result = make_result(
        predicate="LocatedIn",
        values=("Apple", "California"),
        roles=("entity", "location"),
    )

    assert render_metta(result) == ["(LocatedIn entity_0 entity_1)"]


def test_renders_multiple_assertions_in_order() -> None:
    result = make_result(
        predicate="Has",
        values=("dog", "fur"),
        roles=("owner", "possessed"),
    )
    second = make_result(
        predicate="CanDo",
        values=("dog", "fur"),
        roles=("agent", "action"),
    )
    result.extraction.assertions.append(second.extraction.assertions[0])

    assert render_metta(result) == [
        "(Has entity_0 entity_1)",
        "(CanDo entity_0 entity_1)",
    ]


def test_rejects_predicate_that_would_change_expression_structure() -> None:
    result = make_result(
        predicate="Has)",
        values=("company", "car"),
        roles=("owner", "possessed"),
    )

    with pytest.raises(MettaRenderError, match="predicate"):
        render_metta(result)


def test_validates_and_parses_rendered_expressions() -> None:
    atoms = validate_rendered_metta(
        ["(Has entity_0 entity_1)", "(Not (LocatedIn entity_0 entity_1))"]
    )

    assert [str(atom) for atom in atoms] == [
        "(Has entity_0 entity_1)",
        "(Not (LocatedIn entity_0 entity_1))",
    ]


def test_rejects_invalid_rendered_expression() -> None:
    with pytest.raises(MettaRenderError, match="Rendered invalid MeTTa"):
        validate_rendered_metta(["(Unknown entity_0 entity_1)"])


def test_renders_negative_evaluation() -> None:
    result = make_result(
        predicate="Evaluation",
        relation="open",
        values=("Alice", "door"),
        roles=("agent", "patient"),
        fallback=True,
        polarity="negative",
    )

    assert render_metta(result) == ["(Not (Evaluation open (List entity_0 entity_1)))"]


def test_renders_zero_argument_evaluation() -> None:
    result = make_result(
        predicate="Evaluation",
        relation="rain",
        values=(),
        roles=(),
        fallback=True,
    )

    assert render_metta(result) == ["(Evaluation rain (List))"]


@pytest.mark.parametrize(
    ("modality", "polarity", "expected"),
    [
        ("obligation", "positive", "(Obligation (Evaluation deliver (List entity_0)))"),
        (
            "obligation",
            "negative",
            "(Obligation (Not (Evaluation deliver (List entity_0))))",
        ),
        ("permission", "positive", "(Permission (Evaluation deliver (List entity_0)))"),
        (
            "permission",
            "negative",
            "(Not (Permission (Evaluation deliver (List entity_0))))",
        ),
    ],
)
def test_renders_assertion_modality(
    modality: str, polarity: str, expected: str
) -> None:
    result = make_result(
        predicate="Evaluation",
        relation="deliver",
        values=("supplier",),
        roles=("agent",),
        fallback=True,
        polarity=polarity,
        modality=modality,
    )

    assert render_metta(result) == [expected]


def _rule_result(antecedent_count: int) -> SemanticParseResult:
    properties = ("red", "rough")[:antecedent_count]
    source_text = "Mira " + " and ".join(properties) + " then nice."
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "rule-test",
                "corpus_id": "unit-tests",
                "source_text": source_text,
            },
            "provenance": {
                "provider": "test",
                "model": "fixture",
                "parser_version": "test-v1",
                "prompt_version": "test-v1",
                "ontology_versions": ["test-v1"],
            },
            "extraction": {
                "entities": [
                    {
                        "id": "mira",
                        "entity_type": "Entity",
                        "mention": {"text": "Mira"},
                    },
                    *[
                        {
                            "id": f"property_{index}",
                            "entity_type": "Entity",
                            "mention": {"text": prop},
                        }
                        for index, prop in enumerate((*properties, "nice"))
                    ],
                ],
                "assertions": [],
                "rules": [
                    {
                        "scope": "specific",
                        "antecedents": [
                            {
                                "predicate": "PropertyOf",
                                "arguments": [
                                    {"entity_id": "mira", "role": "entity"},
                                    {
                                        "entity_id": f"property_{index}",
                                        "role": "property",
                                    },
                                ],
                                "factuality": "hypothetical",
                                "confidence": 0.99,
                                "evidence": [{"text": prop}],
                            }
                            for index, prop in enumerate(properties)
                        ],
                        "consequents": [
                            {
                                "predicate": "PropertyOf",
                                "arguments": [
                                    {"entity_id": "mira", "role": "entity"},
                                    {
                                        "entity_id": f"property_{len(properties)}",
                                        "role": "property",
                                    },
                                ],
                                "factuality": "hypothetical",
                                "confidence": 0.99,
                                "evidence": [{"text": "nice"}],
                            }
                        ],
                        "evidence": [{"text": "then nice."}],
                    }
                ],
            },
        }
    )


def test_renders_single_antecedent_rule() -> None:
    assert render_metta(_rule_result(1)) == [
        "(Implies (PropertyOf mira property_0) (PropertyOf mira property_1))"
    ]


def test_renders_multiple_antecedents_with_and() -> None:
    assert render_metta(_rule_result(2)) == [
        "(Implies (And (PropertyOf mira property_0) "
        "(PropertyOf mira property_1)) (PropertyOf mira property_2))"
    ]


def test_refuses_non_asserted_standalone_claims() -> None:
    result = make_result(
        predicate="Evaluation",
        relation="leave",
        values=("Mira",),
        roles=("agent",),
        fallback=True,
        factuality="speculative",
    )

    with pytest.raises(MettaRenderError, match="is speculative"):
        render_metta(result)
    assert render_metta(result, asserted_only=True) == []
