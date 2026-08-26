"""Tests for the structured semantic-parser data contract."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from parser.semantic import ReferenceSemanticParser, SemanticParseError
from parser.semantic.schema import SemanticParseResult


def valid_result_data() -> dict[str, Any]:
    """Return a minimal valid structured parser response."""
    return {
        "assertions": [
            {
                "predicate": "Evaluation",
                "relation": "buy",
                "arguments": [
                    {"value": "Ben", "role": "agent", "type": "Person"},
                    {"value": "car", "role": "patient", "type": None},
                ],
                "fallback": True,
                "polarity": "positive",
                "confidence": 0.98,
                "source_span": "Ben bought a car.",
                "alternatives": [],
            }
        ]
    }


def test_accepts_the_structured_output_contract() -> None:
    result = SemanticParseResult.model_validate(valid_result_data())

    assertion = result.assertions[0]
    assert assertion.predicate == "Evaluation"
    assert assertion.arguments[0].role == "agent"
    assert assertion.confidence == 0.98


def test_applies_safe_optional_defaults() -> None:
    data = valid_result_data()
    assertion = data["assertions"][0]
    del assertion["fallback"]
    del assertion["polarity"]
    del assertion["alternatives"]

    result = SemanticParseResult.model_validate(data)

    assert result.assertions[0].fallback is False
    assert result.assertions[0].polarity == "positive"
    assert result.assertions[0].alternatives == []


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    data = valid_result_data()
    data["assertions"][0]["confidence"] = confidence

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


def test_rejects_unknown_polarity() -> None:
    data = valid_result_data()
    data["assertions"][0]["polarity"] = "uncertain"

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("predicate", ""),
        ("arguments", []),
        ("source_span", ""),
    ],
)
def test_rejects_empty_required_assertion_values(
    field: str,
    value: object,
) -> None:
    data = valid_result_data()
    data["assertions"][0][field] = value

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


def test_rejects_empty_assertion_collection() -> None:
    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate({"assertions": []})


@pytest.mark.parametrize("level", ["result", "assertion", "argument"])
def test_rejects_extra_fields_at_every_level(level: str) -> None:
    data = valid_result_data()
    if level == "result":
        data["unexpected"] = "value"
    elif level == "assertion":
        data["assertions"][0]["unexpected"] = "value"
    else:
        data["assertions"][0]["arguments"][0]["unexpected"] = "value"

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


def known_predicate_result(
    *,
    predicate: str = "Has",
    relation: str | None = None,
    roles: tuple[str, ...] = ("owner", "possessed"),
    fallback: bool = False,
) -> SemanticParseResult:
    """Return structured output for predicate-specific validation tests."""
    data = valid_result_data()
    assertion = data["assertions"][0]
    assertion["predicate"] = predicate
    assertion["relation"] = relation
    assertion["arguments"] = [
        {"value": f"value_{index}", "role": role} for index, role in enumerate(roles)
    ]
    assertion["fallback"] = fallback
    return SemanticParseResult.model_validate(data)


def test_accepts_configured_predicate_arity_and_roles() -> None:
    result = known_predicate_result()

    assert ReferenceSemanticParser.validate_predicates(result) is result
    assert ReferenceSemanticParser.validate_arguments(result) is result


def test_rejects_unknown_predicate() -> None:
    result = known_predicate_result(predicate="InventedPredicate")

    with pytest.raises(SemanticParseError, match="Unknown semantic predicate"):
        ReferenceSemanticParser.validate_predicates(result)


@pytest.mark.parametrize(
    ("relation", "fallback", "message"),
    [
        ("own", False, "relation=null"),
        (None, True, "fallback must be false"),
    ],
)
def test_rejects_invalid_known_predicate_contract(
    relation: str | None,
    fallback: bool,
    message: str,
) -> None:
    result = known_predicate_result(relation=relation, fallback=fallback)

    with pytest.raises(SemanticParseError, match=message):
        ReferenceSemanticParser.validate_predicates(result)


def test_rejects_wrong_predicate_arity() -> None:
    result = known_predicate_result(roles=("owner",))

    with pytest.raises(SemanticParseError, match="requires 2 arguments"):
        ReferenceSemanticParser.validate_arguments(result)


def test_rejects_wrong_predicate_role_order() -> None:
    result = known_predicate_result(roles=("possessed", "owner"))

    with pytest.raises(SemanticParseError, match="requires argument roles"):
        ReferenceSemanticParser.validate_arguments(result)


def test_accepts_evaluation_fallback_contract() -> None:
    result = known_predicate_result(
        predicate="Evaluation",
        relation="buy",
        roles=("agent", "patient"),
        fallback=True,
    )

    assert ReferenceSemanticParser.validate_predicates(result) is result
    assert ReferenceSemanticParser.validate_arguments(result) is result


def test_rejects_evaluation_without_relation() -> None:
    result = known_predicate_result(
        predicate="Evaluation",
        relation=None,
        roles=("agent",),
        fallback=True,
    )

    with pytest.raises(SemanticParseError, match="requires a relation"):
        ReferenceSemanticParser.validate_predicates(result)
