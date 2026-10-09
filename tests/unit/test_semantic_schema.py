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
                "factuality": "asserted",
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
    assert assertion.factuality == "asserted"
    assert assertion.confidence == 0.98


def test_applies_safe_optional_defaults() -> None:
    data = valid_result_data()
    assertion = data["assertions"][0]

    del assertion["fallback"]
    del assertion["polarity"]
    del assertion["factuality"]
    del assertion["alternatives"]

    result = SemanticParseResult.model_validate(data)

    assertion = result.assertions[0]

    assert assertion.fallback is False
    assert assertion.polarity == "positive"
    assert assertion.factuality == "asserted"
    assert assertion.alternatives == []


@pytest.mark.parametrize(
    "factuality",
    [
        "asserted",
        "opinion",
        "speculative",
        "hypothetical",
    ],
)
def test_accepts_supported_factuality_values(
    factuality: str,
) -> None:
    data = valid_result_data()
    data["assertions"][0]["factuality"] = factuality

    result = SemanticParseResult.model_validate(data)

    assert result.assertions[0].factuality == factuality


@pytest.mark.parametrize(
    "factuality",
    [
        "fact",
        "uncertain",
        "possible",
        "conditional",
        "",
    ],
)
def test_rejects_unknown_factuality(
    factuality: str,
) -> None:
    data = valid_result_data()
    data["assertions"][0]["factuality"] = factuality

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


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


def test_allows_empty_arguments_for_evaluation() -> None:
    data = valid_result_data()
    assertion = data["assertions"][0]

    assertion["predicate"] = "Evaluation"
    assertion["relation"] = "rain"
    assertion["arguments"] = []
    assertion["fallback"] = True

    result = SemanticParseResult.model_validate(data)

    assert result.assertions[0].arguments == []


def test_rejects_empty_arguments_for_non_evaluation() -> None:
    data = valid_result_data()
    assertion = data["assertions"][0]

    assertion["predicate"] = "Has"
    assertion["relation"] = None
    assertion["arguments"] = []
    assertion["fallback"] = False

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
    """Return structured output for predicate validation tests."""
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


def test_repairs_known_predicate_contract() -> None:
    result = known_predicate_result(
        predicate="UsedFor",
        relation="cut",
        roles=("entity", "purpose"),
        fallback=True,
    )

    repaired = ReferenceSemanticParser.repair_known_predicate_contract(result)

    assertion = repaired.assertions[0]

    assert assertion.predicate == "UsedFor"
    assert assertion.relation is None
    assert assertion.fallback is False

    # Original model output is not mutated.
    assert result.assertions[0].relation == "cut"
    assert result.assertions[0].fallback is True

    assert ReferenceSemanticParser.validate_predicates(repaired) is repaired
    assert ReferenceSemanticParser.validate_arguments(repaired) is repaired


def test_repairs_contract_for_any_known_predicate() -> None:
    result = known_predicate_result(
        predicate="Contains",
        relation="contain",
        roles=("container", "contained"),
        fallback=True,
    )

    repaired = ReferenceSemanticParser.repair_known_predicate_contract(result)

    assertion = repaired.assertions[0]

    assert assertion.relation is None
    assert assertion.fallback is False

    assert ReferenceSemanticParser.validate_predicates(repaired) is repaired
    assert ReferenceSemanticParser.validate_arguments(repaired) is repaired


def test_negative_speculative_assertion_preserves_both_dimensions() -> None:
    data = valid_result_data()

    assertion = data["assertions"][0]
    assertion["polarity"] = "negative"
    assertion["factuality"] = "speculative"

    result = SemanticParseResult.model_validate(data)

    assertion = result.assertions[0]

    assert assertion.polarity == "negative"
    assert assertion.factuality == "speculative"


def test_negative_hypothetical_assertion_preserves_both_dimensions() -> None:
    data = valid_result_data()

    assertion = data["assertions"][0]
    assertion["polarity"] = "negative"
    assertion["factuality"] = "hypothetical"

    result = SemanticParseResult.model_validate(data)

    assertion = result.assertions[0]

    assert assertion.polarity == "negative"
    assert assertion.factuality == "hypothetical"


def test_accepts_rule_only_semantic_result() -> None:
    result = SemanticParseResult.model_validate(
        {
            "assertions": [],
            "rules": [
                {
                    "antecedents": [
                        {
                            "predicate": "PropertyOf",
                            "relation": None,
                            "arguments": [
                                {
                                    "value": "someone",
                                    "role": "entity",
                                    "type": None,
                                },
                                {
                                    "value": "red",
                                    "role": "property",
                                    "type": None,
                                },
                            ],
                            "fallback": False,
                            "polarity": "positive",
                            "factuality": "hypothetical",
                            "confidence": 0.99,
                            "source_span": "someone is red",
                            "alternatives": [],
                        }
                    ],
                    "consequents": [
                        {
                            "predicate": "PropertyOf",
                            "relation": None,
                            "arguments": [
                                {
                                    "value": "someone",
                                    "role": "entity",
                                    "type": None,
                                },
                                {
                                    "value": "nice",
                                    "role": "property",
                                    "type": None,
                                },
                            ],
                            "fallback": False,
                            "polarity": "positive",
                            "factuality": "hypothetical",
                            "confidence": 0.99,
                            "source_span": "they are nice",
                            "alternatives": [],
                        }
                    ],
                    "source_span": ("If someone is red then they are nice."),
                }
            ],
        }
    )

    assert result.assertions == []
    assert len(result.rules) == 1
    assert len(result.rules[0].antecedents) == 1
    assert len(result.rules[0].consequents) == 1


def test_accepts_assertions_and_rules_together() -> None:
    data = valid_result_data()

    data["rules"] = [
        {
            "antecedents": [
                {
                    "predicate": "PropertyOf",
                    "relation": None,
                    "arguments": [
                        {
                            "value": "someone",
                            "role": "entity",
                            "type": None,
                        },
                        {
                            "value": "red",
                            "role": "property",
                            "type": None,
                        },
                    ],
                    "fallback": False,
                    "polarity": "positive",
                    "factuality": "hypothetical",
                    "confidence": 0.99,
                    "source_span": "someone is red",
                    "alternatives": [],
                }
            ],
            "consequents": [
                {
                    "predicate": "PropertyOf",
                    "relation": None,
                    "arguments": [
                        {
                            "value": "someone",
                            "role": "entity",
                            "type": None,
                        },
                        {
                            "value": "nice",
                            "role": "property",
                            "type": None,
                        },
                    ],
                    "fallback": False,
                    "polarity": "positive",
                    "factuality": "hypothetical",
                    "confidence": 0.99,
                    "source_span": "they are nice",
                    "alternatives": [],
                }
            ],
            "source_span": ("If someone is red then they are nice."),
        }
    ]

    result = SemanticParseResult.model_validate(data)

    assert len(result.assertions) == 1
    assert len(result.rules) == 1


def test_rejects_empty_semantic_result() -> None:
    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(
            {
                "assertions": [],
                "rules": [],
            }
        )


def test_rejects_rule_without_antecedents() -> None:
    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(
            {
                "assertions": [],
                "rules": [
                    {
                        "antecedents": [],
                        "consequents": [
                            {
                                "predicate": "Evaluation",
                                "relation": "work",
                                "arguments": [],
                                "fallback": True,
                                "polarity": "positive",
                                "factuality": "hypothetical",
                                "confidence": 0.99,
                                "source_span": "it works",
                                "alternatives": [],
                            }
                        ],
                        "source_span": ("If tested then it works."),
                    }
                ],
            }
        )


def test_rejects_rule_without_consequents() -> None:
    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(
            {
                "assertions": [],
                "rules": [
                    {
                        "antecedents": [
                            {
                                "predicate": "Evaluation",
                                "relation": "test",
                                "arguments": [],
                                "fallback": True,
                                "polarity": "positive",
                                "factuality": "hypothetical",
                                "confidence": 0.99,
                                "source_span": "If tested",
                                "alternatives": [],
                            }
                        ],
                        "consequents": [],
                        "source_span": ("If tested then it works."),
                    }
                ],
            }
        )
