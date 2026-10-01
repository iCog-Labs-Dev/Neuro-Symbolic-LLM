from __future__ import annotations

import pytest

from parser.semantic import ReferenceSemanticParser, SemanticParseError
from parser.semantic.schema import SemanticParseResult


def make_result(assertions: list[dict[str, object]]) -> SemanticParseResult:
    """Build structured semantic output for repair/regression tests."""
    return SemanticParseResult.model_validate({"assertions": assertions})


# Coreference repair


def test_rejects_when_unresolved_pronoun_breaks_only_known_predicate() -> None:
    result = make_result(
        [
            {
                "predicate": "StateOf",
                "relation": None,
                "arguments": [
                    {"value": "She", "role": "entity"},
                    {"value": "tired", "role": "state"},
                ],
                "fallback": False,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.9,
                "source_span": "She was tired.",
                "alternatives": [],
            }
        ]
    )

    with pytest.raises(
        SemanticParseError,
        match="No grounded assertions remain",
    ):
        ReferenceSemanticParser.repair_ambiguous_pronouns(result)


def test_removes_unresolved_pronoun_from_evaluation() -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": "be",
                "arguments": [
                    {"value": "he", "role": "agent"},
                    {"value": "late", "role": "patient"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.9,
                "source_span": "he was late",
                "alternatives": [],
            }
        ]
    )

    repaired = ReferenceSemanticParser.repair_ambiguous_pronouns(result)

    assert len(repaired.assertions) == 1

    assertion = repaired.assertions[0]

    assert assertion.predicate == "Evaluation"
    assert assertion.relation == "be"
    assert assertion.fallback is True
    assert [argument.value for argument in assertion.arguments] == ["late"]


def test_preserves_resolved_coreference() -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": "enter",
                "arguments": [
                    {"value": "Alice", "role": "agent"},
                    {"value": "room", "role": "patient"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.99,
                "source_span": "She entered the room.",
                "alternatives": [],
            }
        ]
    )

    repaired = ReferenceSemanticParser.repair_ambiguous_pronouns(result)

    assert len(repaired.assertions) == 1

    assertion = repaired.assertions[0]

    assert assertion.relation == "enter"
    assert [argument.value for argument in assertion.arguments] == [
        "Alice",
        "room",
    ]
    assert [argument.role for argument in assertion.arguments] == [
        "agent",
        "patient",
    ]


def test_rejects_when_only_argument_is_unresolved_pronoun() -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": "leave",
                "arguments": [
                    {"value": "he", "role": "agent"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.7,
                "source_span": "he should leave",
                "alternatives": [],
            }
        ]
    )

    with pytest.raises(
        SemanticParseError,
        match="No grounded assertions remain",
    ):
        ReferenceSemanticParser.repair_ambiguous_pronouns(result)


def test_keeps_grounded_assertion_and_drops_incomplete_known_predicate() -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": "examine",
                "arguments": [
                    {"value": "doctor", "role": "agent"},
                    {"value": "patient", "role": "patient"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.99,
                "source_span": "The doctor examined the patient.",
                "alternatives": [],
            },
            {
                "predicate": "StateOf",
                "relation": None,
                "arguments": [
                    {"value": "She", "role": "entity"},
                    {"value": "tired", "role": "state"},
                ],
                "fallback": False,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.7,
                "source_span": "She was tired.",
                "alternatives": [
                    "entity could be doctor",
                    "entity could be patient",
                ],
            },
        ]
    )

    repaired = ReferenceSemanticParser.repair_ambiguous_pronouns(result)

    assert len(repaired.assertions) == 1

    assertion = repaired.assertions[0]

    assert assertion.predicate == "Evaluation"
    assert assertion.relation == "examine"
    assert [argument.value for argument in assertion.arguments] == [
        "doctor",
        "patient",
    ]


def test_keeps_grounded_assertion_and_drops_pronoun_only_evaluation() -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": "tell",
                "arguments": [
                    {"value": "John", "role": "agent"},
                    {"value": "Peter", "role": "patient"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.99,
                "source_span": "John told Peter",
                "alternatives": [],
            },
            {
                "predicate": "Evaluation",
                "relation": "leave",
                "arguments": [
                    {"value": "he", "role": "agent"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.7,
                "source_span": "he should leave",
                "alternatives": [
                    "agent could be John",
                    "agent could be Peter",
                ],
            },
        ]
    )

    repaired = ReferenceSemanticParser.repair_ambiguous_pronouns(result)

    assert len(repaired.assertions) == 1

    assertion = repaired.assertions[0]

    assert assertion.relation == "tell"
    assert [argument.value for argument in assertion.arguments] == [
        "John",
        "Peter",
    ]


def test_coreference_repair_does_not_mutate_original() -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": "be",
                "arguments": [
                    {"value": "he", "role": "agent"},
                    {"value": "late", "role": "patient"},
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.9,
                "source_span": "he was late",
                "alternatives": [],
            }
        ]
    )

    repaired = ReferenceSemanticParser.repair_ambiguous_pronouns(result)

    assert [argument.value for argument in repaired.assertions[0].arguments] == ["late"]

    assert [argument.value for argument in result.assertions[0].arguments] == [
        "he",
        "late",
    ]


# Semantic role consistency


@pytest.mark.parametrize(
    ("relation", "value", "role"),
    [
        ("work", "machine", "agent"),
        ("fail", "machine", "patient"),
        ("stop", "process", "patient"),
        ("expand", "company", "patient"),
        ("succeed", "product", "patient"),
        ("work", "drug", "agent"),
        ("recover", "patient", "patient"),
        ("sleep", "child", "agent"),
        ("open", "door", "patient"),
        ("increase", "temperature", "entity"),
    ],
)
def test_single_participant_semantic_roles(
    relation: str,
    value: str,
    role: str,
) -> None:
    result = make_result(
        [
            {
                "predicate": "Evaluation",
                "relation": relation,
                "arguments": [
                    {
                        "value": value,
                        "role": role,
                    }
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "asserted",
                "confidence": 0.99,
                "source_span": "test",
                "alternatives": [],
            }
        ]
    )

    assertion = result.assertions[0]

    assert assertion.arguments[0].value == value
    assert assertion.arguments[0].role == role
