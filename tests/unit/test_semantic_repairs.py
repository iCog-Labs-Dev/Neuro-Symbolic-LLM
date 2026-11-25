"""Tests for semantic parser normalization and repair behavior."""

from __future__ import annotations

import pytest

from parser.semantic import ReferenceSemanticParser, SemanticParseError
from parser.semantic.backends import CallableBackend
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import SemanticParserConfig
from tests.semantic_fixtures import result as fixture_result


def _parser() -> ReferenceSemanticParser:
    return ReferenceSemanticParser(
        backend=CallableBackend(lambda **_: "{}"),
        config=SemanticParserConfig(model_name="fixture"),
    )


def _assertion(
    predicate: str,
    entity_id: str,
    evidence: str,
    *,
    relation: str | None = None,
    fallback: bool = False,
    factuality: str = "asserted",
) -> dict:
    return {
        "predicate": predicate,
        "relation": relation,
        "arguments": [{"entity_id": entity_id, "role": "event_id"}],
        "fallback": fallback,
        "factuality": factuality,
        "confidence": 0.9,
        "evidence": [{"text": evidence}],
    }


def _result(source: str, extraction: dict) -> SemanticParseResult:
    base = fixture_result(source)
    data = base.model_dump(mode="json")
    data["extraction"] = extraction
    return SemanticParseResult.model_validate(data)


def test_unresolved_pronoun_is_kept_and_recorded_as_a_repair() -> None:
    result = _result(
        "She left.",
        {
            "entities": [
                {"id": "she", "entity_type": "Entity", "mention": {"text": "She"}}
            ],
            "assertions": [_assertion("Event", "she", "She left.")],
            "rules": [],
        },
    )

    repaired, changes = _parser().repair_ambiguous_pronouns(result)

    assert repaired.extraction.assertions[0].arguments[0].entity_id == "she"
    assert [change.kind for change in changes] == ["unresolved_pronoun"]
    assert result.provenance.repairs == []


def test_pronoun_kind_is_cleared_without_mutating_input() -> None:
    result = _result(
        "She left.",
        {
            "entities": [
                {
                    "id": "she",
                    "entity_type": "Entity",
                    "kind": "she",
                    "mention": {"text": "She"},
                }
            ],
            "assertions": [_assertion("Event", "she", "She left.")],
            "rules": [],
        },
    )

    repaired, changes = _parser().repair_ambiguous_pronouns(result)

    assert repaired.extraction.entities[0].kind is None
    assert result.extraction.entities[0].kind == "she"
    assert {change.kind for change in changes} == {"pronoun_kind", "unresolved_pronoun"}


def test_known_predicate_contract_repair_clears_relation_and_fallback() -> None:
    result = _result(
        "Mira left.",
        {
            "entities": [
                {"id": "mira", "entity_type": "Entity", "mention": {"text": "Mira"}}
            ],
            "assertions": [
                _assertion(
                    "Event", "mira", "Mira left.", relation="leave", fallback=True
                )
            ],
            "rules": [],
        },
    )

    repaired, changes = _parser().repair_known_predicate_contract(result)

    assertion = repaired.extraction.assertions[0]
    assert assertion.relation is None
    assert assertion.fallback is False
    assert [change.kind for change in changes] == ["predicate_contract"]
    assert result.extraction.assertions[0].relation == "leave"


def test_unresolved_pronoun_in_universal_rule_binds_to_its_variable() -> None:
    result = _result(
        "They shall leave.",
        {
            "entities": [
                {"id": "they", "entity_type": "Entity", "mention": {"text": "They"}}
            ],
            "assertions": [],
            "rules": [
                {
                    "scope": "universal",
                    "variables": [{"id": "$person", "entity_type": "Entity"}],
                    "antecedents": [
                        _assertion(
                            "Evaluation",
                            "they",
                            "They",
                            relation="leave",
                            fallback=True,
                            factuality="hypothetical",
                        )
                    ],
                    "consequents": [
                        _assertion(
                            "Evaluation",
                            "they",
                            "leave",
                            relation="leave",
                            fallback=True,
                            factuality="hypothetical",
                        )
                    ],
                    "evidence": [{"text": "They shall leave."}],
                }
            ],
        },
    )

    repaired, changes = _parser().repair_ambiguous_pronouns(result)

    rule = repaired.extraction.rules[0]
    assert rule.antecedents[0].arguments[0].entity_id == "$person"
    assert rule.consequents[0].arguments[0].entity_id == "$person"
    assert [change.kind for change in changes] == ["pronoun_in_rule", "pronoun_in_rule"]


@pytest.mark.parametrize("role", ["agent", "patient", "entity"])
def test_semantic_argument_roles_are_preserved(role: str) -> None:
    result = fixture_result("Mira left.")
    argument = result.extraction.assertions[0].arguments[0]
    argument.role = role

    reparsed = SemanticParseResult.model_validate(result.model_dump())
    assert reparsed.extraction.assertions[0].arguments[0].role == role


def test_rejects_unresolved_pronoun_with_multiple_universal_variables() -> None:
    result = _result(
        "She shall leave.",
        {
            "entities": [
                {"id": "she", "entity_type": "Entity", "mention": {"text": "She"}}
            ],
            "assertions": [],
            "rules": [
                {
                    "scope": "universal",
                    "variables": [
                        {"id": "$person", "entity_type": "Entity"},
                        {"id": "$agent", "entity_type": "Entity"},
                    ],
                    "antecedents": [
                        _assertion(
                            "Evaluation",
                            "she",
                            "She",
                            relation="leave",
                            fallback=True,
                            factuality="hypothetical",
                        )
                    ],
                    "consequents": [
                        _assertion(
                            "Evaluation",
                            "she",
                            "leave",
                            relation="leave",
                            fallback=True,
                            factuality="hypothetical",
                        )
                    ],
                    "evidence": [{"text": "She shall leave."}],
                }
            ],
        },
    )

    with pytest.raises(SemanticParseError, match="universal rule"):
        _parser().repair_ambiguous_pronouns(result)
