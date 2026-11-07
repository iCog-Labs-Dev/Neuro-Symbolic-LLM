"""Unit tests for the current combined semantic extraction contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticExtraction,
    SemanticParseResult,
)

CATALYST_SOURCE = (
    "Although the catalyst doubled the reaction speed, "
    "it did not improve the final yield."
)

TREATMENT_SOURCE = (
    "Although the new treatment reduced the patients' recovery time, "
    "it did not reduce their hospital costs."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def valid_result_data() -> dict[str, Any]:
    """Return a minimal valid semantic parse result."""
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


def catalyst_result_data() -> dict[str, Any]:
    """Return a valid extraction for the catalyst example."""
    data = valid_result_data()
    data["document"]["source_text"] = CATALYST_SOURCE

    data["extraction"]["entities"] = [
        {
            "id": "catalyst",
            "entity_type": "Entity",
            "mention": {"text": "catalyst"},
        },
        {
            "id": "final_yield",
            "entity_type": "Entity",
            "mention": {"text": "final yield"},
        },
    ]

    data["extraction"]["assertions"] = [
        {
            "predicate": "Evaluation",
            "relation": "double",
            "arguments": [
                {"entity_id": "catalyst", "role": "agent"},
            ],
            "fallback": True,
            "confidence": 0.98,
            "evidence": [
                {
                    "text": "the catalyst doubled the reaction speed",
                    "occurrence": 0,
                }
            ],
        },
        {
            "predicate": "Evaluation",
            "relation": "improve",
            "arguments": [
                {"entity_id": "catalyst", "role": "agent"},
                {"entity_id": "final_yield", "role": "theme"},
            ],
            "fallback": True,
            "polarity": "negative",
            "confidence": 0.98,
            "evidence": [
                {
                    "text": "it did not improve the final yield",
                    "occurrence": 0,
                }
            ],
        },
    ]

    return data


def implicit_proposition() -> dict[str, Any]:
    """Return a source-grounded implicit proposition with symbolic meaning."""
    return {
        "id": "expected_1",
        "entity_type": "Proposition",
        "origin": "implicit",
        "mention": None,
        "content": "The catalyst improves the final yield",
        "evidence": [
            {
                "text": "Although the catalyst doubled the reaction speed",
                "occurrence": 0,
            }
        ],
        "symbolic_content": {
            "predicate": "Evaluation",
            "relation": "improve",
            "arguments": [
                {"entity_id": "catalyst", "role": "agent"},
                {"entity_id": "final_yield", "role": "theme"},
            ],
            "fallback": True,
            "polarity": "positive",
            "factuality": "hypothetical",
        },
    }


def append_implicit_proposition(data: dict[str, Any]) -> None:
    """Append an implicit proposition to an extraction.

    catalyst and final_yield must already be declared.
    """
    data["extraction"]["entities"].append(implicit_proposition())


def catalyst_combined_extraction() -> SemanticExtraction:
    """Build a complete general and discourse extraction."""
    data = catalyst_result_data()

    data["extraction"]["entities"].extend(
        [
            {
                "id": "clause_a",
                "entity_type": "Clause",
                "mention": {"text": "Although the catalyst doubled the reaction speed"},
            },
            {
                "id": "clause_b",
                "entity_type": "Clause",
                "mention": {"text": "it did not improve the final yield"},
            },
        ]
    )

    append_implicit_proposition(data)

    data["extraction"]["assertions"].extend(
        [
            {
                "predicate": "Concession",
                "arguments": [
                    {"entity_id": "clause_a", "role": "statement"},
                    {
                        "entity_id": "clause_b",
                        "role": "counter_statement",
                    },
                ],
                "confidence": 0.95,
                "evidence": [{"text": CATALYST_SOURCE}],
            },
            {
                "predicate": "EvokesExpectation",
                "arguments": [
                    {"entity_id": "clause_a", "role": "trigger"},
                    {
                        "entity_id": "expected_1",
                        "role": "expected_result",
                    },
                ],
                "confidence": 0.72,
                "evidence": [
                    {"text": ("Although the catalyst doubled the reaction speed")}
                ],
            },
            {
                "predicate": "Defeats",
                "arguments": [
                    {
                        "entity_id": "clause_b",
                        "role": "defeating_outcome",
                    },
                    {
                        "entity_id": "expected_1",
                        "role": "prior_expectation",
                    },
                ],
                "confidence": 0.85,
                "evidence": [{"text": "it did not improve the final yield"}],
            },
        ]
    )

    return SemanticExtraction.model_validate(data["extraction"])


def treatment_extraction() -> SemanticExtraction:
    """Build a combined general and discourse treatment extraction."""
    entities = [
        {
            "id": "clause_a",
            "entity_type": "Clause",
            "mention": {
                "text": (
                    "Although the new treatment reduced " "the patients' recovery time"
                )
            },
        },
        {
            "id": "clause_b",
            "entity_type": "Clause",
            "mention": {"text": "it did not reduce their hospital costs"},
        },
        {
            "id": "new_treatment",
            "entity_type": "Entity",
            "mention": {"text": "new treatment"},
        },
        {
            "id": "recovery_time",
            "entity_type": "Entity",
            "mention": {"text": "recovery time"},
        },
        {
            "id": "hospital_costs",
            "entity_type": "Entity",
            "mention": {"text": "hospital costs"},
        },
        {
            "id": "expected_1",
            "entity_type": "Proposition",
            "origin": "implicit",
            "mention": None,
            "content": "The new treatment reduces hospital costs",
            "evidence": [
                {
                    "text": (
                        "Although the new treatment reduced "
                        "the patients' recovery time"
                    )
                }
            ],
            "symbolic_content": {
                "predicate": "Evaluation",
                "relation": "reduce",
                "arguments": [
                    {
                        "entity_id": "new_treatment",
                        "role": "agent",
                    },
                    {
                        "entity_id": "hospital_costs",
                        "role": "theme",
                    },
                ],
                "fallback": True,
                "polarity": "positive",
                "factuality": "hypothetical",
            },
        },
    ]

    assertions = [
        {
            "predicate": "Evaluation",
            "relation": "reduce",
            "arguments": [
                {"entity_id": "new_treatment", "role": "agent"},
                {"entity_id": "recovery_time", "role": "theme"},
            ],
            "fallback": True,
            "polarity": "positive",
            "confidence": 0.95,
            "evidence": [
                {"text": ("the new treatment reduced " "the patients' recovery time")}
            ],
        },
        {
            "predicate": "Evaluation",
            "relation": "reduce",
            "arguments": [
                {"entity_id": "new_treatment", "role": "agent"},
                {"entity_id": "hospital_costs", "role": "theme"},
            ],
            "fallback": True,
            "polarity": "negative",
            "confidence": 0.95,
            "evidence": [{"text": "it did not reduce their hospital costs"}],
        },
        {
            "predicate": "Concession",
            "arguments": [
                {"entity_id": "clause_a", "role": "statement"},
                {
                    "entity_id": "clause_b",
                    "role": "counter_statement",
                },
            ],
            "confidence": 0.95,
            "evidence": [{"text": TREATMENT_SOURCE}],
        },
        {
            "predicate": "EvokesExpectation",
            "arguments": [
                {"entity_id": "clause_a", "role": "trigger"},
                {
                    "entity_id": "expected_1",
                    "role": "expected_result",
                },
            ],
            "confidence": 0.70,
            "evidence": [
                {
                    "text": (
                        "Although the new treatment reduced "
                        "the patients' recovery time"
                    )
                }
            ],
        },
        {
            "predicate": "Defeats",
            "arguments": [
                {
                    "entity_id": "clause_b",
                    "role": "defeating_outcome",
                },
                {
                    "entity_id": "expected_1",
                    "role": "prior_expectation",
                },
            ],
            "confidence": 0.85,
            "evidence": [{"text": "it did not reduce their hospital costs"}],
        },
        {
            "predicate": "DiscourseFocus",
            "arguments": [{"entity_id": "clause_b", "role": "focal_item"}],
            "confidence": 0.90,
            "evidence": [{"text": "it did not reduce their hospital costs"}],
        },
    ]

    return SemanticExtraction.model_validate(
        {
            "entities": entities,
            "assertions": assertions,
            "rules": [],
        }
    )


# ---------------------------------------------------------------------------
# Structured contract tests
# ---------------------------------------------------------------------------


def test_accepts_the_current_structured_parse_contract():
    result = SemanticParseResult.model_validate(valid_result_data())

    assert result.contract_version == SEMANTIC_CONTRACT_VERSION
    assert result.document.document_id == "fixture-1"
    assert result.extraction.assertions[0].predicate == "Evaluation"
    assert result.extraction.assertions[0].arguments[0].entity_id == "mira"


def test_applies_optional_field_defaults():
    data = valid_result_data()
    assertion = data["extraction"]["assertions"][0]

    for field in (
        "fallback",
        "polarity",
        "factuality",
        "modality",
        "alternatives",
    ):
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
    [
        ("polarity", "uncertain"),
        ("factuality", "possible"),
        ("modality", "required"),
    ],
)
def test_rejects_unsupported_assertion_classification(field, value):
    data = valid_result_data()
    data["extraction"]["assertions"][0][field] = value

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_rejects_confidence_outside_unit_interval(confidence):
    data = valid_result_data()
    data["extraction"]["assertions"][0]["confidence"] = confidence

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


@pytest.mark.parametrize(
    "level",
    ["result", "document", "assertion", "argument"],
)
def test_forbids_unrecognized_fields(level):
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


def test_requires_at_least_one_assertion_or_rule():
    data = valid_result_data()
    data["extraction"]["assertions"] = []

    with pytest.raises(
        ValidationError,
        match="at least one assertion or rule",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_duplicate_entities_and_unresolved_arguments():
    duplicate = valid_result_data()

    duplicate["extraction"]["entities"].append(
        deepcopy(duplicate["extraction"]["entities"][0])
    )

    with pytest.raises(ValidationError, match="Entity IDs must be unique"):
        SemanticParseResult.model_validate(duplicate)

    unresolved = valid_result_data()
    unresolved["extraction"]["assertions"][0]["arguments"][0]["entity_id"] = "missing"

    with pytest.raises(
        ValidationError,
        match="unknown entity or rule variable",
    ):
        SemanticParseResult.model_validate(unresolved)


def test_locates_repeated_source_quotes_by_occurrence():
    data = valid_result_data()
    data["document"]["source_text"] = "Mira saw Mira."
    data["extraction"]["assertions"][0]["evidence"] = [
        {"text": "Mira", "occurrence": 1}
    ]

    result = SemanticParseResult.model_validate(data)

    assert result.locate(result.extraction.assertions[0].evidence[0]) == (
        9,
        13,
    )


def test_rejects_source_quotes_that_do_not_resolve():
    data = valid_result_data()
    data["extraction"]["assertions"][0]["evidence"] = [
        {"text": "Mira", "occurrence": 1}
    ]

    with pytest.raises(
        ValidationError,
        match="occurrence 1 is not in the source text",
    ):
        SemanticParseResult.model_validate(data)


def test_accepts_a_specific_rule_with_hypothetical_literals():
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


def test_rejects_rule_with_non_hypothetical_literal():
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

    with pytest.raises(
        ValidationError,
        match="must have hypothetical factuality",
    ):
        SemanticParseResult.model_validate(data)


# ---------------------------------------------------------------------------
# Explicit entity tests
# ---------------------------------------------------------------------------


def test_accepts_explicit_entity_with_source_mention():
    result = SemanticParseResult.model_validate(valid_result_data())
    entity = result.extraction.entities[0]

    assert entity.origin == "explicit"
    assert entity.mention.text == "Mira"
    assert entity.content is None
    assert entity.evidence == []
    assert entity.symbolic_content is None


def test_rejects_explicit_entity_without_mention():
    data = valid_result_data()
    data["extraction"]["entities"][0].pop("mention")

    with pytest.raises(
        ValidationError,
        match="Explicit entities require",
    ):
        SemanticParseResult.model_validate(data)


def test_defaults_missing_origin_to_explicit():
    data = valid_result_data()
    data["extraction"]["entities"][0].pop("origin", None)

    result = SemanticParseResult.model_validate(data)

    assert result.extraction.entities[0].origin == "explicit"


def test_rejects_explicit_entity_with_implicit_content():
    data = valid_result_data()
    data["extraction"]["entities"][0]["content"] = "Some content"

    with pytest.raises(
        ValidationError,
        match="Explicit entities cannot contain implicit",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_explicit_entity_with_symbolic_content():
    data = valid_result_data()
    data["extraction"]["entities"][0]["symbolic_content"] = {
        "predicate": "Evaluation",
        "relation": "leave",
        "arguments": [{"entity_id": "mira", "role": "agent"}],
        "fallback": True,
        "factuality": "hypothetical",
    }

    with pytest.raises(
        ValidationError,
        match="symbolic_content",
    ):
        SemanticParseResult.model_validate(data)


# ---------------------------------------------------------------------------
# Implicit proposition tests
# ---------------------------------------------------------------------------


def test_accepts_implicit_proposition():
    data = catalyst_result_data()
    append_implicit_proposition(data)

    result = SemanticParseResult.model_validate(data)

    proposition = next(
        entity for entity in result.extraction.entities if entity.id == "expected_1"
    )

    assert proposition.origin == "implicit"
    assert proposition.entity_type == "Proposition"
    assert proposition.mention is None
    assert proposition.content == ("The catalyst improves the final yield")
    assert proposition.symbolic_content.predicate == "Evaluation"
    assert proposition.symbolic_content.relation == "improve"
    assert proposition.symbolic_content.factuality == "hypothetical"


def test_rejects_implicit_proposition_without_content():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["content"] = None
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(ValidationError, match="semantic content"):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_without_evidence():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["evidence"] = []
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(ValidationError, match="supporting evidence"):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_non_proposition_entity():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["entity_type"] = "Event"
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="Only implicit propositions",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_with_source_mention():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["mention"] = {"text": "The catalyst improves the final yield"}
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="cannot have a source mention",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_with_invalid_evidence():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["evidence"] = [{"text": "The catalyst improves the final yield"}]
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="not in the source text",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_without_symbolic_content():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition.pop("symbolic_content")
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="Implicit propositions require symbolic_content",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_non_hypothetical_symbolic_content():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["symbolic_content"]["factuality"] = "asserted"
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(data)


def test_rejects_symbolic_content_without_evaluation_relation():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["symbolic_content"]["relation"] = None
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="requires a relation",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_symbolic_content_without_fallback():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["symbolic_content"]["fallback"] = False
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="requires fallback=true",
    ):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_with_unknown_symbolic_argument():
    data = catalyst_result_data()
    proposition = implicit_proposition()

    proposition["symbolic_content"]["arguments"][1]["entity_id"] = "missing_entity"

    data["extraction"]["entities"].append(proposition)

    with pytest.raises(
        ValidationError,
        match="unknown entity",
    ):
        SemanticParseResult.model_validate(data)


def test_locates_implicit_proposition_evidence():
    data = catalyst_result_data()
    append_implicit_proposition(data)

    result = SemanticParseResult.model_validate(data)

    proposition = next(
        entity for entity in result.extraction.entities if entity.id == "expected_1"
    )

    quote = proposition.evidence[0]
    start, end = result.locate(quote)

    assert result.document.source_text[start:end] == quote.text


# ---------------------------------------------------------------------------
# Combined general and discourse extraction tests
# ---------------------------------------------------------------------------


def test_combined_extraction_contains_general_and_discourse_semantics():
    extraction = catalyst_combined_extraction()

    predicates = [assertion.predicate for assertion in extraction.assertions]

    assert "Evaluation" in predicates
    assert "Concession" in predicates
    assert "EvokesExpectation" in predicates
    assert "Defeats" in predicates


def test_combined_extraction_declares_all_argument_entities():
    extraction = catalyst_combined_extraction()

    known_ids = {entity.id for entity in extraction.entities}

    for assertion in extraction.assertions:
        for argument in assertion.arguments:
            assert argument.entity_id in known_ids


def test_combined_extraction_reuses_shared_implicit_expectation():
    extraction = catalyst_combined_extraction()

    expectation = next(
        assertion
        for assertion in extraction.assertions
        if assertion.predicate == "EvokesExpectation"
    )

    defeat = next(
        assertion
        for assertion in extraction.assertions
        if assertion.predicate == "Defeats"
    )

    assert expectation.arguments[1].entity_id == "expected_1"
    assert defeat.arguments[1].entity_id == "expected_1"


def test_combined_extraction_preserves_general_assertions():
    extraction = catalyst_combined_extraction()

    evaluations = [
        assertion
        for assertion in extraction.assertions
        if assertion.predicate == "Evaluation"
    ]

    assert len(evaluations) == 2
    assert {assertion.polarity for assertion in evaluations} == {
        "positive",
        "negative",
    }


def test_combined_extraction_rejects_missing_implicit_expectation():
    extraction = catalyst_combined_extraction()

    extraction_data = extraction.model_dump(mode="python")
    extraction_data["entities"] = [
        entity for entity in extraction_data["entities"] if entity["id"] != "expected_1"
    ]

    with pytest.raises(
        ValidationError,
        match="unknown entity",
    ):
        SemanticExtraction.model_validate(extraction_data)


def test_combined_extraction_rejects_duplicate_entity_ids():
    extraction = catalyst_combined_extraction()
    extraction_data = extraction.model_dump(mode="python")

    extraction_data["entities"].append(deepcopy(extraction_data["entities"][0]))

    with pytest.raises(
        ValidationError,
        match="Entity IDs must be unique",
    ):
        SemanticExtraction.model_validate(extraction_data)


def test_combined_extraction_rejects_undeclared_argument():
    extraction = catalyst_combined_extraction()
    extraction_data = extraction.model_dump(mode="python")

    extraction_data["assertions"][0]["arguments"][0]["entity_id"] = "unknown_entity"

    with pytest.raises(
        ValidationError,
        match="unknown entity",
    ):
        SemanticExtraction.model_validate(extraction_data)


def test_combined_extraction_preserves_specific_rules():
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

    assert len(result.extraction.rules) == 1
    assert result.extraction.rules[0].scope == "specific"


def test_combined_extraction_produces_one_parse_result():
    data = catalyst_result_data()
    combined = catalyst_combined_extraction()

    data["extraction"] = combined.model_dump(mode="python")
    data["document"]["document_id"] = "combined-001"

    result = SemanticParseResult.model_validate(data)

    assert isinstance(result, SemanticParseResult)
    assert result.document.document_id == "combined-001"

    predicates = {assertion.predicate for assertion in result.extraction.assertions}

    assert {
        "Evaluation",
        "Concession",
        "EvokesExpectation",
        "Defeats",
    }.issubset(predicates)

    assert any(
        entity.id == "expected_1"
        and entity.origin == "implicit"
        and entity.symbolic_content is not None
        for entity in result.extraction.entities
    )


def test_combined_extraction_preserves_evidence_grounding():
    data = catalyst_result_data()
    data["extraction"] = catalyst_combined_extraction().model_dump(mode="python")

    result = SemanticParseResult.model_validate(data)

    for entity in result.extraction.entities:
        if entity.mention is not None:
            start, end = result.locate(entity.mention)
            assert result.document.source_text[start:end] == (entity.mention.text)

        for quote in entity.evidence:
            start, end = result.locate(quote)
            assert result.document.source_text[start:end] == quote.text

    for assertion in result.extraction.assertions:
        for quote in assertion.evidence:
            start, end = result.locate(quote)
            assert result.document.source_text[start:end] == quote.text


# ---------------------------------------------------------------------------
# Treatment regression tests
# ---------------------------------------------------------------------------


def test_treatment_combined_extraction_contains_all_relations():
    extraction = treatment_extraction()

    predicates = [assertion.predicate for assertion in extraction.assertions]

    assert predicates.count("Evaluation") == 2
    assert predicates.count("Concession") == 1
    assert predicates.count("EvokesExpectation") == 1
    assert predicates.count("Defeats") == 1
    assert predicates.count("DiscourseFocus") == 1


def test_treatment_declares_shared_implicit_expectation():
    extraction = treatment_extraction()

    entity_ids = {entity.id for entity in extraction.entities}

    assert "expected_1" in entity_ids

    expectation = next(
        assertion
        for assertion in extraction.assertions
        if assertion.predicate == "EvokesExpectation"
    )

    defeat = next(
        assertion
        for assertion in extraction.assertions
        if assertion.predicate == "Defeats"
    )

    assert expectation.arguments[1].entity_id == "expected_1"
    assert defeat.arguments[1].entity_id == "expected_1"


def test_treatment_preserves_positive_and_negative_polarity():
    extraction = treatment_extraction()

    evaluations = [
        assertion
        for assertion in extraction.assertions
        if assertion.predicate == "Evaluation"
    ]

    assert len(evaluations) == 2
    assert {assertion.polarity for assertion in evaluations} == {
        "positive",
        "negative",
    }


def test_treatment_symbolic_expectation_is_hypothetical():
    extraction = treatment_extraction()

    proposition = next(
        entity for entity in extraction.entities if entity.id == "expected_1"
    )

    assert proposition.symbolic_content is not None
    assert proposition.symbolic_content.predicate == "Evaluation"
    assert proposition.symbolic_content.relation == "reduce"
    assert proposition.symbolic_content.factuality == "hypothetical"


def test_treatment_combined_extraction_is_source_grounded():
    data = valid_result_data()
    data["document"]["source_text"] = TREATMENT_SOURCE
    data["extraction"] = treatment_extraction().model_dump(mode="python")

    result = SemanticParseResult.model_validate(data)

    assert len(result.extraction.entities) == 6
    assert len(result.extraction.assertions) == 6
    assert result.extraction.rules == []


def test_treatment_rejects_missing_implicit_expectation():
    extraction = treatment_extraction()
    data = extraction.model_dump(mode="python")

    data["entities"] = [
        entity for entity in data["entities"] if entity["id"] != "expected_1"
    ]

    with pytest.raises(
        ValidationError,
        match="unknown entity",
    ):
        SemanticExtraction.model_validate(data)
