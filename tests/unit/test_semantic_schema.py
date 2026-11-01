"""Unit tests for the versioned semantic parser output contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from parser.semantic.catalog_prompt import build_discourse_prompt
from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    DiscourseExtraction,
    SemanticEntity,
    SemanticExtraction,
    SemanticParseResult,
    SourceQuote,
)
from parser.semantic.semantic_parser import _BaseSemanticParser

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
    data = valid_result_data()
    data["document"]["source_text"] = CATALYST_SOURCE

    data["extraction"]["entities"][0] = {
        "id": "catalyst",
        "entity_type": "Entity",
        "mention": {
            "text": "catalyst",
            "occurrence": 0,
        },
    }

    data["extraction"]["assertions"][0] = {
        "predicate": "Evaluation",
        "relation": "increase",
        "arguments": [{"entity_id": "catalyst", "role": "agent"}],
        "fallback": True,
        "confidence": 0.98,
        "evidence": [
            {
                "text": "the catalyst doubled the reaction speed",
                "occurrence": 0,
            }
        ],
    }

    return data


def implicit_proposition() -> dict[str, Any]:
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
    }


def clause_a() -> SemanticEntity:
    return SemanticEntity(
        id="clause_a",
        entity_type="Clause",
        mention=SourceQuote(text="the catalyst doubled the reaction speed"),
    )


def clause_b() -> SemanticEntity:
    return SemanticEntity(
        id="clause_b",
        entity_type="Clause",
        mention=SourceQuote(text="it did not improve the final yield"),
    )


def expected_proposition() -> SemanticEntity:
    return SemanticEntity.model_validate(implicit_proposition())


def general_result() -> SemanticParseResult:
    data = catalyst_result_data()

    data["document"]["document_id"] = "doc_001"
    data["extraction"]["entities"].extend(
        [
            clause_a().model_dump(mode="python"),
            clause_b().model_dump(mode="python"),
        ]
    )

    return SemanticParseResult.model_validate(data)


def discourse_result() -> DiscourseExtraction:
    return DiscourseExtraction.model_validate(
        {
            "entities": [
                clause_a(),
                clause_b(),
                expected_proposition(),
            ],
            "assertions": [
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
                        {
                            "text": (
                                "Although the catalyst doubled " "the reaction speed"
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
                    "evidence": [{"text": "it did not improve the final yield"}],
                },
            ],
        }
    )


def merge_extractions(
    general: SemanticExtraction,
    discourse: DiscourseExtraction,
) -> SemanticExtraction:
    """Call the merger implemented in semantic_parser.py."""
    return _BaseSemanticParser.merge_discourse(general, discourse)


def merge_discourse_result(
    general: SemanticParseResult,
    discourse: DiscourseExtraction,
) -> SemanticParseResult:
    """Merge and validate the complete output contract."""
    merged = merge_extractions(general.extraction, discourse)

    data = general.model_dump(mode="python")
    data["extraction"] = merged.model_dump(mode="python")

    return SemanticParseResult.model_validate(data)


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

    assert result.locate(result.extraction.assertions[0].evidence[0]) == (9, 13)


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


# ---------------------------------------------------------------------------
# Implicit proposition tests
# ---------------------------------------------------------------------------


def test_accepts_implicit_proposition():
    data = catalyst_result_data()
    data["extraction"]["entities"].append(implicit_proposition())

    result = SemanticParseResult.model_validate(data)
    proposition = result.extraction.entities[1]

    assert proposition.id == "expected_1"
    assert proposition.origin == "implicit"
    assert proposition.entity_type == "Proposition"
    assert proposition.mention is None
    assert proposition.content == "The catalyst improves the final yield"


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

    with pytest.raises(ValidationError, match="Only implicit propositions"):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_with_fabricated_mention():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["mention"] = {"text": "The catalyst improves the final yield"}
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(ValidationError, match="fabricated mention"):
        SemanticParseResult.model_validate(data)


def test_rejects_implicit_proposition_with_invalid_evidence():
    data = catalyst_result_data()
    proposition = implicit_proposition()
    proposition["evidence"] = [{"text": "The catalyst improves the final yield"}]
    data["extraction"]["entities"].append(proposition)

    with pytest.raises(ValidationError, match="not in the source text"):
        SemanticParseResult.model_validate(data)


def test_locates_implicit_proposition_evidence():
    data = catalyst_result_data()
    data["extraction"]["entities"].append(implicit_proposition())

    result = SemanticParseResult.model_validate(data)
    quote = result.extraction.entities[1].evidence[0]

    start, end = result.locate(quote)

    assert result.document.source_text[start:end] == quote.text


# ---------------------------------------------------------------------------
# Discourse schema tests
# ---------------------------------------------------------------------------


def test_accepts_shared_implicit_proposition_in_discourse_relations():
    data = catalyst_result_data()

    data["extraction"]["entities"].extend(
        [
            clause_a().model_dump(mode="python"),
            clause_b().model_dump(mode="python"),
            implicit_proposition(),
        ]
    )

    discourse = discourse_result()

    data["extraction"]["assertions"].extend(
        [assertion.model_dump(mode="python") for assertion in discourse.assertions]
    )

    result = SemanticParseResult.model_validate(data)

    expectations = [
        assertion
        for assertion in result.extraction.assertions
        if assertion.predicate in ("EvokesExpectation", "Defeats")
    ]

    assert len(expectations) == 2
    assert expectations[0].arguments[1].entity_id == "expected_1"
    assert expectations[1].arguments[1].entity_id == "expected_1"


def test_accepts_empty_discourse_extraction():
    extraction = DiscourseExtraction()

    assert extraction.entities == []
    assert extraction.assertions == []


def test_accepts_discourse_extraction_with_implicit_proposition():
    extraction = DiscourseExtraction(
        entities=[
            clause_a(),
            expected_proposition(),
        ],
        assertions=[
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
                    {"text": "Although the catalyst doubled the reaction speed"}
                ],
            }
        ],
    )

    assert extraction.entities[1].origin == "implicit"


def test_rejects_duplicate_discourse_entity_ids():
    proposition = implicit_proposition()

    with pytest.raises(ValidationError, match="unique"):
        DiscourseExtraction(entities=[proposition, deepcopy(proposition)])


def test_discourse_schema_allows_cross_stage_entity_references():
    """The merger, not the discourse schema, checks cross-stage IDs."""
    extraction = DiscourseExtraction(
        entities=[implicit_proposition()],
        assertions=[
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
                    {"text": "Although the catalyst doubled the reaction speed"}
                ],
            }
        ],
    )

    assert extraction.assertions[0].arguments[0].entity_id == "clause_a"


# ---------------------------------------------------------------------------
# Discourse prompt tests
# ---------------------------------------------------------------------------


CLAUSES = """
clause_a: the catalyst doubled the reaction speed
clause_b: it did not improve the final yield
"""

GENERAL = """
The catalyst increased reaction speed.
The catalyst did not improve final yield.
"""


def test_builds_discourse_prompt():
    prompt = build_discourse_prompt(
        CATALYST_SOURCE,
        CLAUSES,
        GENERAL,
    )

    assert prompt.version.startswith("discourse-v1+sha256.")
    assert CATALYST_SOURCE in prompt.text
    assert "Concession" in prompt.text
    assert "EvokesExpectation" in prompt.text
    assert "Defeats" in prompt.text


def test_includes_general_semantic_context():
    prompt = build_discourse_prompt(
        CATALYST_SOURCE,
        CLAUSES,
        GENERAL,
    )

    assert "The catalyst increased reaction speed." in prompt.text


def test_rejects_empty_source():
    with pytest.raises(ValueError, match="source_text"):
        build_discourse_prompt("", CLAUSES, GENERAL)


def test_rejects_empty_clause_context():
    with pytest.raises(ValueError, match="clause_context"):
        build_discourse_prompt(CATALYST_SOURCE, "", GENERAL)


# ---------------------------------------------------------------------------
# Discourse merger tests
# ---------------------------------------------------------------------------


def test_merges_general_and_discourse_entities():
    merged = merge_extractions(
        general_result().extraction,
        discourse_result(),
    )

    ids = [entity.id for entity in merged.entities]

    assert len(ids) == len(set(ids))
    assert "catalyst" in ids
    assert "clause_a" in ids
    assert "clause_b" in ids
    assert "expected_1" in ids


def test_reuses_identical_clause_entities():
    merged = merge_extractions(
        general_result().extraction,
        discourse_result(),
    )

    matches = [entity for entity in merged.entities if entity.id == "clause_a"]

    assert len(matches) == 1


def test_adds_implicit_proposition():
    merged = merge_extractions(
        general_result().extraction,
        discourse_result(),
    )

    proposition = next(
        entity for entity in merged.entities if entity.id == "expected_1"
    )

    assert proposition.origin == "implicit"
    assert proposition.content == "The catalyst improves the final yield"


def test_preserves_general_and_adds_discourse_assertions():
    merged = merge_extractions(
        general_result().extraction,
        discourse_result(),
    )

    predicates = [a.predicate for a in merged.assertions]

    assert predicates.count("Evaluation") == 1
    assert predicates.count("Concession") == 1
    assert predicates.count("EvokesExpectation") == 1
    assert predicates.count("Defeats") == 1


def test_preserves_general_rules():
    general = general_result().extraction

    merged = merge_extractions(
        general,
        DiscourseExtraction(),
    )

    assert merged.rules == general.rules


def test_rejects_conflicting_entity_definitions():
    general = general_result().extraction

    conflicting = SemanticEntity(
        id="clause_a",
        entity_type="Clause",
        mention=SourceQuote(text="it did not improve the final yield"),
    )

    discourse = DiscourseExtraction(entities=[conflicting])

    with pytest.raises(Exception, match="conflicts"):
        merge_extractions(general, discourse)


def test_rejects_unknown_discourse_entity():
    general = general_result().extraction

    discourse = DiscourseExtraction(
        entities=[expected_proposition()],
        assertions=[
            {
                "predicate": "EvokesExpectation",
                "arguments": [
                    {
                        "entity_id": "missing_clause",
                        "role": "trigger",
                    },
                    {
                        "entity_id": "expected_1",
                        "role": "expected_result",
                    },
                ],
                "confidence": 0.72,
                "evidence": [
                    {"text": "Although the catalyst doubled the reaction speed"}
                ],
            }
        ],
    )

    with pytest.raises(Exception, match="unknown entity"):
        merge_extractions(general, discourse)


def test_rejects_invalid_discourse_evidence():
    general = general_result()

    invalid = DiscourseExtraction(
        entities=[
            clause_a(),
            clause_b(),
            expected_proposition(),
        ],
        assertions=[
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
                "evidence": [{"text": "This sentence does not exist"}],
            }
        ],
    )

    with pytest.raises(ValidationError):
        merge_discourse_result(general, invalid)


def test_produces_one_validated_semantic_parse_result():
    result = merge_discourse_result(
        general_result(),
        discourse_result(),
    )

    assert isinstance(result, SemanticParseResult)
    assert result.document.document_id == "doc_001"

    predicates = [assertion.predicate for assertion in result.extraction.assertions]

    assert "Evaluation" in predicates
    assert "Concession" in predicates
    assert "EvokesExpectation" in predicates
    assert "Defeats" in predicates

    assert any(
        entity.id == "expected_1" and entity.origin == "implicit"
        for entity in result.extraction.entities
    )


def test_merges_empty_discourse_result():
    result = merge_discourse_result(
        general_result(),
        DiscourseExtraction(),
    )

    assert len(result.extraction.assertions) == 1
    assert result.extraction.assertions[0].predicate == "Evaluation"


def test_does_not_duplicate_discourse_assertions():
    general = general_result().extraction
    discourse = discourse_result()

    once = merge_extractions(general, discourse)
    twice = merge_extractions(once, discourse)

    assert len(twice.assertions) == len(once.assertions)


# ---------------------------------------------------------------------------
# Treatment example regression tests
# ---------------------------------------------------------------------------


def treatment_general() -> SemanticExtraction:
    return SemanticExtraction.model_validate(
        {
            "entities": [
                {
                    "id": "clause_a",
                    "entity_type": "Clause",
                    "mention": {
                        "text": (
                            "Although the new treatment reduced "
                            "the patients' recovery time"
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
            ],
            "assertions": [
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
                        {
                            "text": (
                                "the new treatment reduced "
                                "the patients' recovery time"
                            )
                        }
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
            ],
            "rules": [],
        }
    )


def treatment_discourse() -> DiscourseExtraction:
    general = treatment_general()

    entities = [entity.model_dump(mode="python") for entity in general.entities]

    entities.append(
        {
            "id": "expected_1",
            "entity_type": "Proposition",
            "origin": "implicit",
            "mention": None,
            "content": "the new treatment reduces hospital costs",
            "evidence": [
                {
                    "text": (
                        "Although the new treatment reduced "
                        "the patients' recovery time"
                    )
                }
            ],
        }
    )

    return DiscourseExtraction.model_validate(
        {
            "entities": entities,
            "assertions": [
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
            ],
        }
    )


def test_treatment_merges_all_discourse_relations():
    merged = merge_extractions(
        treatment_general(),
        treatment_discourse(),
    )

    predicates = [a.predicate for a in merged.assertions]

    assert predicates.count("Evaluation") == 2
    assert predicates.count("Concession") == 1
    assert predicates.count("EvokesExpectation") == 1
    assert predicates.count("Defeats") == 1
    assert predicates.count("DiscourseFocus") == 1


def test_treatment_declares_shared_implicit_expectation():
    merged = merge_extractions(
        treatment_general(),
        treatment_discourse(),
    )

    entity_ids = {entity.id for entity in merged.entities}

    assert "expected_1" in entity_ids

    expectation = next(
        a for a in merged.assertions if a.predicate == "EvokesExpectation"
    )

    defeat = next(a for a in merged.assertions if a.predicate == "Defeats")

    assert expectation.arguments[1].entity_id == "expected_1"
    assert defeat.arguments[1].entity_id == "expected_1"


def test_treatment_preserves_positive_and_negative_polarity():
    merged = merge_extractions(
        treatment_general(),
        treatment_discourse(),
    )

    evaluations = [a for a in merged.assertions if a.predicate == "Evaluation"]

    assert len(evaluations) == 2
    assert {a.polarity for a in evaluations} == {
        "positive",
        "negative",
    }


def test_treatment_rejects_missing_implicit_expectation():
    discourse = treatment_discourse()

    discourse.entities = [
        entity for entity in discourse.entities if entity.id != "expected_1"
    ]

    with pytest.raises(Exception, match="unknown entity"):
        merge_extractions(treatment_general(), discourse)
