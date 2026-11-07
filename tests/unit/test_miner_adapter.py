"""Tests for source evidence and rule conversion at the miner boundary."""

import pytest

from parser.semantic.schema import SemanticParseResult


def _adapt(result: SemanticParseResult):
    pytest.importorskip("hybrid_miner.contracts.semantic_document")
    from symbolic_pipeline.miner_adapter import to_semantic_document

    return to_semantic_document(result)


def _provenance() -> dict[str, object]:
    return {
        "provider": "test",
        "model": "fixture",
        "parser_version": "test-v1",
        "prompt_version": "test-v1",
        "ontology_versions": ["upper-v1"],
    }


def test_converts_repeated_unicode_source_quotes_to_character_spans() -> None:
    source = "🙂 Mira left. Mira returned."
    result = SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "unicode-source",
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {
                        "id": "mira_first",
                        "entity_type": "Entity",
                        "mention": {"text": "Mira", "occurrence": 0},
                    },
                    {
                        "id": "mira_second",
                        "entity_type": "Entity",
                        "mention": {"text": "Mira", "occurrence": 1},
                    },
                ],
                "assertions": [
                    {
                        "predicate": "Has",
                        "arguments": [
                            {"entity_id": "mira_first", "role": "owner"},
                            {"entity_id": "mira_second", "role": "possessed"},
                        ],
                        "confidence": 0.9,
                        "evidence": [{"text": "Mira returned", "occurrence": 0}],
                    }
                ],
                "rules": [],
            },
        }
    )

    document = _adapt(result)

    assert [(entity.span.start, entity.span.end) for entity in document.entities] == [
        (2, 6),
        (13, 17),
    ]
    assert [(span.start, span.end) for span in document.assertions[0].source_spans] == [
        (13, 26)
    ]


def test_specific_rule_preserves_literal_evidence_and_clause_span() -> None:
    source = "Mira is red. Mira is nice."
    result = SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "rule-source",
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {
                        "id": "mira",
                        "entity_type": "Entity",
                        "mention": {"text": "Mira"},
                    },
                    {
                        "id": "red",
                        "entity_type": "Entity",
                        "mention": {"text": "red"},
                    },
                    {
                        "id": "nice",
                        "entity_type": "Entity",
                        "mention": {"text": "nice"},
                    },
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
                                    {"entity_id": "red", "role": "property"},
                                ],
                                "factuality": "hypothetical",
                                "confidence": 0.9,
                                "evidence": [{"text": "Mira is red."}],
                            }
                        ],
                        "consequents": [
                            {
                                "predicate": "PropertyOf",
                                "arguments": [
                                    {"entity_id": "mira", "role": "entity"},
                                    {"entity_id": "nice", "role": "property"},
                                ],
                                "factuality": "hypothetical",
                                "confidence": 0.9,
                                "evidence": [{"text": "Mira is nice."}],
                            }
                        ],
                        "evidence": [{"text": source}],
                    }
                ],
            },
        }
    )

    document = _adapt(result)
    clause = next(entity for entity in document.entities if entity.type == "Clause")
    antecedent_span = document.assertions[0].source_spans[0]
    consequent_span = document.assertions[1].source_spans[0]

    assert (clause.span.start, clause.span.end) == (0, len("Mira is red."))
    assert (antecedent_span.start, antecedent_span.end) == (0, len("Mira is red."))
    assert document.assertions[1].conditional_on == clause.id
    assert (consequent_span.start, consequent_span.end) == (
        source.index("Mira is nice."),
        len(source),
    )


def test_rules_with_shared_antecedent_reuse_conditional_clause() -> None:
    source = "If the alarm rings, Alice will leave, " "but Bob will remain."
    first_clause = "If the alarm rings, Alice will leave"
    second_clause = "but Bob will remain"

    result = SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "shared-antecedent",
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {
                        "id": "alarm",
                        "entity_type": "Entity",
                        "mention": {"text": "alarm"},
                    },
                    {
                        "id": "alice",
                        "entity_type": "Entity",
                        "mention": {"text": "Alice"},
                    },
                    {
                        "id": "bob",
                        "entity_type": "Entity",
                        "mention": {"text": "Bob"},
                    },
                ],
                "assertions": [],
                "rules": [
                    {
                        "scope": "specific",
                        "antecedents": [
                            {
                                "predicate": "Evaluation",
                                "relation": "ring",
                                "arguments": [{"entity_id": "alarm", "role": "theme"}],
                                "fallback": True,
                                "factuality": "hypothetical",
                                "confidence": 0.95,
                                "evidence": [{"text": first_clause}],
                            }
                        ],
                        "consequents": [
                            {
                                "predicate": "Evaluation",
                                "relation": "leave",
                                "arguments": [{"entity_id": "alice", "role": "agent"}],
                                "fallback": True,
                                "factuality": "hypothetical",
                                "confidence": 0.95,
                                "evidence": [{"text": "Alice will leave"}],
                            }
                        ],
                        "evidence": [{"text": first_clause}],
                    },
                    {
                        "scope": "specific",
                        "antecedents": [
                            {
                                "predicate": "Evaluation",
                                "relation": "ring",
                                "arguments": [{"entity_id": "alarm", "role": "theme"}],
                                "fallback": True,
                                "factuality": "hypothetical",
                                "confidence": 0.95,
                                "evidence": [{"text": second_clause}],
                            }
                        ],
                        "consequents": [
                            {
                                "predicate": "Evaluation",
                                "relation": "remain",
                                "arguments": [{"entity_id": "bob", "role": "agent"}],
                                "fallback": True,
                                "factuality": "hypothetical",
                                "confidence": 0.95,
                                "evidence": [{"text": second_clause}],
                            }
                        ],
                        "evidence": [{"text": second_clause}],
                    },
                ],
            },
        }
    )

    document = _adapt(result)

    conditional_clauses = [
        entity
        for entity in document.entities
        if entity.type == "Clause" and entity.id.startswith("clause-")
    ]

    assert len(conditional_clauses) == 1

    clause = conditional_clauses[0]

    ring_facts = [
        assertion
        for assertion in document.assertions
        if assertion.provisional_relation_id == "ring"
    ]

    assert len(ring_facts) == 1

    consequent_facts = {
        assertion.provisional_relation_id: assertion
        for assertion in document.assertions
        if assertion.provisional_relation_id in {"leave", "remain"}
    }

    assert set(consequent_facts) == {"leave", "remain"}

    assert consequent_facts["leave"].conditional_on == clause.id
    assert consequent_facts["remain"].conditional_on == clause.id

    assert all(
        assertion.factuality == "hypothetical"
        for assertion in consequent_facts.values()
    )

    assert (
        clause.span.start,
        clause.span.end,
    ) == (0, len(first_clause))
