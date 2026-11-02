"""Typed universal-rule conversion without assertion or polarity loss."""

from copy import deepcopy

import pytest

from parser.semantic.schema import SemanticParseResult
from symbolic_pipeline.miner_adapter import MinerAdapterError, to_semantic_document
from symbolic_pipeline.miner_bundle import load_catalog_bundle


def rule_result() -> SemanticParseResult:
    catalog, _ = load_catalog_bundle()
    text = "For every object, if it is red then it is not blue."

    def part(property_id, evidence, polarity="positive"):
        return {
            "predicate": "PropertyOf",
            "arguments": [
                {"entity_id": "$x", "role": "entity"},
                {"entity_id": property_id, "role": "property"},
            ],
            "factuality": "hypothetical",
            "polarity": polarity,
            "confidence": 0.9,
            "evidence": [{"text": evidence}],
        }

    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "rule-probe",
                "corpus_id": "rules",
                "source_text": text,
            },
            "provenance": {
                "provider": "test",
                "model": "fixture",
                "parser_version": "test",
                "prompt_version": "test",
                "ontology_versions": list(catalog.versions),
            },
            "extraction": {
                "entities": [
                    {"id": value, "entity_type": "Entity", "mention": {"text": value}}
                    for value in ("red", "blue")
                ],
                "assertions": [],
                "rules": [
                    {
                        "scope": "universal",
                        "variables": [{"id": "$x", "entity_type": "Entity"}],
                        "antecedents": [part("red", "it is red")],
                        "consequents": [part("blue", "it is not blue", "negative")],
                        "evidence": [{"text": text}],
                    }
                ],
            },
        }
    )


def test_preserves_signed_rule_without_asserting_its_parts():
    source = rule_result()
    before = source.model_dump_json()
    document = to_semantic_document(source)
    _, bundle = load_catalog_bundle()
    bundle.validate_document(document)
    assert source.model_dump_json() == before
    assert document.assertions == ()
    assert len(document.rules) == 1
    rule = document.rules[0]
    assert rule.id == "rule-0"
    assert rule.quantifier == "universal"
    assert [(v.id, v.type) for v in rule.variables] == [("$x", "Entity")]
    assert rule.antecedents[0].arguments == ("$x", "red")
    assert rule.consequents[0].arguments == ("$x", "blue")
    assert rule.antecedents[0].polarity == "positive"
    assert rule.consequents[0].polarity == "negative"
    assert all(
        part.factuality == "hypothetical"
        for part in (*rule.antecedents, *rule.consequents)
    )
    assert all(entity.id != "$x" for entity in document.entities)
    assert rule.source_spans[0].start == 0
    assert rule.source_spans[0].end == len(source.document.source_text)
    span = rule.consequents[0].source_spans[0]
    assert source.document.source_text[span.start : span.end] == "it is not blue"


def test_rule_variables_remain_local_and_rule_ids_distinct():
    payload = rule_result().model_dump()
    payload["extraction"]["rules"].append(deepcopy(payload["extraction"]["rules"][0]))
    document = to_semantic_document(SemanticParseResult.model_validate(payload))
    assert [rule.id for rule in document.rules] == ["rule-0", "rule-1"]
    assert document.rules[0].variables == document.rules[1].variables
    assert document.assertions == ()


def test_unused_and_head_only_variables_are_not_silently_removed():
    payload = rule_result().model_dump()
    rule = payload["extraction"]["rules"][0]
    rule["variables"].append({"id": "$y", "entity_type": "Entity"})
    rule["consequents"][0]["arguments"][0]["entity_id"] = "$y"
    with pytest.raises(MinerAdapterError, match="consequent variables must occur"):
        to_semantic_document(SemanticParseResult.model_validate(payload))


def test_modal_rule_literal_is_rejected_without_dropping_modality():
    payload = rule_result().model_dump()
    payload["extraction"]["rules"][0]["consequents"][0]["modality"] = "obligation"
    with pytest.raises(MinerAdapterError, match="refusing to discard"):
        to_semantic_document(SemanticParseResult.model_validate(payload))


def test_evaluation_rule_does_not_invent_existential_event_variables():
    payload = rule_result().model_dump()
    part = payload["extraction"]["rules"][0]["consequents"][0]
    part.update(predicate="Evaluation", relation="repair", fallback=True)
    with pytest.raises(MinerAdapterError, match="event reification"):
        to_semantic_document(SemanticParseResult.model_validate(payload))


def test_literal_alternatives_and_evidence_are_retained():
    payload = rule_result().model_dump()
    payload["extraction"]["rules"][0]["antecedents"][0]["alternatives"] = [
        "reading_a",
        "reading_b",
    ]
    document = to_semantic_document(SemanticParseResult.model_validate(payload))
    literal = document.rules[0].antecedents[0]
    assert literal.alternatives == ("reading_a", "reading_b")
    assert literal.confidence == 0.9
    assert literal.argument_types == ("Entity", "Entity")


def test_both_conjunctions_and_negative_antecedents_are_preserved():
    payload = rule_result().model_dump()
    rule = payload["extraction"]["rules"][0]
    other = deepcopy(rule["consequents"][0])
    rule["antecedents"].append(other)
    rule["consequents"].append(deepcopy(rule["antecedents"][0]))
    document = to_semantic_document(SemanticParseResult.model_validate(payload))
    assert len(document.rules[0].antecedents) == 2
    assert len(document.rules[0].consequents) == 2
    assert document.rules[0].antecedents[1].polarity == "negative"
    assert document.rules[0].consequents[1].polarity == "positive"
    assert document.assertions == ()


def test_emission_keeps_rule_clauses_inside_declarative_graph(tmp_path):
    from hybrid_miner.emission.emit_petta import emit_petta

    document = to_semantic_document(rule_result())
    _, bundle = load_catalog_bundle()
    source = tmp_path / "documents.jsonl"
    source.write_text(document.canonical_json() + "\n", encoding="utf-8")
    result = emit_petta(
        bundle=bundle,
        input_path=source,
        atoms_path=tmp_path / "facts.metta",
        metadata_path=tmp_path / "metadata.metta",
        provenance_path=tmp_path / "provenance.jsonl",
    )
    atoms = (tmp_path / "facts.metta").read_text(encoding="utf-8").splitlines()
    assert result.assertions == 0
    assert result.rules == 1
    assert len(atoms) == 3
    assert atoms[0].startswith("(ExtractedRule@semantic-rule-v1 ")
    assert atoms[1].startswith("(RuleAntecedentPositive@semantic-rule-v1 ")
    assert atoms[2].startswith("(RuleConsequentNegative@semantic-rule-v1 ")
    assert all(not atom.startswith("(PropertyOf") for atom in atoms)
