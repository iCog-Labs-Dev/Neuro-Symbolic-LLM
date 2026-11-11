"""Tests for source evidence and rule conversion at the miner boundary."""

import json
import os
from pathlib import Path

import pytest

from parser.semantic.schema import SemanticParseResult

pytest.importorskip("hybrid_miner.contracts.semantic_document")

from symbolic_pipeline.miner_adapter import MinerAdapterError  # noqa: E402


def _adapt(result: SemanticParseResult):
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


def test_negative_evaluation_uses_signed_relation_fact() -> None:
    result = SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "negative-evaluation",
                "corpus_id": "unit-tests",
                "source_text": "The lamp was not repaired by Sora.",
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {
                        "id": "lamp",
                        "entity_type": "Entity",
                        "mention": {"text": "lamp"},
                    },
                    {
                        "id": "sora",
                        "entity_type": "Entity",
                        "mention": {"text": "Sora"},
                    },
                ],
                "assertions": [
                    {
                        "predicate": "Evaluation",
                        "relation": "repair",
                        "arguments": [
                            {"entity_id": "lamp", "role": "patient"},
                            {"entity_id": "sora", "role": "agent"},
                        ],
                        "polarity": "negative",
                        "fallback": True,
                        "confidence": 0.95,
                        "evidence": [{"text": "The lamp was not repaired by Sora."}],
                    }
                ],
                "rules": [],
            },
        }
    )

    document = _adapt(result)

    relation_fact = next(
        fact
        for fact in document.assertions
        if fact.predicate == "ProvisionalRelation"
        and fact.provisional_relation_id == "repair"
    )
    assert relation_fact.polarity == "negative"
    assert not any(fact.predicate == "Negated" for fact in document.assertions)


def _modal_result(document_id, source, assertions, entities):
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": document_id,
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {"id": e, "entity_type": "Entity", "mention": {"text": m}}
                    for e, m in entities
                ],
                "assertions": assertions,
                "rules": [],
            },
        }
    )


@pytest.mark.parametrize(
    ("source", "quote", "polarity", "modality"),
    [
        ("Nora may leave.", "may leave", "positive", "permission"),
        ("Nora must leave.", "must leave", "positive", "obligation"),
        ("Nora must not leave.", "must not leave", "negative", "obligation"),
        (
            "Nora is allowed not to leave.",
            "allowed not to leave",
            "negative",
            "permission",
        ),
    ],
)
def test_modal_evaluation_signs_its_relation_fact(source, quote, polarity, modality):
    result = _modal_result(
        "modal-evaluation",
        source,
        [
            {
                "predicate": "Evaluation",
                "relation": "leave",
                "fallback": True,
                "polarity": polarity,
                "modality": modality,
                "confidence": 0.9,
                "arguments": [{"entity_id": "nora", "role": "agent"}],
                "evidence": [{"text": quote}],
            }
        ],
        [("nora", "Nora")],
    )

    document = _adapt(result)

    relation = next(
        fact for fact in document.assertions if fact.provisional_relation_id == "leave"
    )
    assert (relation.polarity, relation.modality) == (polarity, modality)
    # no separate modality fact next to a plain "leave" fact
    assert not any(
        fact.provisional_relation_id in {"permission", "obligation"}
        for fact in document.assertions
    )


def test_modal_catalog_fact_keeps_its_modality() -> None:
    result = _modal_result(
        "modal-has",
        "The supplier must own the equipment.",
        [
            {
                "predicate": "Has",
                "modality": "obligation",
                "confidence": 0.9,
                "arguments": [
                    {"entity_id": "supplier", "role": "owner"},
                    {"entity_id": "equipment", "role": "possessed"},
                ],
                "evidence": [{"text": "must own"}],
            }
        ],
        [("supplier", "supplier"), ("equipment", "equipment")],
    )

    document = _adapt(result)

    has = next(fact for fact in document.assertions if fact.predicate == "Has")
    assert (has.polarity, has.modality) == ("positive", "obligation")


def test_negative_non_evaluation_assertion_is_preserved() -> None:
    result = SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "negative-has",
                "corpus_id": "unit-tests",
                "source_text": "Nora does not own the car.",
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {
                        "id": "nora",
                        "entity_type": "Entity",
                        "mention": {"text": "Nora"},
                    },
                    {"id": "car", "entity_type": "Entity", "mention": {"text": "car"}},
                ],
                "assertions": [
                    {
                        "predicate": "Has",
                        "arguments": [
                            {"entity_id": "nora", "role": "owner"},
                            {"entity_id": "car", "role": "possessed"},
                        ],
                        "polarity": "negative",
                        "confidence": 0.95,
                        "evidence": [{"text": "Nora does not own the car."}],
                    }
                ],
                "rules": [],
            },
        }
    )

    document = _adapt(result)

    has_fact = next(fact for fact in document.assertions if fact.predicate == "Has")
    assert has_fact.polarity == "negative"


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


_EXAMPLES = (
    Path(__file__).parents[2] / "configs/parser_config/catalog_document_examples.jsonl"
)


def _example_result(source_text: str) -> SemanticParseResult:
    """Build a parse result from one worked example, with catalog versions."""

    from parser.ontology.catalog import load_ontology_catalog

    for line in _EXAMPLES.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["source_text"] == source_text:
            break
    else:
        raise AssertionError(f"no worked example for {source_text!r}")
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "worked-example",
                "corpus_id": "unit-tests",
                "source_text": source_text,
            },
            "provenance": {
                **_provenance(),
                "ontology_versions": list(load_ontology_catalog().versions),
            },
            "extraction": {
                key: record[key] for key in ("entities", "assertions", "rules")
            },
        }
    )


def _universal_rule_result(
    antecedents: list[dict[str, object]],
    consequents: list[dict[str, object]],
    variables: list[dict[str, str]] | None = None,
) -> SemanticParseResult:
    """Build a one-rule parse of "All red things are nice."."""

    source = "All red things are nice."
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "universal-rule",
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [],
                "assertions": [],
                "rules": [
                    {
                        "scope": "universal",
                        "variables": (
                            variables
                            if variables is not None
                            else [{"id": "$x", "entity_type": "Entity"}]
                        ),
                        "antecedents": antecedents,
                        "consequents": consequents,
                        "evidence": [{"text": source}],
                    }
                ],
            },
        }
    )


def _part(
    relation: str, arguments: list[dict[str, str]], **fields: object
) -> dict[str, object]:
    """One hypothetical Evaluation rule part quoting the whole sentence."""

    return {
        "predicate": "Evaluation",
        "relation": relation,
        "arguments": arguments,
        "fallback": True,
        "factuality": "hypothetical",
        "confidence": 0.95,
        "evidence": [{"text": "All red things are nice."}],
        **fields,
    }


_THEME_X = [{"entity_id": "$x", "role": "theme"}]


@pytest.mark.parametrize(
    "source_text",
    [
        "All red things are nice.",
        "No red container is blue.",
        "All red and round things are nice.",
        "All red containers are heavy.",
    ],
)
def test_universal_worked_examples_become_valid_miner_rules(source_text: str) -> None:
    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    document = _adapt(_example_result(source_text))
    _, bundle = load_catalog_bundle()

    bundle.validate_document(document)
    assert len(document.rules) == 1
    assert document.assertions == ()


def test_universal_rule_keeps_negative_consequent_and_roles() -> None:
    document = _adapt(_example_result("No red container is blue."))
    rule = document.rules[0]

    assert [variable.id for variable in rule.variables] == ["$x"]
    assert [literal.provisional_relation_id for literal in rule.antecedents] == [
        "container",
        "red",
    ]
    consequent = rule.consequents[0]
    assert consequent.provisional_relation_id == "blue"
    assert consequent.polarity == "negative"
    assert consequent.arguments == ("$x",)


def test_universal_rule_is_emitted_as_rule_graph(tmp_path: Path) -> None:
    from hybrid_miner.emission.emit_petta import emit_petta

    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    document = _adapt(_example_result("No red container is blue."))
    _, bundle = load_catalog_bundle()
    source = tmp_path / "documents.jsonl"
    source.write_text(document.model_dump_json() + "\n", encoding="utf-8")

    result = emit_petta(
        bundle=bundle,
        input_path=source,
        atoms_path=tmp_path / "atoms.metta",
        metadata_path=tmp_path / "metadata.metta",
        provenance_path=tmp_path / "provenance.jsonl",
    )

    atoms = (tmp_path / "atoms.metta").read_text(encoding="utf-8")
    assert result.rules == 1
    assert "RuleConsequentNegative" in atoms
    assert ":blue " in atoms


def test_rejects_modality_inside_universal_rule() -> None:
    result = _universal_rule_result(
        [_part("red", _THEME_X)],
        [_part("nice", _THEME_X, modality="obligation")],
    )

    with pytest.raises(MinerAdapterError, match="obligation in a universal rule"):
        _adapt(result)


def test_rejects_universal_evaluation_without_participants() -> None:
    result = _universal_rule_result(
        [_part("red", _THEME_X), _part("rain", [])],
        [_part("nice", _THEME_X)],
    )

    with pytest.raises(MinerAdapterError, match="without participants"):
        _adapt(result)


def test_rejects_variable_that_appears_only_in_consequent() -> None:
    result = _universal_rule_result(
        [_part("employee", _THEME_X)],
        [
            _part(
                "have",
                [
                    {"entity_id": "$x", "role": "agent"},
                    {"entity_id": "$y", "role": "patient"},
                ],
            )
        ],
        variables=[
            {"id": "$x", "entity_type": "Entity"},
            {"id": "$y", "entity_type": "Entity"},
        ],
    )

    with pytest.raises(MinerAdapterError, match="consequent variables"):
        _adapt(result)


def _all_worked_examples() -> list[SemanticParseResult]:
    """Every worked example, each with its own document ID."""

    results = []
    lines = _EXAMPLES.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        result = _example_result(json.loads(line)["source_text"])
        document = result.document.model_copy(
            update={"document_id": f"worked-example-{index}"}
        )
        results.append(result.model_copy(update={"document": document}))
    return results


def test_every_worked_example_reaches_the_miner(tmp_path: Path) -> None:
    """All worked examples convert, and PeTTa admits every emitted fact.

    The PeTTa part needs the runtime of hybrid-miner's CI; set PETTA_ROOT and
    MINER_ROOT to its pinned checkouts to run it.
    """

    from hybrid_miner.emission.emit_petta import emit_petta

    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    documents = [_adapt(result) for result in _all_worked_examples()]
    _, bundle = load_catalog_bundle()
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "".join(document.canonical_json() + "\n" for document in documents),
        encoding="utf-8",
    )
    paths = {
        "atoms_path": tmp_path / "atoms.metta",
        "metadata_path": tmp_path / "metadata.metta",
        "provenance_path": tmp_path / "provenance.jsonl",
    }
    emitted = emit_petta(bundle=bundle, input_path=source, **paths)

    assert len(documents) == len(_EXAMPLES.read_text(encoding="utf-8").splitlines())
    assert emitted.rules == 4

    petta = os.environ.get("PETTA_ROOT", "")
    miner = os.environ.get("MINER_ROOT", "")
    if not petta or not miner:
        pytest.skip("PETTA_ROOT and MINER_ROOT are not set; PeTTa admission skipped")
    from hybrid_miner.mining.admitted_support import mine_admitted_support

    run = mine_admitted_support(
        bundle=bundle,
        documents_path=source,
        emitted_atoms_path=paths["atoms_path"],
        metadata_path=paths["metadata_path"],
        provenance_path=paths["provenance_path"],
        petta_root=Path(petta),
        miner_root=Path(miner),
        admitted_atoms_path=tmp_path / "admitted.metta",
        quarantine_path=tmp_path / "quarantine.jsonl",
        support_path=tmp_path / "support.jsonl",
        min_support=2,
    )
    assert (run.admission.accepted, run.admission.rejected) == (emitted.facts, 0)


def _norm_result(*, with_fact: bool, second_rule: bool = False) -> SemanticParseResult:
    """A universal rule with an obligation, which the miner rule cannot hold."""

    source = "Nora signed the register. Every visitor must sign the register."
    whole = [{"text": "Every visitor must sign the register."}]
    visitor = {
        "predicate": "Evaluation",
        "relation": "visitor",
        "arguments": [{"entity_id": "$x", "role": "theme"}],
        "fallback": True,
        "factuality": "hypothetical",
        "confidence": 0.9,
        "evidence": whole,
    }
    sign = {
        **visitor,
        "relation": "sign",
        "arguments": [
            {"entity_id": "$x", "role": "agent"},
            {"entity_id": "book", "role": "patient"},
        ],
    }
    rule = {
        "scope": "universal",
        "variables": [{"id": "$x", "entity_type": "Entity"}],
        "antecedents": [visitor],
        "consequents": [{**sign, "modality": "obligation"}],
        "evidence": whole,
    }
    entities = [
        {
            "id": "book",
            "entity_type": "Entity",
            "kind": "register",
            "mention": {"text": "register", "occurrence": 1},
        }
    ]
    assertions = []
    if with_fact:
        entities.append(
            {"id": "nora", "entity_type": "Entity", "mention": {"text": "Nora"}}
        )
        assertions.append(
            {
                "predicate": "Evaluation",
                "relation": "sign",
                "arguments": [{"entity_id": "nora", "role": "agent"}],
                "fallback": True,
                "confidence": 0.9,
                "evidence": [{"text": "Nora signed the register."}],
            }
        )
    rules = [rule, {**rule, "consequents": [sign]}] if second_rule else [rule]
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "norm",
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": entities,
                "assertions": assertions,
                "rules": rules,
            },
        }
    )


def test_unsupported_universal_rule_is_refused_unless_skipping_is_asked() -> None:
    from symbolic_pipeline.miner_adapter import to_semantic_document

    result = _norm_result(with_fact=True)

    with pytest.raises(MinerAdapterError, match="obligation in a universal rule"):
        to_semantic_document(result)

    skipped: list[str] = []
    document = to_semantic_document(result, skipped_rules=skipped)

    assert len(skipped) == 1 and "obligation in a universal rule" in skipped[0]
    assert document.rules == ()
    # "Nora signed" is kept; nothing of the skipped rule becomes a fact
    relations = [fact.provisional_relation_id for fact in document.assertions]
    assert relations == ["sign", None]
    assert [fact.predicate for fact in document.assertions][1] == "Agent"


def test_skipping_keeps_supported_rules_and_their_numbering() -> None:
    from symbolic_pipeline.miner_adapter import to_semantic_document

    skipped: list[str] = []
    document = to_semantic_document(
        _norm_result(with_fact=False, second_rule=True), skipped_rules=skipped
    )

    assert len(skipped) == 1
    assert [rule.id for rule in document.rules] == ["rule-0"]
    assert document.rules[0].consequents[0].provisional_relation_id == "sign"
    # the register is still used by the kept rule, so its kind fact stays
    assert [fact.provisional_relation_id for fact in document.assertions] == [
        "kind_register"
    ]


def test_document_with_nothing_left_after_skipping_is_still_refused() -> None:
    from symbolic_pipeline.miner_adapter import to_semantic_document

    with pytest.raises(MinerAdapterError, match="Nothing is left after skipping"):
        to_semantic_document(_norm_result(with_fact=False), skipped_rules=[])


def test_export_lists_skipped_rules_only_when_asked(tmp_path: Path) -> None:
    from symbolic_pipeline.document_export import export_semantic_documents
    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    catalog, bundle = load_catalog_bundle()
    result = _norm_result(with_fact=True)
    provenance = result.provenance.model_copy(
        update={"ontology_versions": list(catalog.versions)}
    )
    source = tmp_path / "results.jsonl"
    source.write_text(
        result.model_copy(update={"provenance": provenance}).model_dump_json() + "\n",
        encoding="utf-8",
    )

    strict = export_semantic_documents(
        source, tmp_path / "strict.jsonl", tmp_path / "strict.rej", bundle=bundle
    )
    assert (strict.accepted, strict.rejected, strict.skipped_rules) == (0, 1, 0)

    kept = export_semantic_documents(
        source,
        tmp_path / "kept.jsonl",
        tmp_path / "kept.rej",
        bundle=bundle,
        skipped_path=tmp_path / "kept.skipped.jsonl",
    )
    assert (kept.accepted, kept.rejected, kept.skipped_rules) == (1, 0, 1)
    line = json.loads((tmp_path / "kept.skipped.jsonl").read_text(encoding="utf-8"))
    assert line["document_id"] == "norm"
    assert "obligation in a universal rule" in line["skipped_rules"][0]


def _declared_event_result(
    document_id: str, source: str, *, negated: str | None
) -> SemanticParseResult:
    """A declared repair event; ``negated`` is the polarity of its Negated fact."""

    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    catalog, _ = load_catalog_bundle()
    whole = [{"text": source}]
    assertions = [
        {
            "predicate": "Event",
            "arguments": [{"entity_id": "fixing", "role": "event_id"}],
            "confidence": 0.9,
            "evidence": whole,
        },
        {
            "predicate": "Agent",
            "arguments": [
                {"entity_id": "fixing", "role": "action"},
                {"entity_id": "sora", "role": "actor"},
            ],
            "confidence": 0.9,
            "evidence": whole,
        },
    ]
    if negated is not None:
        assertions.append(
            {
                "predicate": "Negated",
                "arguments": [{"entity_id": "fixing", "role": "target"}],
                "polarity": negated,
                "confidence": 0.9,
                "evidence": whole,
            }
        )
    return SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": document_id,
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": {
                **_provenance(),
                "ontology_versions": list(catalog.versions),
            },
            "extraction": {
                "entities": [
                    {
                        "id": "sora",
                        "entity_type": "Entity",
                        "mention": {"text": "Sora"},
                    },
                    {
                        "id": "fixing",
                        "entity_type": "Event",
                        "kind": "repair",
                        "mention": {"text": "repaired"},
                    },
                ],
                "assertions": assertions,
                "rules": [],
            },
        }
    )


def _signs(document) -> dict[str, str]:
    return {
        fact.provisional_relation_id or fact.predicate: fact.polarity
        for fact in document.assertions
    }


def test_negated_declared_event_signs_the_facts_that_say_it_happened() -> None:
    done = _adapt(
        _declared_event_result("done", "The lamp was repaired by Sora.", negated=None)
    )
    undone = _adapt(
        _declared_event_result(
            "undone", "The lamp was not repaired by Sora.", negated="positive"
        )
    )

    assert _signs(done) == {
        "Event": "positive",
        "Agent": "positive",
        "kind_repair": "positive",
    }
    assert _signs(undone) == {
        "Event": "negative",
        "Agent": "positive",
        "Negated": "positive",
        "kind_repair": "negative",
    }


def test_double_negation_is_refused() -> None:
    result = _declared_event_result(
        "double",
        "It is not true that the lamp was repaired by Sora.",
        negated="negative",
    )

    with pytest.raises(MinerAdapterError, match="double negation"):
        _adapt(result)


def test_negated_declared_event_never_matches_the_done_pattern(tmp_path: Path) -> None:
    """Real PeTTa: the repair that did not happen is not found as a repair."""

    petta = os.environ.get("PETTA_ROOT", "")
    miner = os.environ.get("MINER_ROOT", "")
    if not petta or not miner:
        pytest.skip("PETTA_ROOT and MINER_ROOT are not set")
    from hybrid_miner.emission import symbols
    from hybrid_miner.emission.emit_petta import emit_petta
    from hybrid_miner.mining.admitted_support import mine_admitted_support
    from hybrid_miner.mining.exact_matches import _query_bindings

    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    _, bundle = load_catalog_bundle()
    documents = [
        _adapt(_declared_event_result(name, source, negated=negated))
        for name, source, negated in (
            ("done", "The lamp was repaired by Sora.", None),
            ("done-too", "The bike was repaired by Sora.", None),
            ("undone", "The lamp was not repaired by Sora.", "positive"),
        )
    ]
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "".join(document.canonical_json() + "\n" for document in documents),
        encoding="utf-8",
    )
    paths = {
        "atoms_path": tmp_path / "atoms.metta",
        "metadata_path": tmp_path / "metadata.metta",
        "provenance_path": tmp_path / "provenance.jsonl",
    }
    emitted = emit_petta(bundle=bundle, input_path=source, **paths)
    run = mine_admitted_support(
        bundle=bundle,
        documents_path=source,
        emitted_atoms_path=paths["atoms_path"],
        metadata_path=paths["metadata_path"],
        provenance_path=paths["provenance_path"],
        petta_root=Path(petta),
        miner_root=Path(miner),
        admitted_atoms_path=tmp_path / "admitted.metta",
        quarantine_path=tmp_path / "quarantine.jsonl",
        support_path=tmp_path / "support.jsonl",
        min_support=2,
    )
    assert (run.admission.accepted, run.admission.rejected) == (emitted.facts, 0)

    version = f"{bundle.bundle_id}@{bundle.bundle_version}"
    kind = "rel:kind_repair"
    queries = {
        0: (
            f"({symbols.fact_head('ProvisionalRelation', version)} {kind} $e)",
            ("$e",),
        ),
        1: (
            f"({symbols.fact_head('ProvisionalRelation', version, 'negative')} "
            f"{kind} $e)",
            ("$e",),
        ),
    }
    found = _query_bindings(
        queries,
        atoms_path=tmp_path / "admitted.metta",
        petta_root=Path(petta),
        swipl="swipl",
        timeout_seconds=120,
    )

    def event(name: str) -> tuple[str]:
        return (symbols.entity_symbol("unit-tests", name, "fixing"),)

    assert sorted(found[0]) == sorted([event("done"), event("done-too")])
    assert found[1] == [event("undone")]
