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


def test_converts_repeated_source_quotes_to_character_spans() -> None:
    source = "Mira left. Mira returned."
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
        (0, 4),
        (11, 15),
    ]
    assert [(span.start, span.end) for span in document.assertions[0].source_spans] == [
        (11, 24)
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


def test_rejects_modality_on_a_catalog_predicate() -> None:
    source = "The supplier must own the equipment."
    result = SemanticParseResult.model_validate(
        {
            "document": {
                "document_id": "catalog-modality",
                "corpus_id": "unit-tests",
                "source_text": source,
            },
            "provenance": _provenance(),
            "extraction": {
                "entities": [
                    {
                        "id": "supplier",
                        "entity_type": "Entity",
                        "mention": {"text": "supplier"},
                    },
                    {
                        "id": "equipment",
                        "entity_type": "Entity",
                        "mention": {"text": "equipment"},
                    },
                ],
                "assertions": [
                    {
                        "predicate": "Has",
                        "arguments": [
                            {"entity_id": "supplier", "role": "owner"},
                            {"entity_id": "equipment", "role": "possessed"},
                        ],
                        "modality": "obligation",
                        "confidence": 0.9,
                        "evidence": [{"text": source}],
                    }
                ],
                "rules": [],
            },
        }
    )

    with pytest.raises(MinerAdapterError, match="obligation on Has"):
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
