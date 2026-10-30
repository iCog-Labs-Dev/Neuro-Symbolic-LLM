"""Integration tests for catalog-guided document parsing (contract 2.0.0).

A fixed model answer goes through the real pipeline: prompt, normalization,
extraction validation, recorded repairs, and validation against the full
ontology catalog.
"""

import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.schema import (
    SemanticExtraction,
    SemanticParseResult,
    SourceDocument,
)
from parser.semantic.semantic_parser import (
    PARSER_VERSION,
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)


class RecordingBackend:
    provider_name = "test-provider"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return json.dumps(self.payload)


class SchemaBackend(RecordingBackend):
    """A backend that supports JSON-schema decoding, like Ollama."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload)
        self.schemas: list[dict[str, Any]] = []

    def generate_json(self, *, prompt: str, model: str, schema: dict[str, Any]) -> str:
        self.schemas.append(schema)
        return self.generate(prompt=prompt, model=model)


def _evidence(text: str) -> list[dict[str, Any]]:
    return [{"text": text}]


def _concession() -> tuple[str, dict[str, Any]]:
    source = "🙂 Although Mira promised Sol, she delayed."
    first = "Although Mira promised Sol"
    second = "she delayed"
    return source, {
        "entities": [
            {"id": "clause_a", "entity_type": "Clause", "mention": {"text": first}},
            {"id": "clause_b", "entity_type": "Clause", "mention": {"text": second}},
        ],
        "assertions": [
            {
                "predicate": "Concession",
                "arguments": [
                    {"entity_id": "clause_a", "role": "statement"},
                    {"entity_id": "clause_b", "role": "counter_statement"},
                ],
                "confidence": 0.9,
                "evidence": _evidence(source[2:]),
            }
        ],
    }


def _parse(
    payload: dict[str, Any], source: str, **parser_options: Any
) -> SemanticParseResult:
    parser = ReferenceSemanticParser(
        backend=RecordingBackend(payload),
        config=SemanticParserConfig(model_name="teacher-model"),
        **parser_options,
    )
    return parser.parse_document(
        SourceDocument(document_id="doc-1", corpus_id="corpus-1", source_text=source)
    )


def test_returns_validated_result_with_runtime_provenance() -> None:
    source, payload = _concession()
    catalog = load_ontology_catalog()
    backend = RecordingBackend(payload)
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        model_revision="revision-1",
    )

    result = parser.parse_document(
        SourceDocument(document_id="doc-1", corpus_id="corpus-1", source_text=source)
    )

    assert result.document.source_text == source
    assert result.provenance.parser_version == PARSER_VERSION
    assert (
        result.provenance.prompt_version
        == build_catalog_document_prompt(source, catalog=catalog).version
    )
    assert result.provenance.ontology_versions == list(catalog.versions)
    assert result.provenance.provider == "test-provider"
    assert result.provenance.model_revision == "revision-1"
    assert result.provenance.repairs == []
    assert result.locate(result.extraction.assertions[0].evidence[0]) == (
        2,
        len(source),
    )
    assert f"<source_text>\n{source}\n</source_text>" in backend.calls[0][0]
    assert SemanticParseResult.model_validate_json(result.model_dump_json()) == result


def test_event_and_agent_share_the_same_typed_event() -> None:
    source = "Mira left."
    payload = {
        "entities": [
            {"id": "mira", "entity_type": "Entity", "mention": {"text": "Mira"}},
            {"id": "departure", "entity_type": "Event", "mention": {"text": "left"}},
        ],
        "assertions": [
            {
                "predicate": "Event",
                "arguments": [{"entity_id": "departure", "role": "event_id"}],
                "confidence": 0.95,
                "evidence": _evidence("left"),
            },
            {
                "predicate": "Agent",
                "arguments": [
                    {"entity_id": "departure", "role": "action"},
                    {"entity_id": "mira", "role": "actor"},
                ],
                "confidence": 0.9,
                "evidence": _evidence("Mira left"),
            },
        ],
    }

    result = _parse(payload, source)

    event, agent = result.extraction.assertions
    assert [argument.entity_id for argument in event.arguments] == ["departure"]
    assert [argument.entity_id for argument in agent.arguments] == [
        "departure",
        "mira",
    ]


def test_worked_examples_are_valid_parse_results() -> None:
    """Every worked example passes the parser unchanged, with no repairs."""
    examples_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "parser_config"
        / "catalog_document_examples.jsonl"
    )
    catalog = load_ontology_catalog()
    examples = examples_path.read_text(encoding="utf-8").splitlines()
    assert examples, "The prompt must contain worked examples"
    for index, line in enumerate(examples, start=1):
        example = json.loads(line)
        parser = ReferenceSemanticParser(
            backend=RecordingBackend(
                {
                    "entities": example["entities"],
                    "assertions": example["assertions"],
                    "rules": example.get("rules", []),
                }
            ),
            config=SemanticParserConfig(model_name="curated-example"),
            catalog=catalog,
        )
        result = parser.parse_document(
            SourceDocument(
                document_id=f"example-{index}",
                corpus_id="catalog-examples",
                source_text=example["source_text"],
            )
        )
        assert result.provenance.repairs == [], f"example {index} needed repairs"
        assert result.extraction == SemanticExtraction.model_validate(
            {key: example.get(key, []) for key in ("entities", "assertions", "rules")}
        ), f"example {index} is not in normalized form"


def test_parser_runs_without_hybrid_miner() -> None:
    """The parser never imports hybrid-miner; only symbolic_pipeline does.

    A fresh interpreter blocks every hybrid_miner import, imports each parser
    module, and loads the catalog the way the command line does.
    """
    script = textwrap.dedent(
        """
        import argparse
        import importlib
        import pkgutil
        import sys

        sys.modules["hybrid_miner"] = None
        import parser

        names = [m.name for m in pkgutil.walk_packages(parser.__path__, "parser.")]
        names += ["parser.student.student_prompt", "parser.student.train"]
        for name in names:
            importlib.import_module(name)
        from parser.semantic.cli import command_catalog

        catalog = command_catalog(
            argparse.Namespace(ontology_modules=None, ontology_bundle=None)
        )
        assert "LocatedIn" in catalog.relations
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_rejects_unknown_predicate() -> None:
    source, payload = _concession()
    payload["assertions"][0]["predicate"] = "InventedRelation"

    with pytest.raises(SemanticParseError, match="Unknown semantic predicate"):
        _parse(payload, source)


def test_accepts_evaluation_fallback_for_an_uncataloged_relation() -> None:
    source, payload = _concession()
    payload["assertions"][0].update(
        predicate="Evaluation", relation="Unclassified Contrast", fallback=True
    )

    result = _parse(payload, source)

    assert result.extraction.assertions[0].relation == "unclassified_contrast"


def test_rejects_confidence_outside_catalog_range() -> None:
    source, payload = _concession()
    payload["assertions"][0]["confidence"] = 0.3
    catalog = load_ontology_catalog()
    narrowed = replace(
        catalog,
        relations={
            **catalog.relations,
            "Concession": replace(
                catalog.relations["Concession"], confidence_range=(0.4, 1.0)
            ),
        },
    )

    with pytest.raises(SemanticParseError, match="outside the catalog range"):
        _parse(payload, source, catalog=narrowed)


def test_rejects_wrong_argument_roles() -> None:
    source, payload = _concession()
    payload["assertions"][0]["arguments"].reverse()

    with pytest.raises(SemanticParseError, match="requires argument roles"):
        _parse(payload, source)


def test_rejects_undeclared_entity_type() -> None:
    source, payload = _concession()
    payload["entities"][0]["entity_type"] = "Artifact"

    with pytest.raises(SemanticParseError, match="unknown type 'Artifact'"):
        _parse(payload, source)


def test_rejects_evidence_not_present_in_source() -> None:
    source, payload = _concession()
    payload["assertions"][0]["evidence"] = _evidence("Not in source")

    with pytest.raises(SemanticParseError, match="is not in the source text"):
        _parse(payload, source)


def test_specific_rule_keeps_its_parts_hypothetical() -> None:
    source = "If it rains, Mira may leave."
    payload = {
        "entities": [
            {"id": "mira", "entity_type": "Entity", "mention": {"text": "Mira"}}
        ],
        "rules": [
            {
                "scope": "specific",
                "antecedents": [
                    {
                        "predicate": "Evaluation",
                        "relation": "rain",
                        "fallback": True,
                        "factuality": "hypothetical",
                        "confidence": 0.9,
                        "evidence": _evidence("it rains"),
                    }
                ],
                "consequents": [
                    {
                        "predicate": "Evaluation",
                        "relation": "leave",
                        "fallback": True,
                        "arguments": [{"entity_id": "mira", "role": "agent"}],
                        "factuality": "hypothetical",
                        "modality": "permission",
                        "confidence": 0.9,
                        "evidence": _evidence("Mira may leave"),
                    }
                ],
                "evidence": _evidence(source[:-1]),
            }
        ],
    }

    result = _parse(payload, source)

    (rule,) = result.extraction.rules
    assert result.extraction.assertions == []
    assert {part.factuality for part in [*rule.antecedents, *rule.consequents]} == {
        "hypothetical"
    }
    assert result.locate(rule.consequents[0].evidence[0])[0] == source.index("Mira")


def test_rejects_an_asserted_part_inside_a_rule() -> None:
    source = "If it rains, Mira may leave."
    payload = {
        "rules": [
            {
                "scope": "specific",
                "antecedents": [
                    {
                        "predicate": "Evaluation",
                        "relation": "rain",
                        "fallback": True,
                        "factuality": "asserted",
                        "confidence": 0.9,
                        "evidence": _evidence("it rains"),
                    }
                ],
                "consequents": [
                    {
                        "predicate": "Evaluation",
                        "relation": "leave",
                        "fallback": True,
                        "factuality": "hypothetical",
                        "confidence": 0.9,
                        "evidence": _evidence("Mira may leave"),
                    }
                ],
                "evidence": _evidence(source[:-1]),
            }
        ],
    }

    with pytest.raises(SemanticParseError, match="hypothetical factuality"):
        _parse(payload, source)


def test_records_repairs_in_provenance() -> None:
    source, payload = _concession()
    payload["assertions"][0].update(relation="concede", fallback=True)

    result = _parse(payload, source)

    assert [repair.kind for repair in result.provenance.repairs] == [
        "predicate_contract"
    ]
    assert result.extraction.assertions[0].relation is None
    assert result.extraction.assertions[0].fallback is False


def test_required_schema_decoding_rejects_an_unsupported_backend_before_generation() -> (
    None
):
    backend = RecordingBackend({"entities": [], "assertions": []})
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        require_structured_output=True,
    )
    with pytest.raises(ValueError, match="does not support JSON-schema decoding"):
        parser.parse_document(
            SourceDocument(document_id="doc", corpus_id="corpus", source_text="Hi.")
        )
    assert not backend.calls


def test_schema_decoding_requests_the_extraction_schema() -> None:
    source, payload = _concession()
    backend = SchemaBackend(payload)
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        require_structured_output=True,
    )

    parser.parse_document(
        SourceDocument(document_id="doc-1", corpus_id="corpus-1", source_text=source)
    )

    assert backend.schemas == [SemanticExtraction.model_json_schema()]
