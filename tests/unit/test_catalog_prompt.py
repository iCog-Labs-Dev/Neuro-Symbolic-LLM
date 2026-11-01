"""Tests for catalog-guided semantic-extraction prompt construction."""

import json
import re

import pytest
import yaml

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import catalog_prompt
from parser.semantic.backends import CallableBackend
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticExtraction,
    SemanticParseProvenance,
    SemanticParseResult,
    SourceDocument,
)
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParserConfig,
    validate_against_catalog,
)


def test_prompt_uses_catalog_signatures_versions_and_current_examples() -> None:
    prompt = build_catalog_document_prompt(
        "Although it rained, Mira left.", catalog=load_ontology_catalog()
    )

    assert prompt.version.startswith("catalog-document-v15+sha256.")
    assert len(prompt.version.split("+sha256.")[1]) == 64
    assert "general-v1" in prompt.text
    assert "upper-v1" in prompt.text
    assert "Concession(statement: Clause, counter_statement: Clause)" in prompt.text
    assert "Causes(cause: Event|Proposition, effect: Event|Proposition)" in prompt.text
    assert "Near(entity: Entity, nearby_entity: Entity)" in prompt.text
    assert "Nora sang." in prompt.text
    assert "Although it snowed, Nora sang." in prompt.text
    assert "If it rains, Nora will leave." in prompt.text


def test_prompt_preserves_source_context_and_current_contract_guidance() -> None:
    source = "  The door was opened by Alice.\nCafé ☕ remained open.  "
    context = "Alice is the caretaker."

    prompt = build_catalog_document_prompt(
        source, context=context, catalog=load_ontology_catalog()
    )

    assert (
        prompt.text.split("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]
        == source
    )
    assert f"<reference_context>\n{context}\n</reference_context>" in prompt.text
    assert "Active and\n  passive descriptions" in prompt.text
    assert "Do not invent an agent for an agentless passive" in prompt.text
    assert "Return exactly the SemanticExtraction object" in prompt.text
    assert "Evidence is a nonempty list of exact source quotes" in prompt.text
    assert "Every entity must have a source-grounded mention" in prompt.text


def test_prompt_uses_only_selected_ontology_modules() -> None:
    catalog = load_ontology_catalog(modules=("upper", "discourse"))
    prompt = build_catalog_document_prompt(
        "Mira left despite the rain.", catalog=catalog
    )

    assert "Ontology versions: discourse-v2, upper-v1" in prompt.text
    assert "Concession(statement: Clause, counter_statement: Clause)" in prompt.text
    assert "Kick(" not in prompt.text
    assert "lexical-v1" not in prompt.text


def test_prompt_excludes_examples_requiring_unselected_modules() -> None:
    prompt = build_catalog_document_prompt(
        "Mira left.", catalog=load_ontology_catalog(modules=("upper",))
    )

    assert "Nora sang." in prompt.text
    assert "Although it snowed, Nora sang." not in prompt.text
    assert "Concession(" not in prompt.text


def test_prompt_rejects_empty_source_text() -> None:
    with pytest.raises(ValueError, match="source_text"):
        build_catalog_document_prompt(" \t ", catalog=load_ontology_catalog())


def test_prompt_identity_excludes_source_text_and_reference_context() -> None:
    catalog = load_ontology_catalog()
    first = build_catalog_document_prompt("Nora left.", catalog=catalog)
    second = build_catalog_document_prompt(
        "Ivo arrived.", context="Nora arrived yesterday.", catalog=catalog
    )

    assert first.version == second.version
    assert first.text != second.text


def test_prompt_identity_tracks_all_guidance_components(monkeypatch) -> None:
    catalog = load_ontology_catalog()
    components = (
        "_catalog_guide",
        "load_shared_extraction_guidance",
        "_worked_examples",
    )

    for component in components:
        first = build_catalog_document_prompt("Nora left.", catalog=catalog)
        original = getattr(catalog_prompt, component)
        monkeypatch.setattr(
            catalog_prompt,
            component,
            lambda *args, _original=original, **kwargs: _original(*args, **kwargs)
            + "\nChanged guidance.",
        )
        second = build_catalog_document_prompt("Nora left.", catalog=catalog)
        assert second.version != first.version
        monkeypatch.setattr(catalog_prompt, component, original)


def test_prompt_identity_tracks_changes_to_system_prompt(tmp_path, monkeypatch) -> None:
    catalog = load_ontology_catalog()
    first = build_catalog_document_prompt("Nora left.", catalog=catalog)
    config = yaml.safe_load(
        catalog_prompt._PROMPT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    config["system_prompt"] += "\nChanged instruction."
    path = tmp_path / "catalog_document_prompt.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(catalog_prompt, "_PROMPT_CONFIG_PATH", path)

    second = build_catalog_document_prompt("Nora left.", catalog=catalog)

    assert second.version != first.version


def test_every_worked_example_matches_schema_catalog_and_source_quotes() -> None:
    catalog = load_ontology_catalog()
    prompt = build_catalog_document_prompt("The door opened.", catalog=catalog)
    example_outputs = re.findall(
        r"<example_output_json>\n(.*?)\n</example_output_json>", prompt.text, re.S
    )
    sources = re.findall(
        r"<example_source_text>\n(.*?)\n</example_source_text>", prompt.text, re.S
    )

    assert example_outputs
    for index, (source, output) in enumerate(
        zip(sources, example_outputs, strict=True)
    ):
        payload = json.loads(output)
        extraction = SemanticExtraction.model_validate(payload)
        result = SemanticParseResult(
            document=SourceDocument(
                document_id=f"example-{index}",
                corpus_id="prompt-examples",
                source_text=source,
            ),
            provenance=SemanticParseProvenance(
                provider="test",
                model="fixture",
                parser_version="test-v1",
                prompt_version=prompt.version,
                ontology_versions=list(catalog.versions),
            ),
            extraction=extraction,
        )
        assert result.contract_version == SEMANTIC_CONTRACT_VERSION
        validate_against_catalog(result, catalog)


def test_parser_exposes_active_prompt_identity_without_generation() -> None:
    catalog = load_ontology_catalog(modules=("upper",))
    parser = ReferenceSemanticParser(
        backend=CallableBackend(
            lambda **_: pytest.fail("Prompt identity lookup must not generate"),
        ),
        config=SemanticParserConfig(model_name="identity-test"),
        catalog=catalog,
    )
    expected = build_catalog_document_prompt("Any source.", catalog=catalog).version

    assert parser.prompt_version == expected
