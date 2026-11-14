"""Tests for general ontology-guided prompt construction."""

import pytest
import yaml

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import catalog_prompt
from parser.semantic.backends import CallableBackend
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParserConfig,
)


def test_prompt_uses_general_catalog_signatures_and_examples():
    prompt = build_catalog_document_prompt(
        "Nora sang.",
        catalog=load_ontology_catalog(),
    )

    assert prompt.version.startswith("general-v1+sha256.")
    assert len(prompt.version.split("+sha256.")[1]) == 64

    # General ontology versions are included.
    assert "upper-v1" in prompt.text
    assert "general-v1" in prompt.text

    # General ontology predicates are available.
    assert "LocatedIn(" in prompt.text
    assert "Causes(" in prompt.text
    assert "Evaluation" in prompt.text

    # General extraction instructions are included.
    assert "semantic roles" in prompt.text
    assert "agentless passive" in prompt.text
    assert "factuality" in prompt.text
    assert "hypothetical" in prompt.text

    # The supplied source is included.
    assert "Source text:\nNora sang." in prompt.text


def test_routed_prompt_keeps_input_heading_and_source_once(monkeypatch):
    source = "Although it rained, Mira left."
    build_general = catalog_prompt.build_catalog_document_prompt

    def legacy_heading(*args, **kwargs):
        general = build_general(*args, **kwargs)
        return catalog_prompt.CatalogDocumentPrompt(
            version=general.version,
            text=general.text.replace("## 14. INPUT", "## 13. INPUT"),
        )

    monkeypatch.setattr(catalog_prompt, "build_catalog_document_prompt", legacy_heading)
    prompt = catalog_prompt.build_routed_extraction_prompt(
        source, catalog=load_ontology_catalog(), run_discourse=True
    )
    headings = [
        line.strip()
        for line in prompt.text.splitlines()
        if line.lstrip().startswith("## ") and line.strip().endswith(". INPUT")
    ]
    assert len(headings) == 1
    assert prompt.text.count(f"Source text:\n{source}") == 1
    assert prompt.text.index(headings[0]) < prompt.text.index(f"Source text:\n{source}")


def test_prompt_excludes_discourse_predicate_signatures():
    prompt = build_catalog_document_prompt(
        "Although it rained, Mira left.",
        catalog=load_ontology_catalog(),
    )

    # Discourse ontology version is not advertised as a general module.
    assert "discourse-v1" not in prompt.text

    # Discourse predicates are not injected into the general catalog guide.
    assert "Concession(statement:" not in prompt.text
    assert "Contrast(item1:" not in prompt.text
    assert "EvokesExpectation(trigger:" not in prompt.text
    assert "Defeats(defeating_outcome:" not in prompt.text
    assert "DiscourseFocus(focal_item:" not in prompt.text


def test_prompt_preserves_source_context_and_current_contract():
    source = "  The door was opened by Alice.\nCafé ☕ remained open.  "
    context = "Alice is the caretaker."

    prompt = build_catalog_document_prompt(
        source,
        context=context,
        catalog=load_ontology_catalog(),
    )

    extracted_source = prompt.text.rsplit("Source text:\n", 1)[1]

    assert extracted_source == source

    assert f"Background context:\n{context}" in prompt.text
    # General semantic guidance remains present.
    assert "semantic roles" in prompt.text
    assert "agentless passive" in prompt.text
    assert "Evaluation" in prompt.text
    assert "factuality" in prompt.text
    assert "hypothetical" in prompt.text

    # Current output contract.
    assert '"entities": []' in prompt.text
    assert '"assertions": []' in prompt.text
    assert '"rules": []' in prompt.text

    # Evidence and source-grounding requirements.
    assert "source evidence" in prompt.text
    assert "original source" in prompt.text


def test_prompt_uses_only_selected_general_ontology_modules():
    catalog = load_ontology_catalog(modules=("upper", "discourse"))

    prompt = build_catalog_document_prompt(
        "Mira left despite the rain.",
        catalog=catalog,
    )

    assert "upper-v1" in prompt.text
    assert "discourse-v1" not in prompt.text

    # General predicates from upper remain available.
    assert "Evaluation" in prompt.text
    assert "Entity(" in prompt.text

    # Discourse signatures are excluded.
    assert "Concession(statement:" not in prompt.text


def test_prompt_keeps_general_guidance_without_discourse_examples():
    prompt = build_catalog_document_prompt(
        "Mira left.",
        catalog=load_ontology_catalog(modules=("upper",)),
    )

    # General semantic extraction remains available.
    assert "Evaluation" in prompt.text
    assert "Entity(" in prompt.text

    # The current source is included.
    assert "Source text:\nMira left." in prompt.text

    # Discourse-specific predicates are excluded.
    assert "Concession(statement:" not in prompt.text
    assert "EvokesExpectation(trigger:" not in prompt.text
    assert "Defeats(defeating_outcome:" not in prompt.text


def test_prompt_rejects_empty_source_text():
    with pytest.raises(ValueError, match="source_text"):
        build_catalog_document_prompt(
            " \t ",
            catalog=load_ontology_catalog(),
        )


def test_prompt_rejects_invalid_context_type():
    with pytest.raises(ValueError, match="context"):
        build_catalog_document_prompt(
            "Nora left.",
            context=None,
            catalog=load_ontology_catalog(),
        )


def test_prompt_identity_excludes_source_text_and_reference_context():
    catalog = load_ontology_catalog()

    first = build_catalog_document_prompt(
        "Nora left.",
        catalog=catalog,
    )

    second = build_catalog_document_prompt(
        "Ivo arrived.",
        context="Nora arrived yesterday.",
        catalog=catalog,
    )

    assert first.version == second.version
    assert first.text != second.text


def test_prompt_identity_tracks_ontology_version_changes(monkeypatch):
    catalog = load_ontology_catalog()

    first = build_catalog_document_prompt(
        "Nora left.",
        catalog=catalog,
    )

    original = catalog_prompt._general_catalog_versions

    monkeypatch.setattr(
        catalog_prompt,
        "_general_catalog_versions",
        lambda selected: original(selected) + ("test-version",),
    )

    second = build_catalog_document_prompt(
        "Nora left.",
        catalog=catalog,
    )

    assert first.version != second.version


def test_prompt_identity_tracks_changes_to_system_prompt(
    tmp_path,
    monkeypatch,
):
    catalog = load_ontology_catalog()

    first = build_catalog_document_prompt(
        "Nora left.",
        catalog=catalog,
    )

    config = yaml.safe_load(
        catalog_prompt._GENERAL_PROMPT_PATH.read_text(encoding="utf-8")
    )

    config["system_prompt"] += "\nAdditional test instruction."

    path = tmp_path / "general_prompt.yaml"

    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        catalog_prompt,
        "_GENERAL_PROMPT_PATH",
        path,
    )

    second = build_catalog_document_prompt(
        "Nora left.",
        catalog=catalog,
    )

    assert first.version != second.version


def test_parser_exposes_active_prompt_identity_without_generation():
    catalog = load_ontology_catalog(modules=("upper",))

    parser = ReferenceSemanticParser(
        backend=CallableBackend(
            lambda **_: pytest.fail("Prompt identity lookup must not generate"),
        ),
        config=SemanticParserConfig(
            model_name="identity-test",
            enable_routing=False,
        ),
        catalog=catalog,
    )

    expected = build_catalog_document_prompt(
        "Any source.",
        catalog=catalog,
    ).version

    assert parser.prompt_version == expected


def test_routed_prompt_finds_numbered_input_section():
    from parser.semantic.catalog_prompt import (
        build_routed_extraction_prompt,
    )

    prompt = build_routed_extraction_prompt(
        "Although it rained, Mira left.",
        catalog=load_ontology_catalog(),
        run_discourse=True,
    )

    assert "FINAL COMBINED EXTRACTION REQUIREMENTS" in prompt.text
    assert "## 14. INPUT" in prompt.text
    assert prompt.text.count("Source text:\nAlthough it rained, Mira left.") == 1
