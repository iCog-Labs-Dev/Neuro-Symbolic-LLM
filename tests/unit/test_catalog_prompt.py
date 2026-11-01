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

    assert prompt.version.startswith("general-v3+sha256.")
    assert len(prompt.version.split("+sha256.")[1]) == 64

    # General ontology versions are included.
    assert "upper-v1" in prompt.text
    assert "general-v1" in prompt.text

    # General ontology predicates are available.
    assert "LocatedIn(" in prompt.text
    assert "Causes(" in prompt.text
    assert "Evaluation" in prompt.text

    # General worked examples are embedded in the prompt.
    assert "Nora sang." in prompt.text
    assert "Nora did not open the gate." in prompt.text
    assert "The parcel was carried by Ivo." in prompt.text
    assert "If it rains, Nora will leave." in prompt.text
    assert "All red things are nice." in prompt.text


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

    extracted_source = prompt.text.split("<source_text>\n", 1)[1].split(
        "\n</source_text>", 1
    )[0]

    assert extracted_source == source

    assert f"<reference_context>\n{context}\n</reference_context>" in prompt.text

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


def test_prompt_keeps_general_examples_without_external_jsonl():
    prompt = build_catalog_document_prompt(
        "Mira left.",
        catalog=load_ontology_catalog(modules=("upper",)),
    )

    assert "Nora sang." in prompt.text
    assert "Nora did not open the gate." in prompt.text
    assert "If it rains, Nora will leave." in prompt.text
    assert "All red things are nice." in prompt.text

    # Old discourse worked example is not part of general examples.
    assert "Although it snowed, Nora sang." not in prompt.text


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
