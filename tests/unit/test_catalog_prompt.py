"""Tests for catalog-guided semantic-document prompt construction."""

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt


def test_prompt_uses_catalog_signatures_and_versions() -> None:
    prompt = build_catalog_document_prompt(
        "Although it rained, Mira left.", catalog=load_ontology_catalog()
    )

    assert prompt.version == "catalog-document-v1"
    assert "discourse-v1" in prompt.text
    assert "upper-v1" in prompt.text
    assert "Concession(statement: Clause, counter_statement: Clause)" in prompt.text
    assert "Causes(cause: Event|Proposition, effect: Event|Proposition)" in prompt.text
    assert "confidence 0.6..1" in prompt.text
    assert "AliasOf(" not in prompt.text


def test_prompt_preserves_source_text_and_semantic_rules() -> None:
    source = "  The door was opened by Alice.\nCafé ☕ remained open.  "

    prompt = build_catalog_document_prompt(source, catalog=load_ontology_catalog())

    assert (
        prompt.text.split("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]
        == source
    )
    assert "Active and\n  passive descriptions" in prompt.text
    assert "Do not invent an agent for an agentless passive" in prompt.text
    assert "ProvisionalRelation" in prompt.text
    assert "legacy Evaluation fallback" in prompt.text
    assert "half-open Unicode code-point offsets" in prompt.text
    assert "extracted-hypothesis" in prompt.text


def test_prompt_rejects_empty_source_text() -> None:
    with pytest.raises(ValueError, match="source_text"):
        build_catalog_document_prompt(" \t ", catalog=load_ontology_catalog())
