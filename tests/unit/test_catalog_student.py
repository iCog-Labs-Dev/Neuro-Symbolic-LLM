"""Validation tests for semantic extraction student pairs and checkpoints."""

import json
from pathlib import Path
from typing import Any

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.catalog_student import (
    validate_catalog_student_checkpoint,
    validate_catalog_student_pair,
)
from parser.semantic.schema import SEMANTIC_CONTRACT_VERSION
from tests.semantic_fixtures import pair as canonical_pair


def _pair(text: str = "Mira left.") -> dict[str, Any]:
    """Create a valid current-contract pair using the offline parser fixture."""
    pair: dict[str, Any] = canonical_pair(text)
    return pair


def test_pair_rebuilds_the_current_extraction_target() -> None:
    pair = _pair()

    prompt, label, result = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog()
    )

    assert "<source_text>\nMira left.\n</source_text>" in prompt
    assert json.loads(label) == result.extraction.model_dump(mode="json")
    assert result.document.source_text == "Mira left."
    assert result.provenance.provider == "fixture"


def test_pair_rejects_stale_prompt_provenance() -> None:
    pair = _pair()
    pair["parser"]["prompt_version"] = "catalog-document-v2"

    with pytest.raises(ValueError, match="prompt version"):
        validate_catalog_student_pair(pair, catalog=load_ontology_catalog())


def test_pair_rejects_mismatched_ontology_versions() -> None:
    pair = _pair()
    pair["parser"]["ontology_versions"] = ["old-ontology-v1"]

    with pytest.raises(ValueError, match="ontology versions"):
        validate_catalog_student_pair(pair, catalog=load_ontology_catalog())


def test_pair_normalizes_omitted_optional_target_fields() -> None:
    pair = _pair()
    target = json.loads(pair["label"])
    del target["assertions"][0]["modality"]
    pair["label"] = json.dumps(target)

    _, canonical_label, result = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog()
    )

    assert result.extraction.assertions[0].modality == "none"
    assert json.loads(canonical_label)["assertions"][0]["modality"] == "none"


def test_pair_preserves_non_asserted_factuality() -> None:
    pair = _pair()
    target = json.loads(pair["label"])
    target["assertions"][0]["factuality"] = "speculative"
    pair["label"] = json.dumps(target)

    _, _, result = validate_catalog_student_pair(pair, catalog=load_ontology_catalog())

    assert result.extraction.assertions[0].factuality == "speculative"


def test_pair_preserves_original_parser_provenance() -> None:
    pair = _pair()
    pair["parser"]["parser_version"] = "catalog-semantic-parser-v3"

    _, _, result = validate_catalog_student_pair(pair, catalog=load_ontology_catalog())

    assert result.provenance.parser_version == "catalog-semantic-parser-v3"


def test_pair_rejects_changed_guidance_with_stale_recorded_version(monkeypatch) -> None:
    from parser.semantic import catalog_prompt

    pair = _pair()
    original = catalog_prompt.load_shared_extraction_guidance
    monkeypatch.setattr(
        catalog_prompt,
        "load_shared_extraction_guidance",
        lambda: original() + "\nChanged role guidance.",
    )

    with pytest.raises(ValueError, match="prompt version"):
        validate_catalog_student_pair(pair, catalog=load_ontology_catalog())


def test_checkpoint_requires_matching_contract_prompt_and_ontology(
    tmp_path: Path,
) -> None:
    catalog = load_ontology_catalog()
    metadata_path = tmp_path / "training_metadata.json"
    metadata = {
        "target_contract_version": SEMANTIC_CONTRACT_VERSION,
        "prompt_version": build_catalog_document_prompt(
            "Checkpoint compatibility.", catalog=catalog
        ).version,
        "ontology_versions": list(catalog.versions),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    validate_catalog_student_checkpoint(tmp_path, catalog=catalog)

    for key, value, expected_error in (
        ("target_contract_version", "old-contract", "target contract"),
        ("prompt_version", "catalog-document-v2", "prompt version"),
        ("ontology_versions", ["old-ontology-v1"], "ontology versions"),
    ):
        metadata[key] = value
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match=expected_error):
            validate_catalog_student_checkpoint(tmp_path, catalog=catalog)
        metadata[key] = {
            "target_contract_version": SEMANTIC_CONTRACT_VERSION,
            "prompt_version": build_catalog_document_prompt(
                "Checkpoint compatibility.", catalog=catalog
            ).version,
            "ontology_versions": list(catalog.versions),
        }[key]
