"""Validate and prompt student pairs for the v2 extraction target.

A pair holds the source text and, as its label, the teacher's final
SemanticExtraction: what the parser kept after normalization, repairs, and
validation. The student learns to write that extraction directly; provenance
stays outside the label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticExtraction,
    SemanticParseProvenance,
    SemanticParseResult,
    SourceDocument,
)

_PAIR_FIELDS = frozenset(
    {"contract_version", "document_id", "corpus_id", "input", "parser", "label"}
)
_SOURCE_METADATA = frozenset({"source_dataset", "source_domain", "reference_context"})


def extraction_label(result: SemanticParseResult) -> str:
    """Return the training label: the extraction as JSON, in schema field order."""
    return result.extraction.model_dump_json()


def make_student_pair(
    result: SemanticParseResult,
    *,
    source_dataset: str | None = None,
    source_domain: str | None = None,
) -> dict[str, Any]:
    """Create a supervised source-to-extraction pair from one parse result."""
    document = result.document
    pair: dict[str, Any] = {
        "contract_version": result.contract_version,
        "document_id": document.document_id,
        "corpus_id": document.corpus_id,
        "input": document.source_text,
        "parser": result.provenance.model_dump(mode="json"),
        "label": extraction_label(result),
    }
    for field, value in (
        ("source_dataset", source_dataset),
        ("source_domain", source_domain),
        ("reference_context", document.context or None),
    ):
        if value is not None:
            pair[field] = value
    return pair


def validate_catalog_student_pair(
    pair: object, *, catalog: OntologyCatalog
) -> tuple[str, str, SemanticParseResult]:
    """Check the label against the same contract used before teacher export.

    The full prompt is regenerated from the exact source and ontology versions.
    A pair from an older prompt or bundle cannot silently enter a new run.
    Returns the prompt text, the label, and the rebuilt parse result.
    """
    from parser.semantic.semantic_parser import validate_against_catalog

    if (
        not isinstance(pair, dict)
        or not set(pair) >= _PAIR_FIELDS
        or set(pair) - _PAIR_FIELDS - _SOURCE_METADATA - {"confidence"}
    ):
        raise ValueError("Catalog student pair has missing or unexpected fields")
    for field in _SOURCE_METADATA & pair.keys():
        if not isinstance(pair[field], str) or not pair[field].strip():
            raise ValueError(f"{field} must be nonempty text")
    if "confidence" in pair:
        confidence = pair["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Pair confidence must be a number in [0, 1]")
    if pair["contract_version"] != SEMANTIC_CONTRACT_VERSION:
        raise ValueError("Unsupported catalog student contract version")
    source_text = pair["input"]
    label = pair["label"]
    provenance = pair["parser"]
    if not isinstance(source_text, str) or not isinstance(label, str):
        raise ValueError("Catalog student input and label must be strings")
    if not isinstance(provenance, dict):
        raise ValueError("Catalog student parser provenance must be an object")
    context = pair.get("reference_context", "")
    prompt = build_catalog_document_prompt(
        source_text, catalog=catalog, context=context
    )
    if provenance.get("prompt_version") != prompt.version:
        raise ValueError(
            "Catalog student prompt version does not match the active prompt"
        )
    if provenance.get("ontology_versions") != list(catalog.versions):
        raise ValueError("Catalog student ontology versions do not match the bundle")
    result = SemanticParseResult(
        document=SourceDocument(
            document_id=pair["document_id"],
            corpus_id=pair["corpus_id"],
            source_text=source_text,
            context=context,
        ),
        provenance=SemanticParseProvenance.model_validate(provenance),
        extraction=SemanticExtraction.model_validate_json(label),
    )
    validate_against_catalog(result, catalog)
    return prompt.text, extraction_label(result), result


def validate_catalog_student_checkpoint(
    model_directory: str | Path, *, catalog: OntologyCatalog
) -> None:
    """Reject a student checkpoint trained for another target or bundle."""

    path = Path(model_directory) / "training_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read student training metadata at {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Student training metadata must be an object")
    prompt_version = build_catalog_document_prompt(
        "Checkpoint compatibility.", catalog=catalog
    ).version
    if metadata.get("target_contract_version") != SEMANTIC_CONTRACT_VERSION:
        raise ValueError("Student checkpoint has the wrong target contract")
    if metadata.get("prompt_version") != prompt_version:
        raise ValueError("Student checkpoint has the wrong prompt version")
    if metadata.get("ontology_versions") != list(catalog.versions):
        raise ValueError("Student checkpoint has the wrong ontology versions")
