"""Validate and prompt student pairs for the v2 extraction target.

A pair contains source text and the teacher's final validated
SemanticExtraction as its label.

The student learns to generate the combined general and discourse
extraction directly in one generation.

Teacher provenance describes how the training label was produced.
Student prompt identity is tracked independently through training
metadata and checkpoint compatibility checks.

Document metadata and provenance remain outside the student label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticExtraction,
    SemanticParseProvenance,
    SemanticParseResult,
    SourceDocument,
)
from parser.student.student_prompt import build_student_prompt_spec

_PAIR_FIELDS = frozenset(
    {
        "contract_version",
        "document_id",
        "corpus_id",
        "input",
        "parser",
        "label",
    }
)

_SOURCE_METADATA = frozenset(
    {
        "source_dataset",
        "source_domain",
        "reference_context",
    }
)


def extraction_label(result: SemanticParseResult) -> str:
    """Serialize the student's extraction target.

    The student predicts SemanticExtraction only, not the complete
    SemanticParseResult.

    The schema determines field ordering and serialization.
    """
    return result.extraction.model_dump_json()


def make_student_pair(
    result: SemanticParseResult,
    *,
    source_dataset: str | None = None,
    source_domain: str | None = None,
) -> dict[str, Any]:
    """Create a supervised pair from a validated teacher parse result.

    The original teacher provenance is retained. It must not be
    overwritten with the student's prompt version.
    """
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
    pair: object,
    *,
    catalog: OntologyCatalog,
) -> tuple[str, str, SemanticParseResult]:
    """Validate a student pair and build its combined extraction prompt.

    Checks:
    - Pair structure and metadata
    - Semantic contract version
    - Teacher provenance
    - Ontology compatibility
    - SemanticExtraction schema
    - Source-grounded evidence
    - Predicate and argument constraints

    The teacher's prompt version is not compared with the student's
    prompt version because they use different generation workflows.

    Returns:
        prompt_text:
            The combined general/discourse student prompt.

        json_label:
            The validated SemanticExtraction JSON label.

        result:
            The reconstructed SemanticParseResult.
    """
    from parser.semantic.semantic_parser import validate_against_catalog

    if not isinstance(pair, dict):
        raise ValueError("Catalog student pair must be a JSON object")

    pair_fields = set(pair)

    if (
        not pair_fields >= _PAIR_FIELDS
        or pair_fields - _PAIR_FIELDS - _SOURCE_METADATA - {"confidence"}
    ):
        raise ValueError("Catalog student pair has missing or unexpected fields")

    for field in _SOURCE_METADATA & pair_fields:
        value = pair[field]

        if not isinstance(value, str) or not value.strip():
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

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("Catalog student input must be nonempty text")

    if not isinstance(label, str):
        raise ValueError("Catalog student label must be a JSON string")

    if not isinstance(provenance, dict):
        raise ValueError("Catalog student parser provenance must be an object")

    context = pair.get("reference_context", "")

    # Build the student-specific combined prompt.
    # This must be the same builder used during student inference.
    prompt = build_student_prompt_spec(
        source_text,
        context=context,
        catalog=catalog,
    )

    # Teacher provenance is preserved independently.
    # Do not compare it with the student's prompt version.
    if not provenance.get("prompt_version"):
        raise ValueError("Student pair is missing teacher prompt provenance")

    if provenance.get("ontology_versions") != list(catalog.versions):
        raise ValueError("Catalog student ontology versions do not match the bundle")

    # Reconstruct the original validated extraction record.
    # Pydantic also validates source quotations against the document.
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

    # Validate ontology predicates, argument signatures and types.
    validate_against_catalog(result, catalog)

    return (
        prompt.text,
        extraction_label(result),
        result,
    )


def validate_catalog_student_checkpoint(
    model_directory: str | Path,
    *,
    catalog: OntologyCatalog,
) -> None:
    """Validate a trained student checkpoint against the active contract.

    The checkpoint must use:
    - The active SemanticExtraction contract
    - The current combined student prompt version
    - The active ontology bundle versions

    Teacher prompt versions are not checkpoint compatibility keys.
    """
    path = Path(model_directory) / "training_metadata.json"

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read student training metadata at {path}") from error

    if not isinstance(metadata, dict):
        raise ValueError("Student training metadata must be an object")

    # The student prompt version is derived from the same combined
    # prompt builder used for dataset construction and inference.
    prompt_version = build_student_prompt_spec(
        "Checkpoint compatibility.",
        catalog=catalog,
    ).version

    if metadata.get("target_contract_version") != SEMANTIC_CONTRACT_VERSION:
        raise ValueError("Student checkpoint has the wrong target contract")

    if metadata.get("prompt_version") != prompt_version:
        raise ValueError("Student checkpoint has the wrong prompt version")

    if metadata.get("ontology_versions") != list(catalog.versions):
        raise ValueError("Student checkpoint has the wrong ontology versions")
