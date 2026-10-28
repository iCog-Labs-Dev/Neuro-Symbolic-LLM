"""Build supporting text contexts from hybrid-miner source records.

Supported input contracts:
    - pattern-record-v1
    - semantic-document-v1

Source spans use half-open Unicode code-point offsets:
    source_text[start:end]

"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemplateContexts:
    """Supporting text contexts associated with one mined pattern."""

    pattern_id: str
    contexts: tuple[str, ...]


def prepare_contexts(
    pattern_id: str,
    contexts: Iterable[str],
) -> TemplateContexts:
    """Validate supporting contexts for one pattern.

    Duplicate text is retained because different source spans may contain
    identical text.
    """
    if not isinstance(pattern_id, str) or not pattern_id.strip():
        raise ValueError("pattern_id must be a nonempty string")

    items = tuple(contexts)

    if not items:
        raise ValueError("contexts must not be empty")

    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError("contexts must contain nonempty strings")

    return TemplateContexts(
        pattern_id=pattern_id,
        contexts=items,
    )


def load_semantic_documents(
    path: str | Path,
) -> dict[tuple[str, str], str]:
    """Load source text from hybrid-miner SemanticDocument JSONL.

    Documents are identified by:
        (corpus_id, document_id)

    The original source_text is preserved without normalization because
    source offsets refer to this exact string.
    """
    documents: dict[tuple[str, str], str] = {}

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Document line {line_number} must contain a JSON object"
                )

            if record.get("contract_version") != "semantic-document-v1":
                raise ValueError(
                    f"Document line {line_number}: "
                    "unsupported SemanticDocument contract version"
                )

            corpus_id = record.get("corpus_id")
            document_id = record.get("document_id")
            source_text = record.get("source_text")

            if not isinstance(corpus_id, str) or not corpus_id.strip():
                raise ValueError(f"Document line {line_number}: invalid corpus_id")

            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(f"Document line {line_number}: invalid document_id")

            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError(f"Document line {line_number}: invalid source_text")

            document_key = (corpus_id, document_id)

            if document_key in documents:
                raise ValueError(f"Duplicate SemanticDocument identity: {document_key}")

            documents[document_key] = source_text

    return documents


def contexts_from_pattern_record(
    record: dict[str, Any],
    documents: dict[tuple[str, str], str],
) -> TemplateContexts:
    """Extract supporting text from a hybrid-miner PatternRecord.

    Expected PatternRecord fields:
        contract_version
        id
        source_spans
        provenance.corpus_id
        provenance.source_document_ids

    Each source span contains:
        document_id
        start
        end

    The miner's source_spans are deduplicated evidence spans, not
    necessarily complete match occurrences. This baseline treats each
    unique span as a separate context.
    """
    if record.get("contract_version") != "pattern-record-v1":
        raise ValueError("Expected pattern-record-v1")

    # The miner uses 'id', not 'pattern_id'.
    pattern_id = record.get("id")

    if not isinstance(pattern_id, str) or not pattern_id.strip():
        raise ValueError("PatternRecord requires a nonempty id")

    provenance = record.get("provenance")

    if not isinstance(provenance, dict):
        raise ValueError("PatternRecord requires provenance")

    corpus_id = provenance.get("corpus_id")

    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("PatternRecord requires provenance.corpus_id")

    source_document_ids = provenance.get("source_document_ids")

    if (
        not isinstance(source_document_ids, list)
        or not source_document_ids
        or any(
            not isinstance(document_id, str) or not document_id.strip()
            for document_id in source_document_ids
        )
    ):
        raise ValueError("PatternRecord requires valid provenance.source_document_ids")

    source_spans = record.get("source_spans")

    if not isinstance(source_spans, list) or not source_spans:
        raise ValueError(
            "PatternRecord has no source_spans; " "cannot extract supporting contexts"
        )

    contexts: list[str] = []
    seen_spans: set[tuple[str, int, int]] = set()

    for span in source_spans:
        if not isinstance(span, dict):
            raise ValueError("Each source span must be a JSON object")

        document_id = span.get("document_id")
        start = span.get("start")
        end = span.get("end")

        if not isinstance(document_id, str) or document_id not in source_document_ids:
            raise ValueError("Source span references a document absent from provenance")

        # bool is a subclass of int in Python, so reject it explicitly.
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
        ):
            raise ValueError("Invalid source span: expected 0 <= start < end")

        document_key = (corpus_id, document_id)

        if document_key not in documents:
            raise ValueError(f"Missing original source document: {document_key}")

        source_text = documents[document_key]

        if end > len(source_text):
            raise ValueError(
                f"Source span [{start}:{end}] exceeds "
                f"document length {len(source_text)} "
                f"for {document_key}"
            )

        span_identity = (document_id, start, end)

        if span_identity in seen_spans:
            raise ValueError(f"Duplicate source span: {span_identity}")

        seen_spans.add(span_identity)

        # Exact extraction: do not modify or normalize the source text.
        is_relation_span = any(
            other["document_id"] == document_id
            and other["start"] <= start
            and other["end"] >= end
            and (other["start"] < start or other["end"] > end)
            for other in source_spans
        )

        if is_relation_span:
            continue
        context = source_text[start:end]

        contexts.append(context)

    return prepare_contexts(
        pattern_id=pattern_id,
        contexts=contexts,
    )


def load_pattern_contexts(
    pattern_records_path: str | Path,
    documents_path: str | Path,
) -> list[TemplateContexts]:
    """Load miner PatternRecords and resolve their supporting contexts.

    Args:
        pattern_records_path:
            Path to pattern-records.jsonl generated by hybrid-miner.

        documents_path:
            Path to accepted.jsonl containing SemanticDocument records.

    Returns:
        One TemplateContexts object per mined pattern.
    """
    documents = load_semantic_documents(documents_path)

    templates: list[TemplateContexts] = []
    seen_pattern_ids: set[str] = set()

    with Path(pattern_records_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Pattern line {line_number} must contain a JSON object"
                )

            template = contexts_from_pattern_record(
                record=record,
                documents=documents,
            )

            if template.pattern_id in seen_pattern_ids:
                raise ValueError(
                    f"Duplicate pattern id at line {line_number}: "
                    f"{template.pattern_id}"
                )

            seen_pattern_ids.add(template.pattern_id)

            templates.append(template)

    return templates
