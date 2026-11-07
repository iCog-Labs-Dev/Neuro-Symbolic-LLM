"""Producer metadata accompanying semantic parse-result JSONL."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parser.semantic.schema import SemanticParseResult


def metadata_path(documents_path: str | Path) -> Path:
    path = Path(documents_path)
    return path.with_name(f"{path.stem}.metadata.jsonl")


def document_metadata(result: SemanticParseResult, **fields: Any) -> dict[str, Any]:
    """Bind producer annotations to a document's identity and exact source."""
    document = result.document
    return {
        "document_id": document.document_id,
        "corpus_id": document.corpus_id,
        "source_sha256": sha256(document.source_text.encode("utf-8")).hexdigest(),
        **fields,
    }


def load_document_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read a sidecar without allowing duplicate document identities."""
    records = {}
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected an object")
                identity = record["corpus_id"], record["document_id"]
                if any(not isinstance(value, str) or not value for value in identity):
                    raise ValueError("invalid document identity")
                if identity in records:
                    raise ValueError("duplicate document identity")
                if not isinstance(record.get("source_sha256"), str):
                    raise ValueError("missing source fingerprint")
                records[identity] = record
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid metadata at {path}:{number}: {error}"
                ) from error
    return records


def source_metadata(
    result: SemanticParseResult, record: dict[str, Any]
) -> dict[str, str]:
    expected = document_metadata(result)
    if any(record.get(field) != value for field, value in expected.items()):
        raise ValueError("Source metadata does not match the document")
    output: dict[str, str] = {}
    for field in ("source_dataset", "source_domain"):
        if field in record:
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonempty text")
            output[field] = value
    return output
