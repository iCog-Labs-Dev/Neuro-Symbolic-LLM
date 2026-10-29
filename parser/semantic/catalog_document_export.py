"""Export validated semantic parse results from source-record JSONL.

Each accepted source becomes one SemanticParseResult line. The export keeps a
fingerprinted checkpoint, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from parser.semantic.document_metadata import document_metadata, metadata_path
from parser.semantic.schema import SourceDocument
from parser.semantic.semantic_parser import ReferenceSemanticParser

_SOURCE_FIELDS = frozenset({"document_id", "corpus_id", "source_text"})
_IDENTIFIER = re.compile(r"^[^\s]+$")


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Counts and paths for one completed document export."""

    accepted: int
    rejected: int
    output_path: Path
    rejected_path: Path
    student_pairs_path: Path | None = None
    metadata_path: Path | None = None


def _source_fields(raw_line: str) -> tuple[str, str, str]:
    if not raw_line.strip():
        raise ValueError("Source JSONL line cannot be empty")
    value = json.loads(raw_line)
    if not isinstance(value, dict) or set(value) != _SOURCE_FIELDS:
        raise ValueError(
            "Source record requires exactly document_id, corpus_id, and source_text"
        )
    document_id = value["document_id"]
    corpus_id = value["corpus_id"]
    source_text = value["source_text"]
    for name, identifier in (("document_id", document_id), ("corpus_id", corpus_id)):
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"{name} must be a nonempty identifier without whitespace")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must contain non-whitespace text")
    return document_id, corpus_id, source_text


def source_record(
    raw_line: str,
    *,
    jsonl: bool,
    corpus_id: str,
) -> tuple[str, str, str, dict[str, str]]:
    """Reuse sentence and pilot-manifest inputs without losing their metadata."""
    if jsonl:
        value = json.loads(raw_line)
        if isinstance(value, dict) and set(value) == _SOURCE_FIELDS:
            return (*_source_fields(raw_line), {})
        if (
            not isinstance(value, dict)
            or "text" not in value
            or set(value) - {"text", "source_dataset", "source_domain"}
        ):
            raise ValueError(
                "Expected source IDs/text or the text/dataset/domain manifest"
            )
        text = value["text"]
        metadata = {}
        for field in ("source_dataset", "source_domain"):
            item = value.get(field)
            if item is not None:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"{field} must be nonempty text or null")
                metadata[field] = item
        dataset = metadata.get("source_dataset", corpus_id)
        namespace = dataset if _IDENTIFIER.fullmatch(dataset) else corpus_id
    else:
        text, metadata, namespace = raw_line, {}, corpus_id
    if not isinstance(text, str) or not text.strip():
        raise ValueError("source_text must contain non-whitespace text")
    if not _IDENTIFIER.fullmatch(namespace):
        raise ValueError(
            "Corpus/dataset ID must not contain whitespace; supply --corpus-id"
        )
    identifier = "text-" + sha256(text.encode("utf-8")).hexdigest()
    return identifier, namespace, text, metadata


def export_catalog_documents(
    parser: ReferenceSemanticParser,
    *,
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    student_pairs_path: str | Path | None = None,
    corpus_id: str | None = None,
    metadata_output: str | Path | None = None,
    include_metta: bool = False,
    asserted_only: bool = False,
    resume: bool = False,
    progress_every: int = 100,
) -> ExportResult:
    """Stream documents and retain fingerprinted checkpoints on interruption."""
    import sys

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    source, accepted, rejected = (
        Path(input_path),
        Path(output_path),
        Path(rejected_path),
    )
    outputs = {
        "documents": accepted,
        "rejected": rejected,
        "metadata": (
            Path(metadata_output)
            if metadata_output is not None
            else metadata_path(accepted)
        ),
    }
    if student_pairs_path is not None:
        outputs["pairs"] = Path(student_pairs_path)
    checkpoint = accepted.with_name(accepted.name + ".checkpoint.json")
    partials = {
        name: path.with_name(path.name + ".partial") for name, path in outputs.items()
    }
    paths = [
        source,
        checkpoint,
        checkpoint.with_name(checkpoint.name + ".tmp"),
        *outputs.values(),
        *partials.values(),
    ]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input and output paths must all differ")
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")
    identity = {
        "source_sha256": _file_digest(source),
        "source_path": str(source.resolve()),
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
        "model": parser.model_name,
        "provider": parser.provider_name,
        "prompt_version": parser.prompt_version,
        "model_revision": getattr(parser, "_model_revision", None),
        "generation_settings": {
            name: getattr(parser._backend, name, None)
            for name in (
                "_max_output_tokens",
                "_max_new_tokens",
                "_think",
                "_temperature",
                "_seed",
                "_num_ctx",
            )
        },
        "corpus_id": corpus_id,
        "include_metta": include_metta,
        "asserted_only": asserted_only,
    }
    state: dict[str, Any] = {
        "identity": identity,
        "line_number": 0,
        "accepted": 0,
        "rejected": 0,
        "complete": False,
        "files": {},
    }
    if resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("identity") != identity:
            raise ValueError(
                "Export checkpoint does not match source, outputs, model, or prompt"
            )
        if set(state.get("files", {})) != set(outputs):
            raise ValueError("Export checkpoint has an invalid artifact set")
        if any(
            not isinstance(state.get(name), int)
            or isinstance(state[name], bool)
            or state[name] < 0
            for name in ("line_number", "accepted", "rejected")
        ) or not isinstance(state.get("complete"), bool):
            raise ValueError("Export checkpoint has invalid counters")
        for name, partial in partials.items():
            path = partial if partial.exists() else outputs[name]
            if path.is_symlink() or path.samefile(source):
                raise ValueError(
                    "Resume artifacts must not alias the source or use symlinks"
                )
            if not partial.exists() and not state["complete"]:
                raise ValueError(f"Missing partial export: {partial}")
            record = state["files"][name]
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("size"), int)
                or isinstance(record["size"], bool)
                or record["size"] < 0
                or not isinstance(record.get("sha256"), str)
            ):
                raise ValueError("Export checkpoint has invalid artifact fingerprints")
            if state["complete"] and path.stat().st_size != record["size"]:
                raise ValueError(f"Completed export artifact differs: {path}")
            if (
                path.stat().st_size < record["size"]
                or _file_digest(path, record["size"]) != record["sha256"]
            ):
                raise ValueError(f"Export checkpoint artifact differs: {path}")
            if outputs[name].exists() and partial.exists():
                raise FileExistsError(f"Output already exists: {outputs[name]}")
    else:
        for path in [checkpoint, *outputs.values(), *partials.values()]:
            if path.exists():
                raise FileExistsError(
                    f"Output already exists: {path}; use --resume for an interrupted export"
                )
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not state["complete"]:
        seen: set[tuple[str, str]] = set()
        try:
            with ExitStack() as files:
                handles, hashes = {}, {}
                for name, partial in partials.items():
                    handle = files.enter_context(
                        partial.open("r+b" if resume else "x+b")
                    )
                    offset = state["files"][name]["size"] if resume else 0
                    handle.truncate(offset)
                    handle.seek(0)
                    digest = sha256()
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                    handles[name], hashes[name] = handle, digest

                def write(name: str, payload: str) -> None:
                    content = (payload + "\n").encode("utf-8")
                    handles[name].write(content)
                    hashes[name].update(content)

                def mark() -> None:
                    state["files"] = {
                        name: {
                            "size": handle.tell(),
                            "sha256": hashes[name].hexdigest(),
                        }
                        for name, handle in handles.items()
                    }

                def persist() -> None:
                    for handle in handles.values():
                        handle.flush()
                    _write_checkpoint(checkpoint, state)

                if not resume:
                    mark()
                    persist()
                with source.open(encoding="utf-8") as source_file:
                    for number, line in enumerate(source_file, 1):
                        raw = line.removesuffix("\n").removesuffix("\r")
                        if number <= state["line_number"]:
                            try:
                                doc_id, corpus, _, _ = source_record(
                                    raw,
                                    jsonl=source.suffix.lower() == ".jsonl",
                                    corpus_id=corpus_id or source.stem,
                                )
                                seen.add((corpus, doc_id))
                            except ValueError:
                                pass
                            continue
                        try:
                            doc_id, corpus, text, annotations = source_record(
                                raw,
                                jsonl=source.suffix.lower() == ".jsonl",
                                corpus_id=corpus_id or source.stem,
                            )
                            key = corpus, doc_id
                            if key in seen:
                                raise ValueError(f"Duplicate source document {key!r}")
                            seen.add(key)
                            result = parser.parse_document(
                                SourceDocument(
                                    document_id=doc_id,
                                    corpus_id=corpus,
                                    source_text=text,
                                )
                            )
                        except ValueError as error:
                            write(
                                "rejected",
                                json.dumps(
                                    {
                                        "line_number": number,
                                        "raw_line": raw,
                                        "error": str(error),
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            state["rejected"] += 1
                        else:
                            metadata = document_metadata(result, **annotations)
                            if include_metta or asserted_only:
                                from parser.semantic.metta_renderer import (
                                    render_metta,
                                    validate_rendered_metta,
                                )

                                try:
                                    expressions = render_metta(
                                        result, asserted_only=asserted_only
                                    )
                                    validate_rendered_metta(expressions)
                                    metadata["metta"] = expressions
                                    metadata["metta_asserted_only"] = asserted_only
                                except ValueError as error:
                                    metadata["rendering_error"] = str(error)
                            payloads = {
                                "documents": result.model_dump_json(),
                                "metadata": json.dumps(metadata, ensure_ascii=False),
                            }
                            if "pairs" in handles:
                                from parser.semantic.catalog_student import (
                                    make_student_pair,
                                )

                                payloads["pairs"] = json.dumps(
                                    make_student_pair(result, **annotations),
                                    ensure_ascii=False,
                                )
                            for name, payload in payloads.items():
                                write(name, payload)
                            state["accepted"] += 1
                        state["line_number"] = number
                        mark()
                        if number % progress_every == 0:
                            persist()
                            print(
                                f"Processed {number} records | accepted={state['accepted']} rejected={state['rejected']}",
                                file=sys.stderr,
                            )
                state["complete"] = True
                persist()
        except BaseException:
            if state["files"]:
                _write_checkpoint(checkpoint, state)
            raise
    for name, partial in partials.items():
        if partial.exists():
            if outputs[name].exists():
                raise FileExistsError(f"Output already exists: {outputs[name]}")
            partial.rename(outputs[name])
    checkpoint.unlink()
    return ExportResult(
        state["accepted"],
        state["rejected"],
        accepted,
        rejected,
        outputs.get("pairs"),
        outputs["metadata"],
    )


def _file_digest(path: Path, limit: int | None = None) -> str:
    digest = sha256()
    with path.open("rb") as file:
        remaining = path.stat().st_size if limit is None else limit
        while remaining:
            chunk = file.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Incomplete artifact: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)
    temporary.replace(path)
