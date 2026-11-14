
"""End-to-end offline neuro-symbolic preparation pipeline.

Workflow:
    1. Prepare the configured offline dataset.
    2. Reuse existing records and acquire missing records.
    3. Run reference semantic extraction through the semantic CLI.
    4. Reuse, append, resume, or rebuild teacher artifacts.
    5. Generate and validate student pairs and splits through the CLI.
    6. Reuse a compatible student checkpoint or train through the CLI.
    7. Evaluate the student through the semantic CLI.
    8. Run student inference over the complete prepared dataset.
    9. Adapt semantic documents to the hybrid-miner contract.
   10. Write pipeline summaries.

This orchestrator coordinates existing components. It does not implement
a separate reference parser, student trainer, or pair splitter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from offline.offline_dataloader import (
    load_config as load_offline_config,
    select_sources,
)
from parser.ontology.catalog import load_manifest_catalog
from parser.semantic.catalog_document_export import source_record
from parser.semantic.schema import SemanticParseResult, SourceDocument
from symbolic_pipeline.miner_adapter import (
    MinerAdapterError,
    to_semantic_document,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OFFLINE_CONFIG = REPO_ROOT / "configs" / "offline_data.yaml"
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "configs" / "parser_config" / "model_backend.yaml"
)
DEFAULT_TRAINING_CONFIG = (
    REPO_ROOT / "configs" / "parser_config" / "student_config.yaml"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "offline_corpus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "symbolic"

DEFAULT_ONTOLOGY_BUNDLE = (
    REPO_ROOT / "parser" / "ontology" / "bundles"
    / "semantic-extraction-v1.yaml"
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def resolve_path(path: Path | str) -> Path:
    """Resolve paths relative to the repository root."""

    path = Path(path)

    return path if path.is_absolute() else REPO_ROOT / path


def timestamp() -> str:
    """Return a unique UTC timestamp."""

    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )


def log_step(number: int, title: str) -> None:
    print(
        f"\n{'=' * 68}\nSTEP {number}: {title}\n{'=' * 68}",
        flush=True,
    )


def count_jsonl(path: Path) -> int:
    """Count nonempty JSONL records."""

    if not path.is_file():
        return 0

    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and reject malformed files."""

    records = []

    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue

            value = json.loads(line)

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{number}"
                )

            records.append(value)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    """Fingerprint a JSON-compatible object."""

    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def valid_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> bool:
    if not path.is_file():
        return False

    try:
        records = read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    return bool(records) or allow_empty


def valid_semantic_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> bool:
    if not valid_jsonl(path, allow_empty=allow_empty):
        return False

    try:
        for record in read_jsonl(path):
            SemanticParseResult.model_validate(record)

    except (ValueError, ValidationError):
        return False

    return True


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return result if isinstance(result, dict) else None


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def archive_paths(
    paths: list[Path],
    label: str,
    recovery_root: Path,
) -> Path | None:
    """Preserve previous artifacts before replacing them."""

    existing = [path for path in paths if path.exists()]

    if not existing:
        return None

    recovery_root.mkdir(parents=True, exist_ok=True)

    archive = recovery_root / f"{label}-{timestamp()}"
    archive.mkdir(parents=True)

    for path in existing:
        destination = archive / path.name

        if destination.exists():
            destination = archive / f"{path.name}-{timestamp()}"

        shutil.move(str(path), str(destination))

    print(f"Preserved previous artifacts: {archive}", flush=True)

    return archive


# ---------------------------------------------------------------------
# Source identity
# ---------------------------------------------------------------------


def source_identity(
    raw_line: str,
    source_name: str,
) -> tuple[str, str]:
    """Use the same document identity rules as semantic export."""

    document_id, corpus_id, _, _ = source_record(
        raw_line,
        jsonl=True,
        corpus_id=source_name,
    )

    return corpus_id, document_id


def load_source_records(
    path: Path,
    source_name: str,
) -> tuple[list[str], dict[tuple[str, str], str]]:
    """Load source records and index them by stable identity."""

    lines: list[str] = []
    indexed: dict[tuple[str, str], str] = {}

    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            raw = line.rstrip("\r\n")

            if not raw.strip():
                continue

            key = source_identity(raw, source_name)

            if key in indexed:
                raise ValueError(
                    f"Duplicate source document at line {number}: {key}"
                )

            lines.append(raw)
            indexed[key] = raw

    return lines, indexed


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("corpus_id", "")),
        str(record.get("document_id", "")),
    )


# ---------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------


def run_cli(
    command: list[str],
    *,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> int:
    """Run an existing project CLI and stream its output live."""

    print(
        "\n$ " + " ".join(command),
        flush=True,
    )

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
    )

    if result.returncode not in accepted_returncodes:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}."
        )

    return result.returncode


def semantic_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "parser.semantic",
        "--config",
        str(resolve_path(args.model_config)),
    ]

    if args.profile:
        command.extend(["--profile", args.profile])

    return command


# ---------------------------------------------------------------------
# Stage 1: Dataset preparation
# ---------------------------------------------------------------------


def dataset_config_fingerprint(
    config: dict[str, Any],
    source: dict[str, Any],
) -> str:
    """Identify source settings independently of requested record count."""

    source_settings = dict(source)
    source_settings.pop("count", None)

    global_settings = {
        "seed": config.get("seed", 42),
        "min_chars": config.get("min_chars", 20),
        "max_chars": config.get("max_chars", 600),
    }

    return json_fingerprint(
        {
            "source": source_settings,
            "global": global_settings,
        }
    )


def prepare_dataset(
    args: argparse.Namespace,
    config: dict[str, Any],
    source: dict[str, Any],
    prepared_path: Path,
    output_dir: Path,
) -> None:
    """Prepare a dataset while preserving and reusing existing records."""

    name = source["name"]
    requested = int(source["count"])

    manifest_path = output_dir / "dataset_manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    log_step(1, "OFFLINE DATASET PREPARATION")

    print(f"Dataset: {name}")
    print(f"Type: {source['domain']}")
    print(f"Configured records: {requested}")

    if requested < 1:
        raise ValueError(
            f"{name}: configured record count must be positive."
        )

    existing_valid = valid_jsonl(prepared_path)
    existing_count = count_jsonl(prepared_path)

    expected_config = dataset_config_fingerprint(config, source)
    previous_manifest = load_manifest(manifest_path)

    config_matches = (
        previous_manifest is None
        or previous_manifest.get("config_fingerprint") == expected_config
    )

    if (
        not args.force
        and existing_valid
        and existing_count == requested
        and config_matches
    ):
        print(
            f"Reusing prepared dataset: {existing_count} records",
            flush=True,
        )

        if previous_manifest is None:
            write_manifest(
                manifest_path,
                {
                    "config_fingerprint": expected_config,
                    "records": existing_count,
                    "source_sha256": file_sha256(prepared_path),
                    "legacy_adoption": True,
                },
            )

        return

    if args.skip_preparation:
        raise FileNotFoundError(
            f"Prepared dataset does not match configuration: {prepared_path}"
        )

    # Generate the requested complete candidate dataset in a staging area.
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{name}-dataset-",
            dir=output_dir,
        )
    )

    try:
        command = [
            sys.executable,
            "-u",
            "-m",
            "offline.offline_dataloader",
            "--config",
            str(resolve_path(args.config)),
            "--output-dir",
            str(staging),
            "--selection",
            name,
            "--mode",
            "separate",
        ]

        run_cli(command)

        candidate = staging / "separate" / f"{name}.jsonl"

        if not valid_jsonl(candidate):
            raise RuntimeError(
                f"Dataset loader did not produce valid data: {candidate}"
            )

        candidate_count = count_jsonl(candidate)

        if candidate_count != requested:
            raise RuntimeError(
                f"Requested {requested} records but prepared "
                f"{candidate_count}. Teacher extraction will not start."
            )

        old_keys: set[tuple[str, str]] = set()
        new_keys: set[tuple[str, str]] = set()

        if existing_valid:
            _, old_index = load_source_records(prepared_path, name)
            _, new_index = load_source_records(candidate, name)

            old_keys = set(old_index)
            new_keys = set(new_index)

        incremental = (
            existing_valid
            and existing_count < requested
            and old_keys.issubset(new_keys)
            and config_matches
        )

        if incremental:
            print(
                f"Existing records: {existing_count}",
                flush=True,
            )
            print(
                f"New records: {requested - existing_count}",
                flush=True,
            )
            print(
                "Existing dataset is a compatible subset. "
                "Appending newly acquired records.",
                flush=True,
            )

        elif existing_valid:
            print(
                "Dataset configuration or contents changed. "
                "Replacing prepared dataset while preserving the previous version.",
                flush=True,
            )

        if prepared_path.exists():
            archive_paths(
                [prepared_path],
                f"{name}-dataset",
                output_dir / "recovery",
            )

        shutil.move(str(candidate), str(prepared_path))

        write_manifest(
            manifest_path,
            {
                "config_fingerprint": expected_config,
                "records": candidate_count,
                "source_sha256": file_sha256(prepared_path),
                "incremental": incremental,
            },
        )

        print(
            f"Prepared: {candidate_count} records",
            flush=True,
        )

    finally:
        if staging.exists():
            shutil.rmtree(staging)


# ---------------------------------------------------------------------
# Teacher artifact validation
# ---------------------------------------------------------------------


def teacher_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "documents": run_dir / "documents.jsonl",
        "rejected": run_dir / "rejected.jsonl",
        "metadata": run_dir / "documents.metadata.jsonl",
        "pairs": run_dir / "pairs.jsonl",
    }


def validate_teacher_artifacts(
    run_dir: Path,
    source_name: str,
) -> tuple[bool, set[tuple[str, str]]]:
    """Validate teacher outputs and recover their processed source identities."""

    paths = teacher_paths(run_dir)

    if not all(path.is_file() for path in paths.values()):
        return False, set()

    if not valid_semantic_jsonl(paths["documents"]):
        return False, set()

    for name in ("rejected", "metadata", "pairs"):
        if not valid_jsonl(paths[name], allow_empty=True):
            return False, set()

    try:
        documents = read_jsonl(paths["documents"])
        rejected = read_jsonl(paths["rejected"])
        metadata = read_jsonl(paths["metadata"])
        pairs = read_jsonl(paths["pairs"])

        if (
            len(documents) != len(metadata)
            or len(documents) != len(pairs)
        ):
            return False, set()

        processed: set[tuple[str, str]] = set()

        for record in documents:
            result = SemanticParseResult.model_validate(record)
            key = (
                result.document.corpus_id,
                result.document.document_id,
            )

            if key in processed:
                return False, set()

            processed.add(key)

        for record in rejected:
            raw_line = record.get("raw_line")

            if not isinstance(raw_line, str):
                return False, set()

            key = source_identity(raw_line, source_name)

            if key in processed:
                return False, set()

            processed.add(key)

        pair_keys = {
            record_key(pair)
            for pair in pairs
        }

        if pair_keys != {
            (
                str(record["document"]["corpus_id"]),
                str(record["document"]["document_id"]),
            )
            for record in documents
        }:
            return False, set()

        return True, processed

    except (OSError, ValueError, KeyError, TypeError, ValidationError):
        return False, set()


def teacher_fingerprint(args: argparse.Namespace) -> str:
    """Fingerprint reference parser settings and ontology/prompt sources."""

    paths = [
        resolve_path(args.model_config),
        REPO_ROOT / "parser" / "semantic" / "catalog_prompt.py",
        REPO_ROOT / "configs" / "parser_config" / "general_prompt.yaml",
        REPO_ROOT / "configs" / "parser_config" / "discourse_prompt.yaml",
        REPO_ROOT / "configs" / "parser_config" / "parser_prompt.yaml",
        *sorted((REPO_ROOT / "parser" / "ontology").rglob("*.yaml")),
    ]

    values = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in paths
    }

    values["profile"] = args.profile

    return json_fingerprint(values)


# ---------------------------------------------------------------------
# Teacher extraction through the semantic CLI
# ---------------------------------------------------------------------


def export_command(
    args: argparse.Namespace,
    input_path: Path,
    export_dir: Path,
    *,
    resume: bool,
) -> list[str]:
    export_dir.mkdir(parents=True, exist_ok=True)

    paths = teacher_paths(export_dir)

    command = semantic_command(args)

    command.extend(
        [
            "export-documents",
            "--input",
            str(input_path),
            "--output",
            str(paths["documents"]),
            "--rejected-output",
            str(paths["rejected"]),
            "--metadata-output",
            str(paths["metadata"]),
            "--student-pairs-output",
            str(paths["pairs"]),
            "--progress-every",
            str(args.progress_every),
            "--corpus-id",
            args.selection_name,
        ]
    )

    command.extend(
        [
            "--ontology-bundle",
            str(resolve_path(args.ontology_bundle)),
        ]
    )

    if resume:
        command.append("--resume")

    return command


def export_teacher(
    args: argparse.Namespace,
    input_path: Path,
    export_dir: Path,
    *,
    resume: bool = False,
) -> None:
    """Execute reference extraction using the existing semantic CLI."""

    command = export_command(
        args,
        input_path,
        export_dir,
        resume=resume,
    )

    # Return code 2 means completed export with rejected source records.
    run_cli(
        command,
        accepted_returncodes=(0, 2),
    )

    valid, _ = validate_teacher_artifacts(
        export_dir,
        args.selection_name,
    )

    if not valid:
        raise RuntimeError(
            f"Teacher export did not produce complete valid artifacts: {export_dir}"
        )


def merge_teacher_artifacts(
    run_dir: Path,
    delta_dir: Path,
) -> None:
    """Merge validated incremental teacher results into existing artifacts."""

    old_paths = teacher_paths(run_dir)
    new_paths = teacher_paths(delta_dir)

    old_documents = read_jsonl(old_paths["documents"])
    new_documents = read_jsonl(new_paths["documents"])

    old_rejected = read_jsonl(old_paths["rejected"])
    new_rejected = read_jsonl(new_paths["rejected"])

    old_metadata = read_jsonl(old_paths["metadata"])
    new_metadata = read_jsonl(new_paths["metadata"])

    old_pairs = read_jsonl(old_paths["pairs"])
    new_pairs = read_jsonl(new_paths["pairs"])

    def merge(
        old: list[dict[str, Any]],
        new: list[dict[str, Any]],
        key_function,
    ) -> list[dict[str, Any]]:
        result = []
        seen = set()

        for record in old + new:
            key = key_function(record)

            if key in seen:
                continue

            seen.add(key)
            result.append(record)

        return result

    documents = merge(
        old_documents,
        new_documents,
        lambda record: (
            record["document"]["corpus_id"],
            record["document"]["document_id"],
        ),
    )

    rejected = merge(
        old_rejected,
        new_rejected,
        lambda record: json.loads(record["raw_line"])["text"],
    )

    metadata = merge(
        old_metadata,
        new_metadata,
        lambda record: (
            record.get("corpus_id"),
            record.get("document_id"),
        ),
    )

    pairs = merge(
        old_pairs,
        new_pairs,
        lambda record: (
            record.get("corpus_id"),
            record.get("document_id"),
        ),
    )

    # All merged files are written atomically.
    write_jsonl(old_paths["documents"], documents)
    write_jsonl(old_paths["rejected"], rejected)
    write_jsonl(old_paths["metadata"], metadata)
    write_jsonl(old_paths["pairs"], pairs)


def prepare_teacher(
    args: argparse.Namespace,
    source_name: str,
    prepared_path: Path,
    output_dir: Path,
) -> Path:
    """Reuse, append, resume, or rebuild teacher extraction."""

    run_dir = output_dir / "parser_e2e"
    recovery_root = output_dir / "recovery"

    manifest_path = output_dir / "teacher_manifest.json"

    _, current_index = load_source_records(
        prepared_path,
        source_name,
    )

    current_keys = set(current_index)

    if not current_keys:
        raise ValueError("No source records available for teacher extraction.")

    expected_fingerprint = teacher_fingerprint(args)

    previous_manifest = load_manifest(manifest_path)

    # ---------------------------------------------------------
    # Inspect existing completed artifacts.
    # ---------------------------------------------------------

    if run_dir.exists() and not args.force:
        valid, old_keys = validate_teacher_artifacts(
            run_dir,
            source_name,
        )

        if valid and old_keys.issubset(current_keys):
            manifest_compatible = (
                previous_manifest is not None
                and previous_manifest.get("teacher_fingerprint")
                == expected_fingerprint
            )

            exact_match = old_keys == current_keys

            # Adopt compatible legacy completed artifacts only when
            # their source set exactly matches the current dataset.
            if exact_match and (
                manifest_compatible or previous_manifest is None
            ):
                print(
                    f"Reusing completed teacher extraction: "
                    f"{len(old_keys)} source records",
                    flush=True,
                )

                write_teacher_manifest(
                    manifest_path,
                    run_dir,
                    current_keys,
                    expected_fingerprint,
                )

                return run_dir

            # Append only when the previous extraction has a manifest
            # proving compatibility with the current parser configuration.
            if (
                not exact_match
                and manifest_compatible
            ):
                new_keys = current_keys - old_keys

                print(
                    f"Existing teacher results: {len(old_keys)}",
                    flush=True,
                )
                print(
                    f"New records requiring extraction: {len(new_keys)}",
                    flush=True,
                )

                delta_records = [
                    current_index[key]
                    for key in current_index
                    if key in new_keys
                ]

                delta_fingerprint = json_fingerprint(
                    sorted(new_keys)
                )

                delta_dir = (
                    recovery_root
                    / f"teacher-delta-{delta_fingerprint[:16]}"
                )

                delta_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                delta_input = delta_dir / "input.jsonl"

                if not delta_input.exists():
                    delta_input.write_text(
                        "".join(record + "\n" for record in delta_records),
                        encoding="utf-8",
                    )

                delta_valid, _ = validate_teacher_artifacts(
                    delta_dir,
                    source_name,
                )

                if not delta_valid:
                    paths = teacher_paths(delta_dir)

                    checkpoint = paths["documents"].with_name(
                        paths["documents"].name + ".checkpoint.json"
                    )

                    if checkpoint.exists():
                        state = load_manifest(checkpoint)

                        identity = (
                            state.get("identity", {})
                            if state
                            else {}
                        )

                        if (
                            identity.get("source_sha256")
                            != file_sha256(delta_input)
                        ):
                            archive_paths(
                                [delta_dir],
                                "corrupted-teacher-delta",
                                recovery_root,
                            )

                            delta_dir.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            delta_input = delta_dir / "input.jsonl"
                            delta_input.write_text(
                                "".join(
                                    record + "\n"
                                    for record in delta_records
                                ),
                                encoding="utf-8",
                            )

                    export_teacher(
                        args,
                        delta_input,
                        delta_dir,
                        resume=checkpoint.exists(),
                    )

                merge_teacher_artifacts(
                    run_dir,
                    delta_dir,
                )

                valid, merged_keys = validate_teacher_artifacts(
                    run_dir,
                    source_name,
                )

                if not valid or merged_keys != current_keys:
                    raise RuntimeError(
                        "Incremental teacher merge failed validation."
                    )

                archive_paths(
                    [run_dir / "splits"],
                    "previous-student-splits",
                    recovery_root,
                )

                write_teacher_manifest(
                    manifest_path,
                    run_dir,
                    current_keys,
                    expected_fingerprint,
                )

                print(
                    f"Teacher extraction complete: "
                    f"{len(merged_keys)} source records",
                    flush=True,
                )

                return run_dir

        # Different source, changed configuration, or corrupted outputs.
        print(
            "Previous teacher extraction is incompatible or corrupted. "
            "Starting a clean extraction.",
            flush=True,
        )

        archive_paths(
            [run_dir],
            "incompatible-parser-run",
            recovery_root,
        )

    elif run_dir.exists() and args.force:
        archive_paths(
            [run_dir],
            "forced-parser-run",
            recovery_root,
        )

    # ---------------------------------------------------------
    # Resume a valid interrupted full extraction.
    # ---------------------------------------------------------

    run_dir.mkdir(parents=True, exist_ok=True)

    paths = teacher_paths(run_dir)

    checkpoint = paths["documents"].with_name(
        paths["documents"].name + ".checkpoint.json"
    )

    resume = False

    if checkpoint.exists():
        state = load_manifest(checkpoint)

        identity = state.get("identity", {}) if state else {}

        if (
            identity.get("source_sha256")
            == file_sha256(prepared_path)
        ):
            resume = True
            print(
                "Matching teacher checkpoint found. Resuming extraction.",
                flush=True,
            )
        else:
            archive_paths(
                [run_dir],
                "incompatible-teacher-checkpoint",
                recovery_root,
            )
            run_dir.mkdir(parents=True, exist_ok=True)

    # If only incomplete artifacts exist without a checkpoint,
    # preserve them and start fresh.
    elif any(path.exists() for path in paths.values()):
        archive_paths(
            [run_dir],
            "corrupted-teacher-artifacts",
            recovery_root,
        )
        run_dir.mkdir(parents=True, exist_ok=True)

    export_teacher(
        args,
        prepared_path,
        run_dir,
        resume=resume,
    )

    valid, processed = validate_teacher_artifacts(
        run_dir,
        source_name,
    )

    if not valid or processed != current_keys:
        raise RuntimeError(
            "Completed teacher extraction does not cover the current dataset."
        )

    write_teacher_manifest(
        manifest_path,
        run_dir,
        current_keys,
        expected_fingerprint,
    )

    return run_dir


def write_teacher_manifest(
    path: Path,
    run_dir: Path,
    source_keys: set[tuple[str, str]],
    fingerprint: str,
) -> None:
    paths = teacher_paths(run_dir)

    write_manifest(
        path,
        {
            "teacher_fingerprint": fingerprint,
            "source_keys": sorted(
                [list(key) for key in source_keys]
            ),
            "counts": {
                name: count_jsonl(file)
                for name, file in paths.items()
            },
            "artifacts": {
                name: file_sha256(file)
                for name, file in paths.items()
            },
            "updated_at": timestamp(),
        },
    )


# ---------------------------------------------------------------------
# Stage 3: Student-pair splitting through the semantic CLI
# ---------------------------------------------------------------------


def split_pairs(
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    """Reuse valid splits or regenerate them through the CLI."""

    pairs_path = run_dir / "pairs.jsonl"
    split_dir = run_dir / "splits"

    paths = [
        split_dir / "train.jsonl",
        split_dir / "validation.jsonl",
        split_dir / "test.jsonl",
    ]

    pairs = read_jsonl(pairs_path)

    expected_ids = {
        record_key(pair)
        for pair in pairs
    }

    valid_existing = all(valid_jsonl(path) for path in paths)

    if valid_existing:
        try:
            existing = [
                record
                for path in paths
                for record in read_jsonl(path)
            ]

            existing_ids = {
                record_key(record)
                for record in existing
            }

            if (
                len(existing_ids) == len(existing)
                and existing_ids == expected_ids
            ):
                print(
                    "Reusing compatible train/validation/test splits.",
                    flush=True,
                )
                return

        except (OSError, ValueError, KeyError):
            pass

    if split_dir.exists():
        archive_paths(
            [split_dir],
            "incompatible-splits",
            run_dir.parent / "recovery",
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".splits-",
            dir=run_dir,
        )
    )

    try:
        command = semantic_command(args)

        command.extend(
            [
                "split-pairs",
                "--input",
                str(pairs_path),
                "--output-dir",
                str(staging),
                "--train-ratio",
                "0.8",
                "--val-ratio",
                "0.1",
                "--test-ratio",
                "0.1",
                "--seed",
                "42",
                "--ontology-bundle",
                str(resolve_path(args.ontology_bundle)),
            ]
        )

        run_cli(command)

        generated = [
            staging / "train.jsonl",
            staging / "validation.jsonl",
            staging / "test.jsonl",
        ]

        if not all(valid_jsonl(path) for path in generated):
            raise RuntimeError(
                "Semantic CLI produced incomplete student splits."
            )

        combined = [
            record
            for path in generated
            for record in read_jsonl(path)
        ]

        combined_ids = {
            record_key(record)
            for record in combined
        }

        if (
            len(combined_ids) != len(combined)
            or combined_ids != expected_ids
        ):
            raise RuntimeError(
                "Generated splits contain overlaps or missing records."
            )

        staging.rename(split_dir)

    finally:
        if staging.exists():
            shutil.rmtree(staging)


# ---------------------------------------------------------------------
# Student checkpoint compatibility
# ---------------------------------------------------------------------


def valid_student_checkpoint(model_dir: Path) -> bool:
    """Check basic completeness of a saved student checkpoint."""

    required = [
        model_dir / "config.json",
        model_dir / "training_metadata.json",
        model_dir / "tokenizer.json",
    ]

    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False

    weights = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]

    return any(
        path.is_file() and path.stat().st_size > 0
        for path in weights
    )


def checkpoint_fingerprint(model_dir: Path) -> str:
    """Fingerprint checkpoint contents."""

    digest = hashlib.sha256()

    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.name.endswith((".tmp", ".partial")):
            continue

        digest.update(
            str(path.relative_to(model_dir)).encode("utf-8")
        )
        digest.update(file_sha256(path).encode("utf-8"))

    return digest.hexdigest()


def student_training_fingerprint(
    args: argparse.Namespace,
    run_dir: Path,
) -> str:
    """Identify the exact student training data and configuration."""

    training_config = resolve_path(args.training_config)

    train_path = run_dir / "splits" / "train.jsonl"
    validation_path = run_dir / "splits" / "validation.jsonl"

    return json_fingerprint(
        {
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(validation_path),
            "training_config_sha256": file_sha256(training_config),
            "method": args.method,
            "training_prompt_sha256": file_sha256(
                REPO_ROOT / "parser" / "student" / "student_prompt.py"
            ),
            "ontology_sha256": file_sha256(
                resolve_path(args.ontology_bundle)
            ),
        }
    )


def train_student(
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
) -> Path:
    """Reuse a compatible checkpoint or train through the semantic CLI."""

    model_dir = run_dir / "student"
    manifest_path = output_dir / "student_manifest.json"

    fingerprint = student_training_fingerprint(
        args,
        run_dir,
    )

    previous = load_manifest(manifest_path)

    compatible = (
        valid_student_checkpoint(model_dir)
        and previous is not None
        and previous.get("training_fingerprint") == fingerprint
        and previous.get("checkpoint_fingerprint")
        == checkpoint_fingerprint(model_dir)
    )

    if compatible:
        print(
            "Compatible student checkpoint found. Skipping training.",
            flush=True,
        )
        return model_dir

    if args.skip_training:
        raise FileNotFoundError(
            "No student checkpoint matching the current training data "
            "and configuration was found."
        )

    if model_dir.exists():
        archive_paths(
            [model_dir],
            "incompatible-student-checkpoint",
            output_dir / "recovery",
        )

    log_step(4, "STUDENT TRAINING")

    command = [
        sys.executable,
        "-u",
        "-m",
        "parser.semantic",
        "train-student",
        "--train-file",
        str(run_dir / "splits" / "train.jsonl"),
        "--val-file",
        str(run_dir / "splits" / "validation.jsonl"),
        "--training-config",
        str(resolve_path(args.training_config)),
        "--output-dir",
        str(model_dir),
        "--method",
        args.method,
    ]

    run_cli(command)

    if not valid_student_checkpoint(model_dir):
        raise RuntimeError(
            "Student training did not produce a complete checkpoint."
        )

    write_manifest(
        manifest_path,
        {
            "training_fingerprint": fingerprint,
            "checkpoint_fingerprint": checkpoint_fingerprint(model_dir),
            "train_records": count_jsonl(
                run_dir / "splits" / "train.jsonl"
            ),
            "validation_records": count_jsonl(
                run_dir / "splits" / "validation.jsonl"
            ),
            "method": args.method,
            "updated_at": timestamp(),
        },
    )

    return model_dir


# ---------------------------------------------------------------------
# Stage 5: Held-out evaluation through the semantic CLI
# ---------------------------------------------------------------------


def evaluate_student(
    args: argparse.Namespace,
    run_dir: Path,
    model_dir: Path,
) -> None:
    """Use the semantic CLI for held-out student evaluation."""

    log_step(5, "STUDENT HELD-OUT EVALUATION")

    command = semantic_command(args)

    command.extend(
        [
            "e2e",
            "--output-dir",
            str(run_dir),
            "--method",
            args.method,
            "--training-config",
            str(resolve_path(args.training_config)),
            "--ontology-bundle",
            str(resolve_path(args.ontology_bundle)),
            "--skip-training",
        ]
    )

    run_cli(command)


# ---------------------------------------------------------------------
# Stage 6: Full-dataset student inference
# ---------------------------------------------------------------------


def infer_records(
    parser,
    records: list[tuple[str, str]],
    source_name: str,
    parsed_path: Path,
    rejected_path: Path,
    progress_every: int,
) -> None:
    """Run the trained student over source records.

    The semantic CLI currently supports held-out evaluation but not
    batch inference. This uses its existing DistilledSemanticParser API.
    """

    accepted = 0
    rejected = 0

    with (
        parsed_path.open("w", encoding="utf-8") as parsed,
        rejected_path.open("w", encoding="utf-8") as rejected_file,
    ):
        for index, (raw_line, _) in enumerate(records, 1):
            try:
                document_id, corpus_id, text, _ = source_record(
                    raw_line,
                    jsonl=True,
                    corpus_id=source_name,
                )

                document = SourceDocument(
                    document_id=document_id,
                    corpus_id=corpus_id,
                    source_text=text,
                )

                result = parser.parse_document(document)

                parsed.write(result.model_dump_json() + "\n")
                accepted += 1

            except Exception as error:
                rejected_file.write(
                    json.dumps(
                        {
                            "line_number": index,
                            "document_id": (
                                document_id
                                if "document_id" in locals()
                                else f"{source_name}-{index}"
                            ),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1

            processed = accepted + rejected

            if (
                processed % progress_every == 0
                or processed == len(records)
            ):
                percent = processed / len(records) * 100

                print(
                    f"\rStudent inference: {processed}/{len(records)} "
                    f"({percent:.1f}%) | "
                    f"Accepted: {accepted} | Rejected: {rejected}",
                    end="",
                    flush=True,
                )

    print(flush=True)


def run_student_inference(
    args: argparse.Namespace,
    source_name: str,
    prepared_path: Path,
    run_dir: Path,
    output_dir: Path,
    model_dir: Path,
) -> Path:
    """Reuse completed inference or process newly added source records."""

    parsed_path = output_dir / "parsed_documents.jsonl"
    rejected_path = output_dir / "parser_rejected.jsonl"
    manifest_path = output_dir / "inference_manifest.json"

    lines, source_index = load_source_records(
        prepared_path,
        source_name,
    )

    current_keys = set(source_index)

    model_fingerprint = checkpoint_fingerprint(model_dir)

    previous = load_manifest(manifest_path)

    previous_keys = (
        {
            tuple(key)
            for key in previous.get("source_keys", [])
        }
        if previous
        else set()
    )

    outputs_valid = (
        valid_semantic_jsonl(parsed_path, allow_empty=True)
        and valid_jsonl(rejected_path, allow_empty=True)
    )

    if (
        outputs_valid
        and previous is not None
        and previous.get("model_fingerprint") == model_fingerprint
        and previous_keys == current_keys
        and previous.get("parsed_sha256") == file_sha256(parsed_path)
        and previous.get("rejected_sha256") == file_sha256(rejected_path)
    ):
        print(
            f"Reusing completed student inference: "
            f"{count_jsonl(parsed_path)} results",
            flush=True,
        )
        return parsed_path

    # Incremental inference is safe only when previous source identities
    # are a subset of the current dataset and the student model is unchanged.
    incremental = (
        outputs_valid
        and previous is not None
        and previous.get("model_fingerprint") == model_fingerprint
        and previous_keys.issubset(current_keys)
    )

    if incremental:
        new_keys = current_keys - previous_keys

        print(
            f"Reusing previous student inference: {len(previous_keys)} records",
            flush=True,
        )
        print(
            f"New student inference records: {len(new_keys)}",
            flush=True,
        )

        if not new_keys:
            return parsed_path

        old_parsed = read_jsonl(parsed_path)
        old_rejected = read_jsonl(rejected_path)

    else:
        if parsed_path.exists() or rejected_path.exists():
            archive_paths(
                [parsed_path, rejected_path, manifest_path],
                "previous-student-inference",
                output_dir / "recovery",
            )

        new_keys = current_keys
        old_parsed = []
        old_rejected = []

    from parser.semantic.semantic_parser import DistilledSemanticParser

    catalog, _ = load_manifest_catalog(
        resolve_path(args.ontology_bundle)
    )

    student = DistilledSemanticParser.from_pretrained(
        str(model_dir),
        catalog=catalog,
    )

    records = [
        (source_index[key], key)
        for key in source_index
        if key in new_keys
    ]

    log_step(6, "FULL-DATASET STUDENT INFERENCE")

    with tempfile.TemporaryDirectory(
        prefix=".student-inference-",
        dir=output_dir,
    ) as temporary:
        temporary = Path(temporary)

        new_parsed_path = temporary / "parsed.jsonl"
        new_rejected_path = temporary / "rejected.jsonl"

        infer_records(
            student,
            records,
            source_name,
            new_parsed_path,
            new_rejected_path,
            args.progress_every,
        )

        new_parsed = read_jsonl(new_parsed_path)
        new_rejected = read_jsonl(new_rejected_path)

        combined_parsed = old_parsed + new_parsed
        combined_rejected = old_rejected + new_rejected

        write_jsonl(parsed_path, combined_parsed)
        write_jsonl(rejected_path, combined_rejected)

    if not valid_semantic_jsonl(parsed_path, allow_empty=True):
        raise RuntimeError("Student inference produced invalid semantic documents.")

    write_manifest(
        manifest_path,
        {
            "source_keys": sorted(
                [list(key) for key in current_keys]
            ),
            "prepared_sha256": file_sha256(prepared_path),
            "model_fingerprint": model_fingerprint,
            "parsed_sha256": file_sha256(parsed_path),
            "rejected_sha256": file_sha256(rejected_path),
            "accepted": count_jsonl(parsed_path),
            "rejected": count_jsonl(rejected_path),
            "updated_at": timestamp(),
        },
    )

    return parsed_path


# ---------------------------------------------------------------------
# Stage 7: Hybrid-miner adaptation
# ---------------------------------------------------------------------


def adapt_documents(
    parsed_path: Path,
    output_path: Path,
    rejected_path: Path,
    progress_every: int,
) -> tuple[int, int]:
    """Adapt validated semantic results to the hybrid-miner contract."""

    accepted = 0
    rejected = 0

    reasons: Counter[str] = Counter()

    with (
        parsed_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as destination,
        rejected_path.open("w", encoding="utf-8") as rejected_file,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            document_id = f"line-{line_number}"

            try:
                result = SemanticParseResult.model_validate_json(line)

                document_id = result.document.document_id

                adapted = to_semantic_document(result)

                destination.write(
                    adapted.model_dump_json() + "\n"
                )

                accepted += 1

            except (
                ValidationError,
                MinerAdapterError,
                ValueError,
            ) as error:
                rejected += 1

                reason = type(error).__name__
                reasons[reason] += 1

                rejected_file.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "document_id": document_id,
                            "error_type": reason,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            processed = accepted + rejected

            if processed % progress_every == 0:
                print(
                    f"\rMiner adaptation: {processed} processed | "
                    f"Accepted: {accepted} | Rejected: {rejected}",
                    end="",
                    flush=True,
                )

    print(flush=True)

    print(f"Miner documents: {accepted}")
    print(f"Adapter rejected: {rejected}")

    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")

    return accepted, rejected


def run_adapter(
    parsed_path: Path,
    output_dir: Path,
    progress_every: int,
) -> tuple[int, int]:
    """Reuse adapter output only when its source fingerprint matches."""

    output_path = output_dir / "miner_documents.jsonl"
    rejected_path = output_dir / "adapter_rejected.jsonl"
    manifest_path = output_dir / "adapter_manifest.json"

    source_fingerprint = file_sha256(parsed_path)

    previous = load_manifest(manifest_path)

    if (
        valid_jsonl(output_path, allow_empty=True)
        and valid_jsonl(rejected_path, allow_empty=True)
        and previous is not None
        and previous.get("parsed_sha256") == source_fingerprint
        and previous.get("output_sha256") == file_sha256(output_path)
        and previous.get("rejected_sha256") == file_sha256(rejected_path)
    ):
        print(
            f"Reusing completed miner adaptation: "
            f"{count_jsonl(output_path)} documents",
            flush=True,
        )

        return (
            count_jsonl(output_path),
            count_jsonl(rejected_path),
        )

    if output_path.exists() or rejected_path.exists():
        archive_paths(
            [output_path, rejected_path, manifest_path],
            "previous-miner-adaptation",
            output_dir / "recovery",
        )

    log_step(7, "HYBRID-MINER DOCUMENT ADAPTATION")

    with tempfile.TemporaryDirectory(
        prefix=".miner-adaptation-",
        dir=output_dir,
    ) as temporary:
        temporary = Path(temporary)

        temporary_output = temporary / "miner_documents.jsonl"
        temporary_rejected = temporary / "adapter_rejected.jsonl"

        accepted, rejected = adapt_documents(
            parsed_path,
            temporary_output,
            temporary_rejected,
            progress_every,
        )

        shutil.move(str(temporary_output), str(output_path))
        shutil.move(str(temporary_rejected), str(rejected_path))

    write_manifest(
        manifest_path,
        {
            "parsed_sha256": source_fingerprint,
            "output_sha256": file_sha256(output_path),
            "rejected_sha256": file_sha256(rejected_path),
            "accepted": accepted,
            "rejected": rejected,
            "updated_at": timestamp(),
        },
    )

    return accepted, rejected


# ---------------------------------------------------------------------
# Stage 8: Final summary
# ---------------------------------------------------------------------


def write_summary(
    output_dir: Path,
    source_name: str,
    prepared_path: Path,
    run_dir: Path,
    parsed_path: Path,
    adapter_accepted: int,
    adapter_rejected: int,
) -> dict[str, Any]:
    """Write a complete per-dataset pipeline summary."""

    summary = {
        "source": source_name,
        "pipeline": "offline_teacher_student_to_miner_adapter",
        "updated_at": timestamp(),
        "paths": {
            "prepared_dataset": str(prepared_path),
            "teacher_documents": str(run_dir / "documents.jsonl"),
            "teacher_rejected": str(run_dir / "rejected.jsonl"),
            "student_pairs": str(run_dir / "pairs.jsonl"),
            "student_train": str(run_dir / "splits" / "train.jsonl"),
            "student_validation": str(
                run_dir / "splits" / "validation.jsonl"
            ),
            "student_test": str(run_dir / "splits" / "test.jsonl"),
            "student_model": str(run_dir / "student"),
            "parsed_documents": str(parsed_path),
            "miner_documents": str(
                output_dir / "miner_documents.jsonl"
            ),
            "adapter_rejected": str(
                output_dir / "adapter_rejected.jsonl"
            ),
        },
        "counts": {
            "prepared": count_jsonl(prepared_path),
            "teacher_accepted": count_jsonl(
                run_dir / "documents.jsonl"
            ),
            "teacher_rejected": count_jsonl(
                run_dir / "rejected.jsonl"
            ),
            "student_pairs": count_jsonl(
                run_dir / "pairs.jsonl"
            ),
            "train": count_jsonl(
                run_dir / "splits" / "train.jsonl"
            ),
            "validation": count_jsonl(
                run_dir / "splits" / "validation.jsonl"
            ),
            "test": count_jsonl(
                run_dir / "splits" / "test.jsonl"
            ),
            "student_inference": count_jsonl(parsed_path),
            "miner_accepted": adapter_accepted,
            "adapter_rejected": adapter_rejected,
        },
    }

    write_manifest(
        output_dir / "summary.json",
        summary,
    )

    return summary


# ---------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------


def run_source(
    args: argparse.Namespace,
    config: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Run all stages for one configured source."""

    source_name = source["name"]

    args.selection_name = source_name

    data_dir = resolve_path(args.data_dir)
    output_root = resolve_path(args.output_dir)

    output_dir = output_root / source_name

    prepared_path = (
        data_dir / "separate" / f"{source_name}.jsonl"
    )

    run_dir = output_dir / "parser_e2e"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Prepare or reuse dataset.
    prepare_dataset(
        args,
        config,
        source,
        prepared_path,
        output_dir,
    )

    # Stage 2: Reference extraction and incremental recovery.
    log_step(2, "REFERENCE SEMANTIC EXTRACTION")

    prepare_teacher(
        args,
        source_name,
        prepared_path,
        output_dir,
    )

    # Stage 3: Pair validation and splitting through CLI.
    log_step(3, "STUDENT DATASET PREPARATION")

    split_pairs(
        args,
        run_dir,
    )

    # Stage 4: Student checkpoint reuse or CLI training.
    model_dir = train_student(
        args,
        run_dir,
        output_dir,
    )

    # Stage 5: Held-out evaluation through CLI.
    evaluate_student(
        args,
        run_dir,
        model_dir,
    )

    # Stage 6: Full-dataset student inference.
    if args.skip_parsing:
        parsed_path = output_dir / "parsed_documents.jsonl"

        if not valid_semantic_jsonl(parsed_path):
            raise FileNotFoundError(
                f"Existing student inference is missing or invalid: "
                f"{parsed_path}"
            )

    else:
        parsed_path = run_student_inference(
            args,
            source_name,
            prepared_path,
            run_dir,
            output_dir,
            model_dir,
        )

    # Stage 7: Hybrid-miner adaptation.
    adapter_accepted, adapter_rejected = run_adapter(
        parsed_path,
        output_dir,
        args.progress_every,
    )

    # Stage 8: Final summary.
    summary = write_summary(
        output_dir,
        source_name,
        prepared_path,
        run_dir,
        parsed_path,
        adapter_accepted,
        adapter_rejected,
    )

    print(
        f"\nCompleted dataset: {source_name}",
        flush=True,
    )

    print(
        json.dumps(summary["counts"], indent=2),
        flush=True,
    )

    return {
        "source": source_name,
        **summary["counts"],
    }


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Configuration-driven offline dataset preparation, "
            "teacher-student semantic parsing and hybrid-miner adaptation."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_OFFLINE_CONFIG,
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )

    parser.add_argument(
        "--ontology-bundle",
        type=Path,
        default=DEFAULT_ONTOLOGY_BUNDLE,
    )

    parser.add_argument("--profile")

    parser.add_argument(
        "--selection",
        help="Dataset name, domain name, or all.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--method",
        choices=("lora", "full"),
        default="lora",
    )

    parser.add_argument(
        "--min-acceptance",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--skip-preparation",
        action="store_true",
    )

    parser.add_argument(
        "--skip-parsing",
        action="store_true",
        help="Reuse existing full-dataset student inference.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Require an existing compatible student checkpoint.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow resuming matching interrupted teacher exports.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force dataset regeneration and a fresh parser run.",
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = build_argument_parser().parse_args()

    if not 0 <= args.min_acceptance <= 1:
        raise ValueError("--min-acceptance must be between 0 and 1.")

    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive.")

    for path in (
        args.config,
        args.model_config,
        args.training_config,
        args.ontology_bundle,
    ):
        resolved = resolve_path(path)

        if not resolved.is_file():
            raise FileNotFoundError(
                f"Required configuration file not found: {resolved}"
            )

    config = load_offline_config(
        resolve_path(args.config)
    )

    sources = select_sources(
        config,
        args.selection,
    )

    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "\n" + "=" * 68
        + "\nOFFLINE NEURO-SYMBOLIC PREPARATION PIPELINE"
        + "\n" + "=" * 68,
        flush=True,
    )

    print(f"Configuration: {resolve_path(args.config)}")
    print(f"Selected datasets: {len(sources)}")

    for source in sources:
        print(
            f"  - {source['name']} "
            f"[{source['domain']}] "
            f"({source['kind']}, count={source['count']})"
        )

    results = []

    for source in sources:
        result = run_source(
            args,
            config,
            source,
        )
        results.append(result)

    numeric_fields = (
        "prepared",
        "teacher_accepted",
        "teacher_rejected",
        "student_pairs",
        "train",
        "validation",
        "test",
        "student_inference",
        "miner_accepted",
        "adapter_rejected",
    )

    totals = {
        field: sum(item[field] for item in results)
        for field in numeric_fields
    }

    summary = {
        "pipeline": "offline_teacher_student_to_miner_adapter",
        "updated_at": timestamp(),
        "datasets": results,
        "totals": totals,
    }

    summary_path = output_root / "offline_pipeline_summary.json"

    write_manifest(
        summary_path,
        summary,
    )

    print(
        "\n" + "=" * 68
        + "\nOFFLINE PIPELINE COMPLETED"
        + "\n" + "=" * 68,
        flush=True,
    )

    print(f"Datasets: {len(results)}")

    for name, value in totals.items():
        print(f"{name}: {value}")

    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
