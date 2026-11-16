"""End-to-end offline neuro-symbolic preparation pipeline.

The offline runner orchestrates existing project components.

Workflow:
    1. Prepare configured offline datasets.
    2. Reuse existing records or acquire missing records.
    3. Run reference extraction through parser.semantic CLI.
    4. Reuse, append, resume, or rebuild teacher extraction.
    5. Generate and validate student pairs and splits through E2E.
    6. Reuse a compatible student checkpoint or train through E2E.
    7. Evaluate the student on the held-out test set through E2E.
    8. Run student inference over the complete prepared dataset.
    9. Adapt semantic documents to the hybrid-miner contract.
   10. Write pipeline summaries.

The semantic CLI owns extraction, pair generation, splitting,
student training, and held-out evaluation.
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

from pydantic import ValidationError

from offline.offline_dataloader import (
    load_config,
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

DEFAULT_OFFLINE_CONFIG = REPO_ROOT / "configs/offline_data.yaml"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs/parser_config/model_backend.yaml"
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs/parser_config/student_config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data/offline_corpus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/symbolic"

DEFAULT_ONTOLOGY_BUNDLE = (
    REPO_ROOT / "parser/ontology/bundles/semantic-extraction-v1.yaml"
)

TEACHER_FILES = {
    "documents": "documents.jsonl",
    "rejected": "rejected.jsonl",
    "metadata": "documents.metadata.jsonl",
    "pairs": "pairs.jsonl",
}


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def resolve_path(path: Path | str) -> Path:
    """Resolve paths relative to the repository root."""

    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def log_step(number: int, title: str) -> None:
    print(
        f"\n{'=' * 68}\nSTEP {number}: {title}\n{'=' * 68}",
        flush=True,
    )


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0

    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue

            value = json.loads(line)

            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")

            records.append(value)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint(value: Any) -> str:
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None


def write_manifest(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(values, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def archive_paths(
    paths: list[Path],
    label: str,
    recovery_root: Path,
) -> Path | None:
    """Preserve existing artifacts before replacing them."""

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
    """Load source records and index them by stable document identity."""

    lines = []
    indexed: dict[tuple[str, str], str] = {}

    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            raw = line.rstrip("\r\n")

            if not raw.strip():
                continue

            key = source_identity(raw, source_name)

            if key in indexed:
                raise ValueError(f"Duplicate source record at line {number}: {key}")

            lines.append(raw)
            indexed[key] = raw

    return lines, indexed


def semantic_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("corpus_id", "")),
        str(record.get("document_id", "")),
    )


def source_fingerprint_key(key: tuple[str, str]) -> str:
    """Serialize a source identity safely for JSON manifest keys."""
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def source_fingerprints(
    indexed_records: dict[tuple[str, str], str],
) -> dict[str, str]:
    """Fingerprint source content independently of document identity."""
    return {
        source_fingerprint_key(key): hashlib.sha256(raw.encode("utf-8")).hexdigest()
        for key, raw in indexed_records.items()
    }


# ---------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------


def run_cli(
    command: list[str],
    *,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> int:
    """Run a project CLI and stream its output directly."""

    print("\n$ " + " ".join(command), flush=True)

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
    )

    if result.returncode not in accepted_returncodes:
        raise RuntimeError(f"Command failed with exit code {result.returncode}.")

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


def e2e_command(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    input_path: Path | None = None,
    resume: bool = False,
    skip_training: bool = False,
) -> list[str]:
    """Build the existing semantic E2E CLI command."""

    command = semantic_command(args)

    command.extend(
        [
            "e2e",
            "--output-dir",
            str(run_dir),
            "--training-config",
            str(resolve_path(args.training_config)),
            "--method",
            args.method,
            "--ontology-bundle",
            str(resolve_path(args.ontology_bundle)),
            "--min-acceptance",
            str(args.min_acceptance),
        ]
    )

    if input_path is not None:
        command.extend(["--input", str(input_path)])

    if resume:
        command.append("--resume")

    if skip_training:
        command.append("--skip-training")

    return command


# ---------------------------------------------------------------------
# Stage 1: Dataset preparation
# ---------------------------------------------------------------------


def dataset_config_fingerprint(
    config: dict[str, Any],
    source: dict[str, Any],
) -> str:
    source_settings = dict(source)
    source_settings.pop("count", None)

    return fingerprint(
        {
            "source": source_settings,
            "seed": config.get("seed", 42),
            "min_chars": config.get("min_chars", 20),
            "max_chars": config.get("max_chars", 600),
        }
    )


def prepare_dataset(
    args: argparse.Namespace,
    config: dict[str, Any],
    source: dict[str, Any],
    prepared_path: Path,
    output_dir: Path,
) -> None:
    """Reuse the dataset or prepare the configured number of records."""

    name = source["name"]
    requested = int(source["count"])

    manifest_path = output_dir / "dataset_manifest.json"
    recovery_root = output_dir / "recovery"

    log_step(1, "OFFLINE DATASET PREPARATION")

    print(f"Dataset: {name}")
    print(f"Type: {source['domain']}")
    print(f"Configured records: {requested}")

    if requested < 1:
        raise ValueError(f"{name}: count must be positive.")

    existing_valid = valid_jsonl(prepared_path)
    existing_count = count_jsonl(prepared_path)

    config_id = dataset_config_fingerprint(config, source)
    previous = load_manifest(manifest_path)

    config_matches = previous is None or previous.get("config_fingerprint") == config_id

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

        if previous is None:
            write_manifest(
                manifest_path,
                {
                    "config_fingerprint": config_id,
                    "source_sha256": file_sha256(prepared_path),
                    "records": existing_count,
                    "legacy_adoption": True,
                },
            )

        return

    if args.skip_preparation:
        raise FileNotFoundError(
            f"Prepared dataset is missing or incompatible: {prepared_path}"
        )

    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{name}-dataset-",
            dir=prepared_path.parent,
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
            raise RuntimeError("Dataset loader produced invalid output.")

        candidate_count = count_jsonl(candidate)

        if candidate_count != requested:
            raise RuntimeError(
                f"Expected {requested} records, received {candidate_count}."
            )

        incremental = False

        if existing_valid and config_matches:
            _, old_index = load_source_records(prepared_path, name)
            _, new_index = load_source_records(candidate, name)

            old_keys = set(old_index)
            new_keys = set(new_index)
            changed_keys = {
                key
                for key in old_keys.intersection(new_keys)
                if old_index[key] != new_index[key]
            }

            incremental = (
                old_keys.issubset(new_keys)
                and len(new_keys) > len(old_keys)
                and not changed_keys
            )

            if incremental:
                print(f"Existing records: {len(old_keys)}")
                print(f"New records: {len(new_keys - old_keys)}")
                print("Existing source records are unchanged.")
            elif changed_keys:
                print(
                    f"Changed existing source records: {len(changed_keys)}; "
                    "teacher extraction will be rebuilt."
                )

        if prepared_path.exists():
            archive_paths(
                [prepared_path],
                f"{name}-dataset",
                recovery_root,
            )

        shutil.move(str(candidate), str(prepared_path))

        write_manifest(
            manifest_path,
            {
                "config_fingerprint": config_id,
                "source_sha256": file_sha256(prepared_path),
                "records": candidate_count,
                "incremental": incremental,
                "updated_at": timestamp(),
            },
        )

        print(f"Prepared: {candidate_count} records", flush=True)

    finally:
        if staging.exists():
            shutil.rmtree(staging)


# ---------------------------------------------------------------------
# Teacher artifact validation
# ---------------------------------------------------------------------


def teacher_paths(run_dir: Path) -> dict[str, Path]:
    return {name: run_dir / filename for name, filename in TEACHER_FILES.items()}


def validate_teacher_artifacts(
    run_dir: Path,
    source_name: str,
) -> tuple[bool, set[tuple[str, str]]]:
    """Validate completed teacher documents, rejected records, pairs and metadata."""

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

        if len(documents) != len(metadata):
            return False, set()

        if len(documents) != len(pairs):
            return False, set()

        processed: set[tuple[str, str]] = set()
        document_keys = set()

        for record in documents:
            result = SemanticParseResult.model_validate(record)

            key = (
                result.document.corpus_id,
                result.document.document_id,
            )

            if key in processed:
                return False, set()

            processed.add(key)
            document_keys.add(key)

        for record in rejected:
            raw_line = record.get("raw_line")

            if not isinstance(raw_line, str):
                return False, set()

            key = source_identity(raw_line, source_name)

            if key in processed:
                return False, set()

            processed.add(key)

        pair_keys = {semantic_key(pair) for pair in pairs}

        if pair_keys != document_keys:
            return False, set()

        return True, processed

    except (OSError, ValueError, KeyError, TypeError, ValidationError):
        return False, set()


def teacher_fingerprint(args: argparse.Namespace) -> str:
    paths = [
        resolve_path(args.model_config),
        REPO_ROOT / "parser/semantic/catalog_prompt.py",
        REPO_ROOT / "configs/parser_config/general_prompt.yaml",
        REPO_ROOT / "configs/parser_config/discourse_prompt.yaml",
        REPO_ROOT / "configs/parser_config/parser_prompt.yaml",
        *sorted((REPO_ROOT / "parser/ontology").rglob("*.yaml")),
    ]

    values = {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths}

    values["profile"] = args.profile

    return fingerprint(values)


def write_teacher_manifest(
    path: Path,
    run_dir: Path,
    source_index: dict[tuple[str, str], str],
    parser_fingerprint: str,
) -> None:
    """Record teacher artifacts and the exact source content they represent."""
    paths = teacher_paths(run_dir)

    write_manifest(
        path,
        {
            "parser_fingerprint": parser_fingerprint,
            "source_keys": sorted([list(key) for key in source_index]),
            "source_fingerprints": source_fingerprints(source_index),
            "artifacts": {name: file_sha256(file) for name, file in paths.items()},
            "counts": {name: count_jsonl(file) for name, file in paths.items()},
            "updated_at": timestamp(),
        },
    )


# ---------------------------------------------------------------------
# Incremental teacher export
# ---------------------------------------------------------------------


def export_delta(
    args: argparse.Namespace,
    source_name: str,
    input_path: Path,
    delta_dir: Path,
    *,
    resume: bool = False,
) -> None:
    """Use the reference export CLI for newly added source records."""

    delta_dir.mkdir(parents=True, exist_ok=True)

    paths = teacher_paths(delta_dir)

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
            "--corpus-id",
            source_name,
            "--ontology-bundle",
            str(resolve_path(args.ontology_bundle)),
            "--progress-every",
            str(args.progress_every),
        ]
    )

    if resume:
        command.append("--resume")

    run_cli(
        command,
        accepted_returncodes=(0, 2),
    )

    valid, _ = validate_teacher_artifacts(
        delta_dir,
        source_name,
    )

    if not valid:
        raise RuntimeError("Incremental teacher export failed validation.")


def merge_unique(
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


def merge_teacher_artifacts(
    run_dir: Path,
    delta_dir: Path,
    source_name: str,
) -> None:
    """Merge incremental extraction results without rerunning old records."""

    old = teacher_paths(run_dir)
    new = teacher_paths(delta_dir)

    documents = merge_unique(
        read_jsonl(old["documents"]),
        read_jsonl(new["documents"]),
        lambda record: (
            record["document"]["corpus_id"],
            record["document"]["document_id"],
        ),
    )

    pairs = merge_unique(
        read_jsonl(old["pairs"]),
        read_jsonl(new["pairs"]),
        semantic_key,
    )

    metadata = merge_unique(
        read_jsonl(old["metadata"]),
        read_jsonl(new["metadata"]),
        lambda record: (
            record.get("corpus_id"),
            record.get("document_id"),
        ),
    )

    def rejected_key(record):
        return source_identity(record["raw_line"], source_name)

    rejected = merge_unique(
        read_jsonl(old["rejected"]),
        read_jsonl(new["rejected"]),
        rejected_key,
    )

    write_jsonl(old["documents"], documents)
    write_jsonl(old["pairs"], pairs)
    write_jsonl(old["metadata"], metadata)
    write_jsonl(old["rejected"], rejected)


# ---------------------------------------------------------------------
# Stage 2: Teacher extraction decision
# ---------------------------------------------------------------------


def prepare_teacher(
    args: argparse.Namespace,
    source_name: str,
    prepared_path: Path,
    output_dir: Path,
) -> tuple[Path | None, bool]:
    """Decide whether E2E should extract, resume, or reuse teacher results."""

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
    current_source_fingerprints = source_fingerprints(current_index)
    previous = load_manifest(manifest_path)

    if run_dir.exists() and not args.force:
        valid, old_keys = validate_teacher_artifacts(
            run_dir,
            source_name,
        )

        if valid:
            previous_source_fingerprints = (
                previous.get("source_fingerprints") if previous is not None else None
            )
            source_content_compatible = isinstance(
                previous_source_fingerprints, dict
            ) and all(
                previous_source_fingerprints.get(source_fingerprint_key(key))
                == current_source_fingerprints.get(source_fingerprint_key(key))
                for key in old_keys
            )
            manifest_compatible = (
                previous is not None
                and previous.get("parser_fingerprint") == expected_fingerprint
                and source_content_compatible
            )

            if (
                old_keys == current_keys
                and manifest_compatible
                and previous_source_fingerprints == current_source_fingerprints
            ):
                print(
                    f"Reusing completed teacher extraction: "
                    f"{len(old_keys)} source records",
                    flush=True,
                )

                write_teacher_manifest(
                    manifest_path,
                    run_dir,
                    current_index,
                    expected_fingerprint,
                )

                return None, not validate_splits(run_dir)

            if old_keys < current_keys and manifest_compatible:
                new_keys = current_keys - old_keys

                print(f"Existing teacher records: {len(old_keys)}")
                print(f"New records requiring extraction: {len(new_keys)}")

                delta_fingerprint = fingerprint(sorted(new_keys))

                delta_dir = recovery_root / f"teacher-delta-{delta_fingerprint[:16]}"

                delta_input = delta_dir / "input.jsonl"

                delta_records = [
                    current_index[key] for key in current_index if key in new_keys
                ]

                delta_dir.mkdir(parents=True, exist_ok=True)

                expected_content = "".join(record + "\n" for record in delta_records)

                # FIX: Ruff SIM102 - combine nested conditions.
                if (
                    delta_input.exists()
                    and delta_input.read_text(encoding="utf-8") != expected_content
                ):
                    archive_paths(
                        [delta_dir],
                        "changed-teacher-delta",
                        recovery_root,
                    )
                    delta_dir.mkdir(parents=True, exist_ok=True)

                delta_input.write_text(
                    expected_content,
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

                    resume_delta = False

                    if checkpoint.exists():
                        state = load_manifest(checkpoint)
                        identity = state.get("identity", {}) if state else {}

                        if identity.get("source_sha256") == file_sha256(delta_input):
                            resume_delta = True
                        else:
                            archive_paths(
                                [delta_dir],
                                "incompatible-teacher-delta",
                                recovery_root,
                            )
                            delta_dir.mkdir(parents=True, exist_ok=True)

                            delta_input = delta_dir / "input.jsonl"

                            delta_input.write_text(
                                expected_content,
                                encoding="utf-8",
                            )

                    export_delta(
                        args,
                        source_name,
                        delta_input,
                        delta_dir,
                        resume=resume_delta,
                    )

                merge_teacher_artifacts(
                    run_dir,
                    delta_dir,
                    source_name,
                )

                merged_valid, merged_keys = validate_teacher_artifacts(
                    run_dir,
                    source_name,
                )

                if not merged_valid or merged_keys != current_keys:
                    raise RuntimeError("Merged teacher artifacts failed validation.")

                archive_paths(
                    [run_dir / "splits"],
                    "previous-student-splits",
                    recovery_root,
                )

                write_teacher_manifest(
                    manifest_path,
                    run_dir,
                    current_index,
                    expected_fingerprint,
                )

                print(
                    f"Incremental teacher extraction complete: "
                    f"{len(merged_keys)} records",
                    flush=True,
                )

                return None, True

        print(
            "Previous teacher extraction is incompatible or corrupted. "
            "Starting a clean extraction.",
            flush=True,
        )

        archive_paths(
            [run_dir],
            "incompatible-teacher-run",
            recovery_root,
        )

        # Do not recreate run_dir here.
        # The semantic E2E CLI creates it for a fresh run.
        return prepared_path, False

    if run_dir.exists() and args.force:
        archive_paths(
            [run_dir],
            "forced-teacher-run",
            recovery_root,
        )

        return prepared_path, False

    return prepared_path, False


# ---------------------------------------------------------------------
# Student split validation
# ---------------------------------------------------------------------


def validate_splits(run_dir: Path) -> bool:
    split_dir = run_dir / "splits"

    paths = [
        split_dir / "train.jsonl",
        split_dir / "validation.jsonl",
        split_dir / "test.jsonl",
    ]

    if not all(valid_jsonl(path) for path in paths):
        return False

    try:
        pairs = read_jsonl(run_dir / "pairs.jsonl")

        expected = {semantic_key(pair) for pair in pairs}

        combined = [record for path in paths for record in read_jsonl(path)]

        actual = {semantic_key(record) for record in combined}

        return len(actual) == len(combined) and actual == expected

    except (OSError, ValueError, KeyError):
        return False


# ---------------------------------------------------------------------
# Student checkpoint compatibility
# ---------------------------------------------------------------------


def valid_student_checkpoint(model_dir: Path) -> bool:
    required = [
        model_dir / "config.json",
        model_dir / "training_metadata.json",
        model_dir / "tokenizer.json",
    ]

    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False

    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )

    return any(
        (model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
        for name in weight_names
    )


def checkpoint_fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256()

    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.name.endswith((".tmp", ".partial")):
            continue

        digest.update(str(path.relative_to(model_dir)).encode("utf-8"))
        digest.update(file_sha256(path).encode("utf-8"))

    return digest.hexdigest()


def student_training_fingerprint(
    args: argparse.Namespace,
    run_dir: Path,
) -> str:
    train_path = run_dir / "splits" / "train.jsonl"
    validation_path = run_dir / "splits" / "validation.jsonl"

    return fingerprint(
        {
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(validation_path),
            "training_config_sha256": file_sha256(resolve_path(args.training_config)),
            "method": args.method,
            "prompt_sha256": file_sha256(
                REPO_ROOT / "parser/student/student_prompt.py"
            ),
            "ontology_sha256": file_sha256(resolve_path(args.ontology_bundle)),
        }
    )


def student_checkpoint_compatible(
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
) -> bool:
    model_dir = run_dir / "student"

    manifest = load_manifest(output_dir / "student_manifest.json")

    if not valid_student_checkpoint(model_dir) or manifest is None:
        return False

    return manifest.get("training_fingerprint") == student_training_fingerprint(
        args, run_dir
    ) and manifest.get("checkpoint_fingerprint") == checkpoint_fingerprint(model_dir)


def record_student_checkpoint(
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
) -> None:
    model_dir = run_dir / "student"

    if not valid_student_checkpoint(model_dir):
        raise RuntimeError(
            "Semantic E2E did not produce a complete student checkpoint."
        )

    write_manifest(
        output_dir / "student_manifest.json",
        {
            "training_fingerprint": student_training_fingerprint(
                args,
                run_dir,
            ),
            "checkpoint_fingerprint": checkpoint_fingerprint(model_dir),
            "method": args.method,
            "updated_at": timestamp(),
        },
    )


# ---------------------------------------------------------------------
# Stage 3-5: One semantic E2E invocation
# ---------------------------------------------------------------------


def run_semantic_e2e(
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
    input_path: Path | None,
    resume: bool,
) -> Path:
    """Invoke the semantic CLI exactly once for this dataset."""

    recovery_root = output_dir / "recovery"

    if run_dir.exists() and not validate_splits(run_dir):
        split_dir = run_dir / "splits"

        if split_dir.exists():
            archive_paths(
                [split_dir],
                "incomplete-student-splits",
                recovery_root,
            )

    cached_splits = run_dir.exists() and validate_splits(run_dir)

    compatible_student = cached_splits and student_checkpoint_compatible(
        args,
        run_dir,
        output_dir,
    )

    if args.skip_training and not compatible_student:
        raise FileNotFoundError("No compatible student checkpoint is available.")

    if compatible_student:
        print(
            "Compatible student checkpoint found. Skipping training.",
            flush=True,
        )

    elif (run_dir / "student").exists():
        archive_paths(
            [run_dir / "student"],
            "incompatible-student-checkpoint",
            recovery_root,
        )

    run_cli(
        e2e_command(
            args,
            run_dir,
            input_path=input_path,
            resume=resume,
            skip_training=compatible_student,
        )
    )

    if not validate_teacher_artifacts(
        run_dir,
        args.selection_name,
    )[0]:
        raise RuntimeError("Semantic E2E did not leave valid teacher artifacts.")

    if not validate_splits(run_dir):
        raise RuntimeError(
            "Semantic E2E did not produce valid train/validation/test splits."
        )

    model_dir = run_dir / "student"

    if not valid_student_checkpoint(model_dir):
        raise RuntimeError("Semantic E2E did not produce a valid student checkpoint.")

    record_student_checkpoint(
        args,
        run_dir,
        output_dir,
    )

    return model_dir


# ---------------------------------------------------------------------
# Stage 6: Full-dataset student inference
# ---------------------------------------------------------------------


def infer_records(
    student,
    source_name: str,
    records: list[tuple[str, tuple[str, str]]],
    parsed_path: Path,
    rejected_path: Path,
    progress_every: int,
) -> None:
    accepted = 0
    rejected = 0
    total = len(records)

    with (
        parsed_path.open("w", encoding="utf-8") as parsed,
        rejected_path.open("w", encoding="utf-8") as rejected_file,
    ):
        for index, (raw_line, key) in enumerate(records, 1):
            document_id = key[1]
            corpus_id = key[0]

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

                result = student.parse_document(document)

                parsed.write(result.model_dump_json() + "\n")
                accepted += 1

            except Exception as error:
                rejected_file.write(
                    json.dumps(
                        {
                            "line_number": index,
                            "document_id": document_id,
                            "corpus_id": corpus_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rejected += 1

            processed = accepted + rejected

            if processed % progress_every == 0 or processed == total:
                percent = processed / total * 100 if total else 0

                print(
                    f"\rStudent inference: {processed}/{total} "
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
    output_dir: Path,
    model_dir: Path,
) -> Path:
    """Reuse completed inference or infer only newly added records."""

    parsed_path = output_dir / "parsed_documents.jsonl"
    rejected_path = output_dir / "parser_rejected.jsonl"
    manifest_path = output_dir / "inference_manifest.json"

    _, source_index = load_source_records(
        prepared_path,
        source_name,
    )

    current_keys = set(source_index)
    current_source_fingerprints = source_fingerprints(source_index)
    model_hash = checkpoint_fingerprint(model_dir)

    previous = load_manifest(manifest_path)

    outputs_valid = valid_semantic_jsonl(parsed_path, allow_empty=True) and valid_jsonl(
        rejected_path, allow_empty=True
    )

    previous_keys = set()
    previous_source_fingerprints = None

    if previous is not None:
        previous_keys = {tuple(key) for key in previous.get("source_keys", [])}
        previous_source_fingerprints = previous.get("source_fingerprints")

    previous_sources_unchanged = isinstance(previous_source_fingerprints, dict) and all(
        previous_source_fingerprints.get(source_fingerprint_key(key))
        == current_source_fingerprints.get(source_fingerprint_key(key))
        for key in previous_keys
    )

    if (
        outputs_valid
        and previous is not None
        and previous.get("model_fingerprint") == model_hash
        and previous_keys == current_keys
        and previous_source_fingerprints == current_source_fingerprints
        and previous.get("prepared_sha256") == file_sha256(prepared_path)
        and previous.get("parsed_sha256") == file_sha256(parsed_path)
        and previous.get("rejected_sha256") == file_sha256(rejected_path)
    ):
        print(
            f"Reusing completed student inference: "
            f"{count_jsonl(parsed_path)} results",
            flush=True,
        )
        return parsed_path

    incremental = (
        outputs_valid
        and previous is not None
        and previous.get("model_fingerprint") == model_hash
        and previous_keys.issubset(current_keys)
        and previous_sources_unchanged
    )

    if incremental:
        new_keys = current_keys - previous_keys

        print(f"Reusing previous inference: {len(previous_keys)} records")
        print(f"New records requiring inference: {len(new_keys)}")

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

    if not new_keys:
        return parsed_path

    from parser.semantic.semantic_parser import DistilledSemanticParser

    catalog, _ = load_manifest_catalog(resolve_path(args.ontology_bundle))

    student = DistilledSemanticParser.from_pretrained(
        str(model_dir),
        catalog=catalog,
    )

    records = [(source_index[key], key) for key in source_index if key in new_keys]

    log_step(6, "FULL-DATASET STUDENT INFERENCE")

    with tempfile.TemporaryDirectory(
        prefix=".student-inference-",
        dir=output_dir,
    ) as temporary_directory:
        # FIX: Keep the original string variable unchanged.
        temporary_path = Path(temporary_directory)

        new_parsed_path = temporary_path / "parsed.jsonl"
        new_rejected_path = temporary_path / "rejected.jsonl"

        infer_records(
            student,
            source_name,
            records,
            new_parsed_path,
            new_rejected_path,
            args.progress_every,
        )

        write_jsonl(
            parsed_path,
            old_parsed + read_jsonl(new_parsed_path),
        )

        write_jsonl(
            rejected_path,
            old_rejected + read_jsonl(new_rejected_path),
        )

    if not valid_semantic_jsonl(parsed_path, allow_empty=True):
        raise RuntimeError("Student inference produced invalid semantic documents.")

    write_manifest(
        manifest_path,
        {
            "source_keys": sorted([list(key) for key in current_keys]),
            "source_fingerprints": current_source_fingerprints,
            "prepared_sha256": file_sha256(prepared_path),
            "model_fingerprint": model_hash,
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
    """Convert validated semantic results to hybrid-miner documents."""

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

                destination.write(adapted.model_dump_json() + "\n")

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

    print(f"Adapted: {accepted}")
    print(f"Rejected: {rejected}")

    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")

    return accepted, rejected


def run_adapter(
    parsed_path: Path,
    output_dir: Path,
    progress_every: int,
) -> tuple[int, int]:
    """Reuse adapter outputs only when their source fingerprints match."""

    output_path = output_dir / "miner_documents.jsonl"
    rejected_path = output_dir / "adapter_rejected.jsonl"
    manifest_path = output_dir / "adapter_manifest.json"

    source_hash = file_sha256(parsed_path)
    previous = load_manifest(manifest_path)

    if (
        valid_jsonl(output_path, allow_empty=True)
        and valid_jsonl(rejected_path, allow_empty=True)
        and previous is not None
        and previous.get("parsed_sha256") == source_hash
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
    ) as temporary_directory:
        # FIX: Keep the original string variable unchanged.
        temporary_path = Path(temporary_directory)

        temporary_output = temporary_path / "miner_documents.jsonl"
        temporary_rejected = temporary_path / "adapter_rejected.jsonl"

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
            "parsed_sha256": source_hash,
            "output_sha256": file_sha256(output_path),
            "rejected_sha256": file_sha256(rejected_path),
            "accepted": accepted,
            "rejected": rejected,
            "updated_at": timestamp(),
        },
    )

    return accepted, rejected


# ---------------------------------------------------------------------
# Stage 8: Summary
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
    summary = {
        "source": source_name,
        "pipeline": "offline_teacher_student_to_miner_adapter",
        "updated_at": timestamp(),
        "paths": {
            "prepared_dataset": str(prepared_path),
            "teacher_documents": str(run_dir / "documents.jsonl"),
            "teacher_rejected": str(run_dir / "rejected.jsonl"),
            "student_pairs": str(run_dir / "pairs.jsonl"),
            "train": str(run_dir / "splits/train.jsonl"),
            "validation": str(run_dir / "splits/validation.jsonl"),
            "test": str(run_dir / "splits/test.jsonl"),
            "student_model": str(run_dir / "student"),
            "parsed_documents": str(parsed_path),
            "miner_documents": str(output_dir / "miner_documents.jsonl"),
            "adapter_rejected": str(output_dir / "adapter_rejected.jsonl"),
        },
        "counts": {
            "prepared": count_jsonl(prepared_path),
            "teacher_accepted": count_jsonl(run_dir / "documents.jsonl"),
            "teacher_rejected": count_jsonl(run_dir / "rejected.jsonl"),
            "student_pairs": count_jsonl(run_dir / "pairs.jsonl"),
            "train": count_jsonl(run_dir / "splits/train.jsonl"),
            "validation": count_jsonl(run_dir / "splits/validation.jsonl"),
            "test": count_jsonl(run_dir / "splits/test.jsonl"),
            "student_inference": count_jsonl(parsed_path),
            "miner_accepted": adapter_accepted,
            "adapter_rejected": adapter_rejected,
        },
    }

    write_manifest(output_dir / "summary.json", summary)

    return summary


# ---------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------


def run_source(
    args: argparse.Namespace,
    config: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    source_name = source["name"]
    args.selection_name = source_name

    data_dir = resolve_path(args.data_dir)
    output_root = resolve_path(args.output_dir)

    prepared_path = data_dir / "separate" / f"{source_name}.jsonl"
    output_dir = output_root / source_name
    run_dir = output_dir / "parser_e2e"

    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Dataset preparation.
    prepare_dataset(
        args,
        config,
        source,
        prepared_path,
        output_dir,
    )

    # STEP 2: Decide teacher extraction behavior.
    log_step(2, "REFERENCE SEMANTIC EXTRACTION")

    input_path, resume = prepare_teacher(
        args,
        source_name,
        prepared_path,
        output_dir,
    )

    # STEP 3-5: Invoke semantic E2E exactly once.
    log_step(3, "SEMANTIC CLI: PAIRS, SPLITS, TRAINING AND EVALUATION")

    model_dir = run_semantic_e2e(
        args,
        run_dir,
        output_dir,
        input_path,
        resume,
    )

    # Record the exact source content represented by the completed extraction.
    _, current_index = load_source_records(prepared_path, source_name)
    write_teacher_manifest(
        output_dir / "teacher_manifest.json",
        run_dir,
        current_index,
        teacher_fingerprint(args),
    )

    # STEP 6: Full-dataset student inference.
    if args.skip_parsing:
        parsed_path = output_dir / "parsed_documents.jsonl"

        if not valid_semantic_jsonl(parsed_path):
            raise FileNotFoundError(f"Completed inference not found: {parsed_path}")
    else:
        parsed_path = run_student_inference(
            args,
            source_name,
            prepared_path,
            output_dir,
            model_dir,
        )

    # STEP 7: Miner adaptation.
    accepted, rejected = run_adapter(
        parsed_path,
        output_dir,
        args.progress_every,
    )

    # STEP 8: Summary.
    summary = write_summary(
        output_dir,
        source_name,
        prepared_path,
        run_dir,
        parsed_path,
        accepted,
        rejected,
    )

    print(f"\nCompleted dataset: {source_name}")
    print(json.dumps(summary["counts"], indent=2))

    return {
        "source": source_name,
        **summary["counts"],
    }


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Configuration-driven offline dataset preparation, "
            "semantic teacher-student parsing and miner adaptation."
        )
    )

    parser.add_argument("--config", type=Path, default=DEFAULT_OFFLINE_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
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
    parser.add_argument("--selection")

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
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
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

    required_paths = (
        args.config,
        args.model_config,
        args.training_config,
        args.ontology_bundle,
    )

    for path in required_paths:
        resolved = resolve_path(path)

        if not resolved.is_file():
            raise FileNotFoundError(f"Required configuration not found: {resolved}")

    config = load_config(resolve_path(args.config))

    sources = select_sources(
        config,
        args.selection,
    )

    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "\n"
        + "=" * 68
        + "\nOFFLINE NEURO-SYMBOLIC PREPARATION PIPELINE"
        + "\n"
        + "=" * 68,
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
        results.append(
            run_source(
                args,
                config,
                source,
            )
        )

    fields = (
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

    totals = {field: sum(item[field] for item in results) for field in fields}

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
        "\n" + "=" * 68 + "\nOFFLINE PIPELINE COMPLETED" + "\n" + "=" * 68,
        flush=True,
    )

    print(f"Datasets: {len(results)}")

    for name, value in totals.items():
        print(f"{name}: {value}")

    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
