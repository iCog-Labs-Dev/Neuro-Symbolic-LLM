"""Run the complete offline teacher-student semantic preparation pipeline.

Stages:
    1. Offline dataset preparation
    2. Teacher extraction and validation
    3. Student-pair generation and dataset splitting
    4. Student training
    5. Student semantic inference
    6. Miner document adaptation

The orchestrator reuses compatible artifacts, resumes valid interrupted
teacher extraction, and archives incompatible runs before restarting.

This pipeline does not execute hybrid-miner pattern mining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from offline.offline_dataloader import load_config, select_sources
from parser.semantic.schema import SemanticParseResult, SourceDocument
from symbolic_pipeline.miner_adapter import (
    MinerAdapterError,
    to_semantic_document,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OFFLINE_CONFIG = REPO_ROOT / "configs" / "offline_data.yaml"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "parser_config" / "model_backend.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "offline_corpus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "symbolic"

CHECKPOINT_NAME = "documents.jsonl.checkpoint.json"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def resolve_path(path: Path) -> Path:
    """Resolve paths relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def run_timestamp() -> str:
    """Return a unique UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def log_step(number: int, title: str) -> None:
    """Print one concise pipeline stage heading."""
    print(f"\nSTEP {number}: {title}", flush=True)


def count_jsonl(path: Path) -> int:
    """Count nonempty JSONL records."""
    if not path.is_file():
        return 0

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def file_sha256(path: Path, limit: int | None = None) -> str:
    """Calculate a SHA-256 fingerprint."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        remaining = path.stat().st_size if limit is None else limit

        while remaining:
            chunk = file.read(min(remaining, 1024 * 1024))

            if not chunk:
                raise ValueError(f"Incomplete file: {path}")

            digest.update(chunk)
            remaining -= len(chunk)

    return digest.hexdigest()


def valid_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> bool:
    """Check whether a JSONL file contains valid JSON."""

    if not path.is_file():
        return False

    found = False

    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                json.loads(line)
                found = True

    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    return found or allow_empty


def valid_semantic_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> bool:
    """Validate records against the semantic parse-result schema."""

    if not valid_jsonl(path, allow_empty=allow_empty):
        return False

    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    SemanticParseResult.model_validate_json(line)

    except (OSError, ValueError, ValidationError):
        return False

    return True


def valid_student_checkpoint(model_dir: Path) -> bool:
    """Check that a completed student checkpoint is available."""

    metadata_path = model_dir / "training_metadata.json"
    config_path = model_dir / "config.json"

    if not metadata_path.is_file() or not config_path.is_file():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(metadata, dict) or not isinstance(config, dict):
        return False

    weight_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )

    for name in weight_names:
        candidate = model_dir / name

        if not candidate.is_file():
            continue

        if name.endswith(".index.json"):
            try:
                index = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            weight_map = index.get("weight_map") if isinstance(index, dict) else None

            if not isinstance(weight_map, dict) or not weight_map:
                continue

            shards = set(weight_map.values())

            if not all(
                isinstance(shard, str) and shard and (model_dir / shard).is_file()
                for shard in shards
            ):
                continue

        return True

    return False


def checkpoint_fingerprint(model_dir: Path) -> str:
    """Fingerprint all files in a completed student checkpoint."""

    digest = hashlib.sha256()

    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.name.endswith((".partial", ".tmp")):
            continue

        digest.update(str(path.relative_to(model_dir)).encode())
        digest.update(file_sha256(path).encode())

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Run fingerprints
# ---------------------------------------------------------------------


def hash_project_files(paths: list[Path]) -> dict[str, str]:
    """Fingerprint the source files that determine parser behavior."""

    fingerprints = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required pipeline configuration not found: {path}"
            )

        fingerprints[str(path.relative_to(REPO_ROOT))] = file_sha256(path)

    return fingerprints


def build_run_fingerprint(
    args: argparse.Namespace,
    prepared_path: Path,
) -> dict[str, Any]:
    """Build the identity of a complete teacher-student run.

    Any relevant change creates a new run:
      - source data
      - teacher model configuration
      - prompt implementation
      - prompt YAML
      - ontology configuration
      - student training configuration
      - training method
      - acceptance threshold
    """

    prompt_files = [
        REPO_ROOT / "parser" / "semantic" / "catalog_prompt.py",
        REPO_ROOT / "configs" / "parser_config" / "general_prompt.yaml",
        REPO_ROOT / "configs" / "parser_config" / "discourse_prompt.yaml",
        REPO_ROOT / "configs" / "parser_config" / "parser_prompt.yaml",
    ]

    ontology_files = sorted((REPO_ROOT / "parser" / "ontology").rglob("*.yaml"))

    training_config = (
        resolve_path(args.training_config) if args.training_config else None
    )

    return {
        "source_sha256": file_sha256(prepared_path),
        "model_config_sha256": file_sha256(resolve_path(args.model_config)),
        "prompt_files": hash_project_files(prompt_files + ontology_files),
        "offline_config_sha256": file_sha256(resolve_path(args.config)),
        "training_config_sha256": (
            file_sha256(training_config)
            if training_config and training_config.is_file()
            else None
        ),
        "profile": args.profile,
        "method": args.method,
        "min_acceptance": args.min_acceptance,
    }


def load_manifest(path: Path) -> dict[str, Any] | None:
    """Read a JSON manifest."""

    if not path.is_file():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None


def write_manifest(
    path: Path,
    values: dict[str, Any],
) -> None:
    """Write a JSON manifest atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(values, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


# ---------------------------------------------------------------------
# Artifact recovery
# ---------------------------------------------------------------------


def archive_incomplete(
    paths: list[Path],
    label: str,
    recovery_root: Path,
) -> Path | None:
    """Preserve existing artifacts before rebuilding a stage."""

    existing = [path for path in paths if path.exists()]

    if not existing:
        return None

    recovery_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_dir = recovery_root / f"{label}-{run_timestamp()}"

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in existing:
        destination = archive_dir / path.name

        if destination.exists():
            destination = archive_dir / (f"{path.name}-{run_timestamp()}")

        shutil.move(
            str(path),
            str(destination),
        )

    print(
        "Preserved previous artifacts",
        flush=True,
    )

    return archive_dir


def inspect_teacher_checkpoint(
    run_dir: Path,
    prepared_path: Path,
) -> str:
    """Check the structural integrity of an interrupted teacher export.

    Returns:
        fresh
        zero_progress
        resumable
        completed
        corrupted
        incompatible
    """

    checkpoint = run_dir / CHECKPOINT_NAME

    documents = run_dir / "documents.jsonl"
    pairs = run_dir / "pairs.jsonl"
    rejected = run_dir / "rejected.jsonl"
    metadata = run_dir / "documents.metadata.jsonl"

    if (
        valid_semantic_jsonl(documents)
        and valid_jsonl(pairs)
        and valid_jsonl(rejected, allow_empty=True)
        and valid_jsonl(metadata, allow_empty=True)
    ):
        return "completed"

    if not checkpoint.is_file():
        return "fresh"

    try:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))

        if not isinstance(state, dict):
            return "corrupted"

        identity = state.get("identity")

        if not isinstance(identity, dict):
            return "corrupted"

        if identity.get("source_sha256") != file_sha256(prepared_path):
            return "incompatible"

        counters = (
            state.get("line_number"),
            state.get("accepted"),
            state.get("rejected"),
        )

        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counters
        ):
            return "corrupted"

        if state.get("complete") is True:
            return "corrupted"

        line_number, accepted, rejected_count = counters

        if line_number == 0 and accepted == 0 and rejected_count == 0:
            return "zero_progress"

        expected_names = {
            "documents",
            "rejected",
            "metadata",
            "pairs",
        }

        files = state.get("files")

        if not isinstance(files, dict) or set(files) != expected_names:
            return "corrupted"

        partial_paths = {
            "documents": run_dir / "documents.jsonl.partial",
            "rejected": run_dir / "rejected.jsonl.partial",
            "metadata": run_dir / "documents.metadata.jsonl.partial",
            "pairs": run_dir / "pairs.jsonl.partial",
        }

        for name, path in partial_paths.items():
            info = files.get(name)

            if not isinstance(info, dict) or not path.is_file():
                return "corrupted"

            expected_size = info.get("size")
            expected_hash = info.get("sha256")

            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
                or not isinstance(expected_hash, str)
            ):
                return "corrupted"

            if path.stat().st_size < expected_size:
                return "corrupted"

            if file_sha256(path, expected_size) != expected_hash:
                return "corrupted"

        return "resumable"

    except (OSError, ValueError, TypeError):
        return "corrupted"


def inspect_splits(run_dir: Path) -> str:
    """Check whether all three nonempty split files are available."""

    split_dir = run_dir / "splits"

    paths = [
        split_dir / "train.jsonl",
        split_dir / "validation.jsonl",
        split_dir / "test.jsonl",
    ]

    if not any(path.exists() for path in paths):
        return "missing"

    if all(valid_jsonl(path) for path in paths):
        return "valid"

    return "incomplete"


def recover_teacher_run(
    run_dir: Path,
    prepared_path: Path,
    recovery_root: Path,
) -> str:
    """Determine the correct action for teacher extraction."""

    status = inspect_teacher_checkpoint(
        run_dir,
        prepared_path,
    )

    if status == "completed":
        print(
            "Completed teacher extraction found.",
            flush=True,
        )
        return status

    if status == "resumable":
        print(
            "Valid teacher checkpoint found. Resuming extraction.",
            flush=True,
        )
        return status

    if status == "fresh":
        return status

    messages = {
        "zero_progress": (
            "Previous extraction processed zero records. " "Starting fresh."
        ),
        "corrupted": (
            "Teacher checkpoint or partial artifacts are corrupted. " "Starting fresh."
        ),
        "incompatible": (
            "Teacher checkpoint source differs from current dataset. " "Starting fresh."
        ),
    }

    print(
        messages[status],
        flush=True,
    )

    archive_incomplete(
        [run_dir],
        f"teacher-{status}",
        recovery_root,
    )

    return "fresh"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the offline pipeline CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Offline dataset, teacher-student semantic parser " "and miner adapter."
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
        "--profile",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--selection",
        type=str,
        default=None,
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
        "--skip-preparation",
        action="store_true",
        help="Require and reuse existing prepared data.",
    )

    parser.add_argument(
        "--skip-parsing",
        action="store_true",
        help="Require and reuse completed student inference.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Use an existing compatible student checkpoint.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly request teacher checkpoint recovery.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Start a fresh parser run while preserving the old run.",
    )

    parser.add_argument(
        "--method",
        choices=("lora", "full"),
        default="lora",
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--min-acceptance",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
    )

    return parser


# ---------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------


def run_command(
    command: list[str],
    *,
    quiet: bool = False,
) -> None:
    """Execute a command and stop immediately on failure."""

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline command failed with exit code {result.returncode}."
        )


def run_parser_command(
    command: list[str],
) -> tuple[int, str]:
    """Run parser CLI and capture its output for recovery decisions."""

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout or ""
    errors = result.stderr or ""

    if output:
        print(output, end="", flush=True)

    if errors:
        print(
            errors,
            end="",
            file=sys.stderr,
            flush=True,
        )

    return result.returncode, output + "\n" + errors


def run_loader_with_progress(
    command: list[str],
) -> None:
    """Display dataset acquisition progress on one updating terminal line."""

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Cannot capture dataset loader output.")

    percent_pattern = re.compile(r"(\d+(?:\.\d+)?)%")

    last_percent = None
    progress_active = False
    recent_output: list[str] = []
    buffer = ""

    try:
        while True:
            char = process.stdout.read(1)

            if not char:
                break

            if char not in ("\r", "\n"):
                buffer += char
                continue

            line = buffer.strip()
            buffer = ""

            if not line:
                continue

            recent_output.append(line)
            recent_output = recent_output[-20:]

            if "[DOWNLOAD_PROGRESS]" in line:
                match = percent_pattern.search(line)

                if match:
                    percent = int(float(match.group(1)))

                    if percent != last_percent:
                        print(
                            f"\r\033[2KDownloading {percent}%",
                            end="",
                            flush=True,
                        )

                        last_percent = percent
                        progress_active = True

                continue

            if "[DOWNLOAD_SKIPPED]" in line:
                if progress_active:
                    print(flush=True)
                    progress_active = False

                print(
                    "All selected books are already available. " "Skipping download.",
                    flush=True,
                )
                continue

            if "[DOWNLOAD_COMPLETE]" in line:
                if progress_active:
                    print(flush=True)

                progress_active = False
                continue

        if buffer.strip():
            recent_output.append(buffer.strip())

        returncode = process.wait()

    except KeyboardInterrupt:
        process.terminate()
        process.wait()

        if progress_active:
            print()

        raise

    if returncode != 0:
        if progress_active:
            print()

        print(
            "Dataset preparation failed.",
            file=sys.stderr,
        )

        for line in recent_output[-10:]:
            print(line, file=sys.stderr)

        raise RuntimeError(f"Dataset loader exited with code {returncode}.")

    if progress_active:
        print(flush=True)


# ---------------------------------------------------------------------
# Stage 1: Dataset preparation
# ---------------------------------------------------------------------


def prepare_dataset(
    args: argparse.Namespace,
    source_name: str,
    source_domain: str,
    prepared_path: Path,
) -> None:
    """Prepare a dataset or reuse valid existing records."""

    log_step(
        1,
        "OFFLINE DATASET PREPARATION",
    )

    print(f"Dataset: {source_name}")
    print(f"Type: {source_domain}")

    if not args.force and valid_jsonl(prepared_path):
        print(
            f"Reusing prepared dataset: {count_jsonl(prepared_path)} records",
            flush=True,
        )
        return

    if args.skip_preparation:
        raise FileNotFoundError(f"Valid prepared dataset not found: {prepared_path}")

    if prepared_path.exists():
        archive_incomplete(
            [prepared_path],
            f"{source_name}-dataset",
            prepared_path.parent / "recovery",
        )

    command = [
        sys.executable,
        "-u",
        "-m",
        "offline.offline_dataloader",
        "--config",
        str(resolve_path(args.config)),
        "--output-dir",
        str(resolve_path(args.data_dir)),
        "--selection",
        source_name,
        "--mode",
        "separate",
    ]

    run_loader_with_progress(command)

    if not valid_jsonl(prepared_path):
        raise RuntimeError(
            f"Dataset preparation did not produce valid records: {prepared_path}"
        )

    print(
        f"Prepared: {count_jsonl(prepared_path)} records",
        flush=True,
    )


# ---------------------------------------------------------------------
# Stage 2: Teacher extraction, validation, splits and training
# ---------------------------------------------------------------------


def build_e2e_command(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    prepared_path: Path | None,
    resume: bool,
    skip_training: bool,
) -> list[str]:
    """Build a parser e2e invocation."""

    command = [
        sys.executable,
        "-m",
        "parser.semantic",
        "--config",
        str(resolve_path(args.model_config)),
    ]

    if args.profile:
        command.extend(["--profile", args.profile])

    command.extend(
        [
            "e2e",
            "--output-dir",
            str(run_dir),
            "--method",
            args.method,
            "--min-acceptance",
            str(args.min_acceptance),
        ]
    )

    if prepared_path is not None:
        command.extend(
            [
                "--input",
                str(prepared_path),
            ]
        )

    if args.training_config:
        command.extend(
            [
                "--training-config",
                str(resolve_path(args.training_config)),
            ]
        )

    if skip_training:
        command.append("--skip-training")

    if resume:
        command.append("--resume")

    return command


def run_parser(
    args: argparse.Namespace,
    source_name: str,
    prepared_path: Path,
    output_dir: Path,
) -> Path:
    """Run or recover the teacher-student parser workflow."""

    parsed_path = output_dir / "parsed_documents.jsonl"
    rejected_path = output_dir / "parser_rejected.jsonl"
    inference_manifest = output_dir / "inference_manifest.json"

    run_dir = output_dir / "parser_e2e"

    # The manifest is deliberately outside parser_e2e.
    # The parser CLI rejects a new run when its output directory exists.
    run_manifest_path = output_dir / "parser_run_manifest.json"

    log_step(
        2,
        "TEACHER-STUDENT SEMANTIC PARSER",
    )

    if not valid_jsonl(prepared_path):
        raise ValueError(f"Prepared dataset is missing or invalid: {prepared_path}")

    if args.skip_parsing:
        if not valid_semantic_jsonl(parsed_path):
            raise FileNotFoundError(
                f"Completed semantic results not found: {parsed_path}"
            )

        print(
            f"Reusing {count_jsonl(parsed_path)} semantic results",
            flush=True,
        )
        return parsed_path

    # ---------------------------------------------------------
    # A. Calculate the current run identity.
    # ---------------------------------------------------------

    expected_fingerprint = build_run_fingerprint(
        args,
        prepared_path,
    )

    previous_manifest = load_manifest(run_manifest_path)

    # ---------------------------------------------------------
    # B. Determine whether the existing parser run is compatible.
    # ---------------------------------------------------------

    if args.force:
        archive_incomplete(
            [run_dir],
            f"{source_name}-forced-run",
            output_dir / "recovery",
        )

        run_manifest_path.unlink(missing_ok=True)

    elif run_dir.exists():
        if previous_manifest is None:
            print(
                "Existing parser run has no compatibility manifest. "
                "Starting a new run.",
                flush=True,
            )

            archive_incomplete(
                [run_dir],
                f"{source_name}-legacy-run",
                output_dir / "recovery",
            )

        elif previous_manifest.get("fingerprint") != expected_fingerprint:
            print(
                "Starting a new parser run.",
                flush=True,
            )

            archive_incomplete(
                [run_dir],
                f"{source_name}-fingerprint-mismatch",
                output_dir / "recovery",
            )

    # Write the manifest before running the parser so that an interrupted
    # extraction can be resumed on the next invocation.
    write_manifest(
        run_manifest_path,
        {
            "source": source_name,
            "fingerprint": expected_fingerprint,
            "updated_at": run_timestamp(),
        },
    )

    # ---------------------------------------------------------
    # C. Inspect teacher checkpoint.
    # ---------------------------------------------------------

    recovery_status = recover_teacher_run(
        run_dir,
        prepared_path,
        output_dir / "recovery",
    )

    # ---------------------------------------------------------
    # D. Inspect cached splits.
    # ---------------------------------------------------------

    split_status = inspect_splits(run_dir)

    if split_status == "incomplete":
        print(
            "Existing train/validation/test splits are incomplete. "
            "Preserving them and regenerating from completed teacher pairs.",
            flush=True,
        )

        archive_incomplete(
            [run_dir / "splits"],
            f"{source_name}-incomplete-splits",
            output_dir / "recovery",
        )

    splits_ready = split_status == "valid"

    # If completed teacher outputs exist but the splits are absent,
    # reuse the completed export and regenerate only the splits.

    # ---------------------------------------------------------
    # E. Determine whether student training can be skipped.
    # ---------------------------------------------------------

    model_dir = run_dir / "student"

    model_ready = valid_student_checkpoint(model_dir)

    skip_training = args.skip_training or model_ready

    if model_ready and not args.skip_training:
        print(
            "Completed student checkpoint found. Skipping training.",
            flush=True,
        )

    # ---------------------------------------------------------
    # F. Build the parser CLI invocation.
    # ---------------------------------------------------------

    # A valid cached split set must be reused without --input.
    # The parser CLI explicitly rejects --input with existing splits.
    if splits_ready:
        print(
            "Reusing compatible train/validation/test splits.",
            flush=True,
        )

        command = build_e2e_command(
            args,
            run_dir,
            prepared_path=None,
            resume=False,
            skip_training=skip_training,
        )

    else:
        should_resume = recovery_status in (
            "resumable",
            "completed",
        )

        if recovery_status == "resumable":
            print(
                "Resuming valid interrupted teacher extraction.",
                flush=True,
            )

        elif recovery_status == "completed":
            print(
                "Reusing completed teacher extraction and generating splits.",
                flush=True,
            )

        elif recovery_status == "zero_progress":
            raise RuntimeError("Zero-progress checkpoint was not archived correctly.")

        else:
            print(
                "Starting fresh teacher extraction.",
                flush=True,
            )

        command = build_e2e_command(
            args,
            run_dir,
            prepared_path=prepared_path,
            resume=should_resume,
            skip_training=skip_training,
        )

    # ---------------------------------------------------------
    # G. Run parser CLI.
    # ---------------------------------------------------------

    returncode, parser_output = run_parser_command(command)

    # If the parser itself detects an incompatible checkpoint,
    # archive the run and retry once without --resume.
    mismatch_message = (
        "Export checkpoint does not match source, outputs, model, or prompt"
    )

    if returncode != 0 and "--resume" in command and mismatch_message in parser_output:
        print(
            "\nParser checkpoint fingerprint mismatch. "
            "Archiving the old run and starting fresh.",
            flush=True,
        )

        archive_incomplete(
            [run_dir],
            f"{source_name}-checkpoint-mismatch",
            output_dir / "recovery",
        )

        command = build_e2e_command(
            args,
            run_dir,
            prepared_path=prepared_path,
            resume=False,
            skip_training=False,
        )

        returncode, parser_output = run_parser_command(command)

    if returncode != 0:
        print(
            "\nTeacher-student parser failed. "
            "Existing recovery artifacts have been preserved.",
            file=sys.stderr,
            flush=True,
        )

        raise RuntimeError(f"Parser CLI failed with exit code {returncode}.")

    # ---------------------------------------------------------
    # H. Verify the parser outputs.
    # ---------------------------------------------------------

    split_paths = [
        run_dir / "splits" / name
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
        )
    ]

    if not all(valid_jsonl(path) for path in split_paths):
        raise RuntimeError(
            "Parser did not produce nonempty, valid train, validation "
            "and test splits. Inspect the accepted student pairs and "
            "source-disjoint split requirements."
        )

    # ---------------------------------------------------------
    # I. Validate student checkpoint.
    # ---------------------------------------------------------

    if args.skip_training and not valid_student_checkpoint(model_dir):
        if valid_student_checkpoint(run_dir):
            model_dir = run_dir
        else:
            raise FileNotFoundError(
                f"No completed student checkpoint found in {run_dir}."
            )

    if not valid_student_checkpoint(model_dir):
        raise FileNotFoundError(f"Student checkpoint is incomplete: {model_dir}")

    # ---------------------------------------------------------
    # J. Student inference.
    # ---------------------------------------------------------

    from parser.ontology.catalog import load_manifest_catalog
    from parser.semantic.semantic_parser import DistilledSemanticParser

    catalog, _ = load_manifest_catalog()

    student = DistilledSemanticParser.from_pretrained(
        str(model_dir),
        catalog=catalog,
    )

    model_fingerprint = checkpoint_fingerprint(model_dir)
    prepared_fingerprint = file_sha256(prepared_path)

    # Reuse inference only when both input and student checkpoint match.
    inference_state = load_manifest(inference_manifest)

    if (
        valid_semantic_jsonl(parsed_path, allow_empty=True)
        and valid_jsonl(rejected_path, allow_empty=True)
        and inference_state is not None
        and inference_state.get("prepared_sha256") == prepared_fingerprint
        and inference_state.get("model_fingerprint") == model_fingerprint
        and inference_state.get("parsed_sha256") == file_sha256(parsed_path)
        and inference_state.get("rejected_sha256") == file_sha256(rejected_path)
    ):
        print(
            f"Reusing completed student inference: "
            f"{count_jsonl(parsed_path)} results",
            flush=True,
        )

        return parsed_path

    if parsed_path.exists() or rejected_path.exists() or inference_manifest.exists():
        archive_incomplete(
            [
                parsed_path,
                rejected_path,
                inference_manifest,
            ],
            f"{source_name}-inference",
            output_dir / "recovery",
        )

    log_step(
        3,
        "STUDENT INFERENCE",
    )

    total = count_jsonl(prepared_path)
    accepted = 0
    rejected = 0

    with (
        prepared_path.open("r", encoding="utf-8") as source,
        parsed_path.open("x", encoding="utf-8") as destination,
        rejected_path.open("x", encoding="utf-8") as rejected_file,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            document_id = f"{source_name}-{line_number}"

            try:
                record = json.loads(line)

                if isinstance(record, str):
                    record = {"text": record}

                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object or string.")

                text = record.get("source_text") or record.get("text")

                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Missing source text.")

                document_id = str(record.get("document_id") or document_id)

                document = SourceDocument(
                    document_id=document_id,
                    corpus_id=str(record.get("corpus_id") or source_name),
                    source_text=text,
                    context=str(record.get("context") or ""),
                )

                result = student.parse_document(document)

                destination.write(result.model_dump_json() + "\n")

                accepted += 1

            except Exception as error:
                rejected += 1

                rejected_file.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "document_id": document_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            processed = accepted + rejected

            if processed % args.progress_every == 0 or processed == total:
                percent = processed / total * 100 if total else 0

                print(
                    f"\rProcessed: {processed}/{total} " f"({percent:.0f}%)",
                    end="",
                    flush=True,
                )

    print()
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")

    if accepted == 0:
        raise RuntimeError("Student inference produced no valid semantic results.")

    write_manifest(
        inference_manifest,
        {
            "prepared_sha256": prepared_fingerprint,
            "model_fingerprint": model_fingerprint,
            "parsed_sha256": file_sha256(parsed_path),
            "rejected_sha256": file_sha256(rejected_path),
            "accepted": accepted,
            "rejected": rejected,
        },
    )

    return parsed_path


# ---------------------------------------------------------------------
# Stage 3: Miner adaptation
# ---------------------------------------------------------------------


def adapt_documents(
    parsed_path: Path,
    output_dir: Path,
    progress_every: int = 25,
) -> tuple[int, int]:
    """Convert validated semantic results into miner documents."""

    output_path = output_dir / "miner_documents.jsonl"
    rejected_path = output_dir / "adapter_rejected.jsonl"

    log_step(
        4,
        "MINER ADAPTATION",
    )

    total = count_jsonl(parsed_path)
    accepted = 0
    rejected = 0

    rejection_reasons: Counter[str] = Counter()

    with (
        parsed_path.open("r", encoding="utf-8") as source,
        output_path.open("x", encoding="utf-8") as destination,
        rejected_path.open("x", encoding="utf-8") as rejected_file,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            document_id = f"line-{line_number}"

            try:
                result = SemanticParseResult.model_validate_json(line)

                document_id = result.document.document_id

                semantic_document = to_semantic_document(result)

                destination.write(semantic_document.model_dump_json() + "\n")

                accepted += 1

            except (
                ValidationError,
                MinerAdapterError,
                ValueError,
            ) as error:
                rejected += 1

                reason = type(error).__name__
                rejection_reasons[reason] += 1

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

            if processed % progress_every == 0 or processed == total:
                percent = processed / total * 100 if total else 0

                print(
                    f"\rProcessed: {processed}/{total} " f"({percent:.0f}%)",
                    end="",
                    flush=True,
                )

    print()
    print(f"Adapted: {accepted}")
    print(f"Rejected: {rejected}")

    if rejection_reasons:
        print("Rejection reasons:")

        for reason, count in sorted(rejection_reasons.items()):
            print(f"  {reason}: {count}")

    return accepted, rejected


def run_adapter(
    parsed_path: Path,
    output_dir: Path,
    progress_every: int,
) -> tuple[int, int]:
    """Reuse compatible adapter outputs or rerun adaptation."""

    output_path = output_dir / "miner_documents.jsonl"
    rejected_path = output_dir / "adapter_rejected.jsonl"
    manifest_path = output_dir / "adapter_manifest.json"

    parsed_fingerprint = file_sha256(parsed_path)

    manifest = load_manifest(manifest_path)

    if (
        valid_jsonl(output_path, allow_empty=True)
        and valid_jsonl(rejected_path, allow_empty=True)
        and manifest is not None
        and manifest.get("parsed_sha256") == parsed_fingerprint
        and manifest.get("output_sha256") == file_sha256(output_path)
        and manifest.get("rejected_sha256") == file_sha256(rejected_path)
    ):
        accepted = count_jsonl(output_path)
        rejected = count_jsonl(rejected_path)

        print(
            f"Reusing completed miner adaptation: {accepted} documents",
            flush=True,
        )

        return accepted, rejected

    if output_path.exists() or rejected_path.exists() or manifest_path.exists():
        archive_incomplete(
            [
                output_path,
                rejected_path,
                manifest_path,
            ],
            "miner-adaptation",
            output_dir / "recovery",
        )

    accepted, rejected = adapt_documents(
        parsed_path,
        output_dir,
        progress_every,
    )

    write_manifest(
        manifest_path,
        {
            "parsed_sha256": parsed_fingerprint,
            "output_sha256": file_sha256(output_path),
            "rejected_sha256": file_sha256(rejected_path),
            "accepted": accepted,
            "rejected": rejected,
        },
    )

    return accepted, rejected


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def write_summary(
    output_dir: Path,
    *,
    source_name: str,
    prepared_path: Path,
    parsed_path: Path,
    accepted: int,
    adapter_rejected: int,
) -> None:
    """Write per-dataset processing statistics."""

    summary = {
        "source": source_name,
        "pipeline": "offline_teacher_student_to_miner_adapter",
        "prepared_dataset": str(prepared_path),
        "parser_e2e_dir": str(output_dir / "parser_e2e"),
        "parsed_documents": str(parsed_path),
        "parser_rejected": str(output_dir / "parser_e2e" / "rejected.jsonl"),
        "miner_documents": str(output_dir / "miner_documents.jsonl"),
        "adapter_rejected": str(output_dir / "adapter_rejected.jsonl"),
        "counts": {
            "prepared": count_jsonl(prepared_path),
            "parsed": count_jsonl(parsed_path),
            "parser_rejected": count_jsonl(
                output_dir / "parser_e2e" / "rejected.jsonl"
            ),
            "adapter_accepted": accepted,
            "adapter_rejected": adapter_rejected,
        },
    }

    write_manifest(
        output_dir / "summary.json",
        summary,
    )


# ---------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------


def run_source(
    args: argparse.Namespace,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete pipeline for one configured dataset."""

    source_name = source["name"]
    source_domain = source["domain"]

    data_dir = resolve_path(args.data_dir)
    output_root = resolve_path(args.output_dir)

    prepared_path = data_dir / "separate" / f"{source_name}.jsonl"

    output_dir = output_root / source_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Stage 1
    prepare_dataset(
        args,
        source_name,
        source_domain,
        prepared_path,
    )

    # Stages 2-5
    parsed_path = run_parser(
        args,
        source_name,
        prepared_path,
        output_dir,
    )

    # Stage 6
    accepted, adapter_rejected = run_adapter(
        parsed_path,
        output_dir,
        args.progress_every,
    )

    write_summary(
        output_dir,
        source_name=source_name,
        prepared_path=prepared_path,
        parsed_path=parsed_path,
        accepted=accepted,
        adapter_rejected=adapter_rejected,
    )

    return {
        "source": source_name,
        "prepared": count_jsonl(prepared_path),
        "parsed": count_jsonl(parsed_path),
        "adapter_accepted": accepted,
        "adapter_rejected": adapter_rejected,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = build_argument_parser().parse_args()

    if not 0 <= args.min_acceptance <= 1:
        raise ValueError("--min-acceptance must be between 0 and 1.")

    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1.")

    config = load_config(resolve_path(args.config))

    selected_sources = select_sources(
        config,
        args.selection,
    )

    print(
        "\nOFFLINE SEMANTIC PREPARATION PIPELINE",
        flush=True,
    )

    results = []

    for source in selected_sources:
        results.append(run_source(args, source))

    totals = {
        "datasets": len(results),
        "prepared": sum(item["prepared"] for item in results),
        "parsed": sum(item["parsed"] for item in results),
        "adapter_accepted": sum(item["adapter_accepted"] for item in results),
        "adapter_rejected": sum(item["adapter_rejected"] for item in results),
    }

    summary = {
        "pipeline": "offline_teacher_student_to_miner_adapter",
        "results": results,
        "totals": totals,
    }

    output_root = resolve_path(args.output_dir)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_root / "offline_pipeline_summary.json"

    write_manifest(
        summary_path,
        summary,
    )

    print("\nOFFLINE PIPELINE COMPLETED")
    print(f"Datasets: {totals['datasets']}")
    print(f"Prepared: {totals['prepared']}")
    print(f"Parsed: {totals['parsed']}")
    print(f"Miner documents: {totals['adapter_accepted']}")
    print(f"Rejected: {totals['adapter_rejected']}")
    print(f"Summary: {summary_path}")

    return 0 if totals["adapter_accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
