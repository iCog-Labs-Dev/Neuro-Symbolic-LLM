"""Run the offline teacher-student semantic preparation pipeline.

Pipeline:
    1. Offline dataset preparation
    2. Teacher-student parser e2e
    3. Student semantic inference
    4. Miner document adaptation

The pipeline prepares miner-compatible semantic documents.
It does not execute hybrid-miner pattern mining.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
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


def resolve_path(path: Path) -> Path:
    """Resolve paths relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def log_step(number: int, title: str) -> None:
    """Print one concise pipeline stage heading."""
    print(f"\nSTEP {number}: {title}", flush=True)


def count_jsonl(path: Path) -> int:
    """Count nonempty JSONL records."""
    if not path.is_file():
        return 0

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline dataset, teacher-student parser and miner adapter."
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
        help="Reuse prepared dataset files.",
    )
    parser.add_argument(
        "--skip-parsing",
        action="store_true",
        help="Reuse existing student semantic results.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate using an existing student checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted teacher extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start a fresh parser run in a new output directory.",
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


def run_command(
    command: list[str],
    *,
    quiet: bool = False,
    accepted_exit_codes: tuple[int, ...] = (0,),
) -> int:
    """Run a command without printing its full invocation."""

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
    )

    if result.returncode not in accepted_exit_codes:
        raise RuntimeError(
            f"Pipeline command failed with exit code {result.returncode}."
        )

    return result.returncode


def run_loader_with_progress(command: list[str]) -> None:
    """Display SPGC download progress on one terminal line."""

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
    recent_output = []
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


def prepare_dataset(
    args: argparse.Namespace,
    source_name: str,
    source_domain: str,
    prepared_path: Path,
) -> None:
    """Prepare one selected offline dataset."""

    log_step(1, "OFFLINE DATASET PREPARATION")
    print(f"Dataset: {source_name}")
    print(f"Type: {source_domain}")

    if args.skip_preparation:
        if not prepared_path.is_file():
            raise FileNotFoundError(f"Prepared dataset not found: {prepared_path}")

        print(f"Reusing prepared data: {count_jsonl(prepared_path)} records")
        return

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

    if not prepared_path.is_file():
        raise FileNotFoundError(f"Dataset loader did not create: {prepared_path}")

    print(f"Prepared: {count_jsonl(prepared_path)} records")


def run_parser(
    args: argparse.Namespace,
    source_name: str,
    prepared_path: Path,
    output_dir: Path,
) -> Path:
    """Run the parser e2e CLI and export student predictions."""

    parsed_path = output_dir / "parsed_documents.jsonl"
    rejected_path = output_dir / "parser_rejected.jsonl"
    run_dir = output_dir / "parser_e2e"

    log_step(2, "TEACHER-STUDENT SEMANTIC PARSER")

    if args.skip_parsing:
        if not parsed_path.is_file():
            raise FileNotFoundError(f"Parsed documents not found: {parsed_path}")

        print(f"Reusing {count_jsonl(parsed_path)} semantic results")
        return parsed_path

    if args.force:
        raise ValueError(
            "Use a new --output-dir for a fresh offline run. "
            "The parser's --force option creates a separate child "
            "directory that this orchestrator does not select."
        )

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
            "--input",
            str(prepared_path),
            "--output-dir",
            str(run_dir),
            "--method",
            args.method,
            "--min-acceptance",
            str(args.min_acceptance),
        ]
    )

    if args.training_config:
        command.extend(
            [
                "--training-config",
                str(resolve_path(args.training_config)),
            ]
        )

    if args.skip_training:
        command.append("--skip-training")

    if args.resume:
        command.append("--resume")

    # Preserve parser CLI output, including student training
    # and held-out evaluation metrics.
    run_command(command)

    model_dir = run_dir / "student"

    if args.skip_training and not (model_dir / "training_metadata.json").exists():
        model_dir = run_dir

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Student checkpoint directory not found: {model_dir}")

    from parser.ontology.catalog import load_manifest_catalog
    from parser.semantic.semantic_parser import DistilledSemanticParser

    catalog, _ = load_manifest_catalog()

    student = DistilledSemanticParser.from_pretrained(
        str(model_dir),
        catalog=catalog,
    )

    if parsed_path.exists() or rejected_path.exists():
        raise FileExistsError(
            "Student inference outputs already exist. "
            "Use --skip-parsing or a new output directory."
        )

    log_step(3, "STUDENT INFERENCE")

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

    return parsed_path


def adapt_documents(
    parsed_path: Path,
    output_dir: Path,
    progress_every: int = 25,
) -> tuple[int, int]:
    """Convert validated student results to miner documents."""

    output_path = output_dir / "miner_documents.jsonl"
    rejected_path = output_dir / "adapter_rejected.jsonl"

    if output_path.exists() or rejected_path.exists():
        raise FileExistsError(
            "Miner adapter outputs already exist. " "Use a new output directory."
        )

    log_step(4, "MINER ADAPTATION")

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


def write_summary(
    output_dir: Path,
    *,
    source_name: str,
    prepared_path: Path,
    parsed_path: Path,
    accepted: int,
    adapter_rejected: int,
) -> None:
    """Save per-dataset processing statistics."""

    summary = {
        "source": source_name,
        "pipeline": "offline_teacher_student_to_miner_adapter",
        "prepared_dataset": str(prepared_path),
        "parser_e2e_dir": str(output_dir / "parser_e2e"),
        "parsed_documents": str(parsed_path),
        "parser_rejected": str(output_dir / "parser_rejected.jsonl"),
        "miner_documents": str(output_dir / "miner_documents.jsonl"),
        "adapter_rejected": str(output_dir / "adapter_rejected.jsonl"),
        "counts": {
            "prepared": count_jsonl(prepared_path),
            "parsed": count_jsonl(parsed_path),
            "parser_rejected": count_jsonl(output_dir / "parser_rejected.jsonl"),
            "adapter_accepted": accepted,
            "adapter_rejected": adapter_rejected,
        },
    }

    summary_path = output_dir / "summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_source(
    args: argparse.Namespace,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete offline pipeline for one dataset."""

    source_name = source["name"]
    source_domain = source["domain"]

    data_dir = resolve_path(args.data_dir)
    output_root = resolve_path(args.output_dir)

    prepared_path = data_dir / "separate" / f"{source_name}.jsonl"

    output_dir = output_root / source_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_dataset(
        args,
        source_name,
        source_domain,
        prepared_path,
    )

    parsed_path = run_parser(
        args,
        source_name,
        prepared_path,
        output_dir,
    )

    accepted, adapter_rejected = adapt_documents(
        parsed_path,
        output_dir,
        progress_every=args.progress_every,
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

    print("\nOFFLINE SEMANTIC PREPARATION PIPELINE", flush=True)

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
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "offline_pipeline_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nOFFLINE PIPELINE COMPLETED")
    print(f"Prepared: {totals['prepared']}")
    print(f"Parsed: {totals['parsed']}")
    print(f"Miner documents: {totals['adapter_accepted']}")
    print(f"Rejected: {totals['adapter_rejected']}")

    return 0 if totals["adapter_accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
