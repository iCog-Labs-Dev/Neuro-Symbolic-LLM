"""Write hybrid-miner SemanticDocument JSONL from parser result JSONL.

Each SemanticParseResult line goes through the miner adapter and the ontology
bundle check. A result the miner contract cannot express is written to the
rejected file with its reason, never dropped silently. ``--bundle-output``
also writes the bundle used for the check, for the miner to load.

    python -m symbolic_pipeline --input results.jsonl --output documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hybrid_miner.ontology.bundle import OntologyBundle

from parser.ontology.catalog import DEFAULT_BUNDLE_MANIFEST
from parser.semantic.schema import SemanticParseResult
from symbolic_pipeline.miner_adapter import to_semantic_document
from symbolic_pipeline.miner_bundle import load_catalog_bundle


@dataclass(frozen=True, slots=True)
class DocumentExportResult:
    """Counts and paths for one completed conversion."""

    accepted: int
    rejected: int
    output_path: Path
    rejected_path: Path
    bundle_path: Path | None = None


def export_semantic_documents(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    *,
    bundle: OntologyBundle,
    bundle_path: Path | None = None,
) -> DocumentExportResult:
    """Convert every parse result and validate it against the ontology bundle.

    With ``bundle_path``, the bundle itself is written there as well.
    """
    outputs = [output_path, rejected_path]
    if bundle_path is not None:
        outputs.append(bundle_path)
    paths = (input_path, *outputs)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input and output paths must all differ")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    if bundle_path is not None:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with bundle_path.open("x", encoding="utf-8") as file:
            file.write(bundle.canonical_json() + "\n")

    accepted = rejected = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("x", encoding="utf-8") as documents,
        rejected_path.open("x", encoding="utf-8") as failures,
    ):
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                document = to_semantic_document(
                    SemanticParseResult.model_validate_json(line)
                )
                bundle.validate_document(document)
            except ValueError as error:
                failures.write(
                    json.dumps(
                        {"line_number": number, "error": str(error)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
            else:
                documents.write(document.canonical_json() + "\n")
                accepted += 1
    return DocumentExportResult(
        accepted, rejected, output_path, rejected_path, bundle_path
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Return 1 on setup errors and 2 when any result is rejected."""
    arguments = argparse.ArgumentParser(
        prog="symbolic-pipeline",
        description="Convert parser result JSONL into hybrid-miner documents.",
    )
    arguments.add_argument("--input", required=True, type=Path)
    arguments.add_argument("--output", required=True, type=Path)
    arguments.add_argument(
        "--rejected-output",
        type=Path,
        help="rejected results (default: <output-stem>.rejected.jsonl)",
    )
    arguments.add_argument(
        "--ontology-bundle",
        type=Path,
        default=DEFAULT_BUNDLE_MANIFEST,
        help="ontology bundle manifest used to validate documents",
    )
    arguments.add_argument(
        "--bundle-output",
        type=Path,
        help="also write the ontology bundle used to validate documents",
    )
    options = arguments.parse_args(argv)
    try:
        _, bundle = load_catalog_bundle(options.ontology_bundle)
        result = export_semantic_documents(
            options.input,
            options.output,
            options.rejected_output
            or options.output.with_name(f"{options.output.stem}.rejected.jsonl"),
            bundle=bundle,
            bundle_path=options.bundle_output,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Accepted: {result.accepted} -> {result.output_path}")
    print(f"Rejected: {result.rejected} -> {result.rejected_path}")
    if result.bundle_path is not None:
        print(f"Ontology bundle: {result.bundle_path}")
    return 2 if result.rejected else 0
