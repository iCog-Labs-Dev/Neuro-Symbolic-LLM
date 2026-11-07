"""Command-line export of ontology-constrained semantic documents."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from parser.ontology.catalog import load_manifest_catalog, load_ontology_catalog
from parser.semantic.catalog_document_export import export_catalog_documents
from parser.semantic.cli import add_document_export_arguments
from parser.semantic.model_config import build_reference_semantic_parser
from parser.semantic.semantic_parser import (
    ModelGenerationError,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the compatibility entry point for semantic document export."""
    arguments = argparse.ArgumentParser(prog="semantic-document-export")
    arguments.add_argument("--config", type=Path)
    arguments.add_argument("--profile")
    add_document_export_arguments(arguments)
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Return nonzero if setup fails or any source record is rejected."""

    arguments = build_argument_parser().parse_args(argv)
    return run_export(arguments)


def run_export(arguments: argparse.Namespace) -> int:
    """Execute the shared parser export from an already parsed command."""
    try:
        if arguments.ontology_modules:
            catalog = load_ontology_catalog(modules=arguments.ontology_modules)
        else:
            manifest_arguments = (
                (arguments.ontology_bundle,)
                if arguments.ontology_bundle is not None
                else ()
            )
            catalog, _ = load_manifest_catalog(*manifest_arguments)
        parser = build_reference_semantic_parser(
            arguments.config,
            profile_name=arguments.profile,
            catalog=catalog,
            model_revision=arguments.model_revision,
            require_structured_output=arguments.require_structured_output,
        )
        result = export_catalog_documents(
            parser,
            input_path=arguments.input,
            output_path=arguments.output,
            rejected_path=arguments.rejected_output
            or arguments.output.with_name(f"{arguments.output.stem}.rejected.jsonl"),
            student_pairs_path=arguments.student_pairs_output,
            corpus_id=arguments.corpus_id,
            metadata_output=getattr(arguments, "metadata_output", None),
            include_metta=getattr(arguments, "include_metta", False),
            asserted_only=getattr(arguments, "asserted_only", False),
            resume=getattr(arguments, "resume", False),
            progress_every=getattr(arguments, "progress_every", 100),
        )
    except (ModelGenerationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Accepted: {result.accepted} -> {result.output_path}")
    print(f"Rejected: {result.rejected} -> {result.rejected_path}")
    if result.student_pairs_path is not None:
        print(f"Student pairs: {result.accepted} -> {result.student_pairs_path}")
    print(f"Source metadata: {result.metadata_path}")
    return 2 if result.rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
