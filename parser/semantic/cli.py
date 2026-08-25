"""Command-line interface for parsing and dataset generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from parser.semantic.dataset import SemanticDatasetBuilder
from parser.semantic.model_config import build_reference_semantic_parser
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the semantic-parser command-line interface."""
    parser = argparse.ArgumentParser(
        prog="semantic-parser",
        description="Convert natural-language text into validated semantics.",
        epilog="Use 'semantic-parser <command> --help' for command options.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="model configuration YAML (default: project configuration)",
    )
    parser.add_argument(
        "--profile",
        help="named model profile (default: active profile from configuration)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    parse_command = commands.add_parser(
        "parse",
        help="parse one sentence",
        description="Parse one sentence into validated MeTTa expressions.",
    )
    parse_command.add_argument("sentence", help="natural-language sentence to parse")
    parse_command.add_argument(
        "--context",
        default="",
        help="optional context used for reference resolution",
    )

    dataset_command = commands.add_parser(
        "build-dataset",
        help="build a text-to-JSON dataset from one sentence per input line",
    )
    dataset_command.add_argument(
        "--input",
        type=Path,
        required=True,
        help="UTF-8 text file containing one sentence per line",
    )
    dataset_command.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSONL path for accepted structured records",
    )
    dataset_command.add_argument(
        "--rejected-output",
        type=Path,
        help="JSONL path for rejected records (default: derived from --output)",
    )
    dataset_command.add_argument(
        "--include-metta",
        action="store_true",
        help="include derived MeTTa expressions for auditing",
    )
    return parser


def _build_semantic_parser(
    arguments: argparse.Namespace,
) -> ReferenceSemanticParser:
    return build_reference_semantic_parser(
        arguments.config,
        profile_name=arguments.profile,
    )


def _run_parse(arguments: argparse.Namespace) -> None:
    parser = _build_semantic_parser(arguments)
    for atom in parser.parse(arguments.sentence, context=arguments.context):
        print(atom)


def _run_dataset(arguments: argparse.Namespace) -> None:
    input_path: Path = arguments.input
    output_path: Path = arguments.output
    rejected_path: Path = arguments.rejected_output or output_path.with_name(
        f"{output_path.stem}.rejected{output_path.suffix}"
    )
    resolved_paths = {
        input_path.resolve(),
        output_path.resolve(),
        rejected_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError(
            "Input, accepted-output, and rejected-output paths must differ"
        )

    sentences = input_path.read_text(encoding="utf-8").splitlines()
    builder = SemanticDatasetBuilder(
        _build_semantic_parser(arguments),
        include_metta=arguments.include_metta,
    )
    accepted, rejected = builder.generate(sentences)
    builder.write_jsonl(accepted, output_path)
    builder.write_rejected_jsonl(rejected, rejected_path)
    print(f"Accepted: {len(accepted)} -> {output_path}")
    print(f"Rejected: {len(rejected)} -> {rejected_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return its process exit status."""
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.command == "parse":
            _run_parse(arguments)
        else:
            _run_dataset(arguments)
    except (ModelGenerationError, OSError, SemanticParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
