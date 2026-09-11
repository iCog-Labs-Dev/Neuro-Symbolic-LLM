"""Command-line interface for parsing, dataset generation, and student training."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from parser.semantic.dataset import SemanticDatasetBuilder, structured_to_pairs
from parser.semantic.model_config import build_reference_semantic_parser
from parser.semantic.semantic_parser import (
    DistilledSemanticParser,
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

    # ── parse command (Reference/Teacher) ─────────────────────────────────────

    parse_command = commands.add_parser(
        "parse",
        help="parse one sentence using reference parser (teacher)",
        description="Parse one sentence into validated MeTTa expressions.",
    )
    parse_command.add_argument("sentence", help="natural-language sentence to parse")
    parse_command.add_argument(
        "--context",
        default="",
        help="optional context used for reference resolution",
    )

    # ── build-dataset command ─────────────────────────────────────────────────

    dataset_command = commands.add_parser(
        "build-dataset",
        help="build a text-to-JSON dataset from sentences",
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

    # ── convert-to-pairs command ──────────────────────────────────────────────

    convert_command = commands.add_parser(
        "convert-to-pairs",
        help="convert structured JSON to training pairs for student",
    )
    convert_command.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input structured JSONL file",
    )
    convert_command.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with (input, label) pairs",
    )
    convert_command.add_argument(
        "--max-per-text",
        type=int,
        default=1,
        help="Max MeTTa expressions per text (default: 1)",
    )
    convert_command.add_argument(
        "--include-confidence",
        action="store_true",
        help="Include confidence scores in output",
    )

    # ── train-student command ─────────────────────────────────────────────────

    train_command = commands.add_parser(
        "train-student",
        help="train student model on teacher-generated pairs",
    )
    train_command.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Training pairs JSONL file",
    )
    train_command.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/distil_parser"),
        help="Output directory for trained model",
    )
    train_command.add_argument(
        "--method",
        choices=["lora", "full"],
        default="full",
        help="Training method: lora or full (default: full)",
    )

    # ── distill command (Student/Distilled) ──────────────────────────────────

    distill_command = commands.add_parser(
        "distill",
        help="parse using distilled (student) model",
        description="Parse a sentence using the fine-tuned student model.",
    )
    distill_command.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/distil_parser"),
        help="Model directory (default: models/distil_parser)",
    )
    distill_command.add_argument(
        "sentence",
        help="natural-language sentence to parse",
    )

    # ── end-to-end command ────────────────────────────────────────────────────

    e2e_command = commands.add_parser(
        "e2e",
        help="end-to-end pipeline: teacher → dataset → student → inference",
        description="Run the complete pipeline from teacher to distilled parser.",
    )
    e2e_command.add_argument(
        "--input",
        type=Path,
        help="UTF-8 text file containing one sentence per line (optional)",
    )
    e2e_command.add_argument(
        "--sentences",
        nargs="+",
        help="Sentences to parse directly from command line",
    )
    e2e_command.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/distil_parser"),
        help="Output directory for trained model",
    )
    e2e_command.add_argument(
        "--method",
        choices=["lora", "full"],
        default="full",
        help="Training method: lora or full (default: full)",
    )
    e2e_command.add_argument(
        "--test-sentence",
        help="Optional test sentence to parse after training",
    )
    e2e_command.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing model",
    )
    e2e_command.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of data",
    )

    return parser


# ── Command implementations ──────────────────────────────────────────────────


def _build_semantic_parser(arguments: argparse.Namespace) -> ReferenceSemanticParser:
    """Build a reference parser (teacher) from config."""
    return build_reference_semantic_parser(
        arguments.config, profile_name=arguments.profile
    )


def _run_parse(arguments: argparse.Namespace) -> None:
    """Run the reference parser (teacher)."""
    parser = _build_semantic_parser(arguments)
    for atom in parser.parse(arguments.sentence, context=arguments.context):
        print(atom)


def _run_dataset(arguments: argparse.Namespace) -> None:
    """Build dataset using teacher."""
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
        _build_semantic_parser(arguments), include_metta=arguments.include_metta
    )
    accepted, rejected = builder.generate(sentences)
    builder.write_jsonl(accepted, output_path)
    builder.write_rejected_jsonl(rejected, rejected_path)
    print(f"Accepted: {len(accepted)} -> {output_path}")
    print(f"Rejected: {len(rejected)} -> {rejected_path}")


def _run_convert_to_pairs(arguments: argparse.Namespace) -> None:
    """Convert structured JSON to (input, label) pairs."""
    input_path: Path = arguments.input
    output_path: Path = arguments.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Converting structured JSON to training pairs...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    pairs = structured_to_pairs(
        str(input_path),
        str(output_path),
        max_per_text=arguments.max_per_text,
        include_confidence=arguments.include_confidence,
    )

    print(f"Generated {len(pairs)} pairs -> {output_path}")


def _run_train_student(arguments: argparse.Namespace) -> None:
    """Train student model."""
    from parser.student.train import train

    print("Training student model...")
    print(f"  Training file: {arguments.train_file}")
    print(f"  Method: {arguments.method}")
    print(f"  Output: {arguments.output_dir}")

    train(
        train_file=str(arguments.train_file),
        output_dir=str(arguments.output_dir),
        method=arguments.method,
    )


def _run_distill(arguments: argparse.Namespace) -> None:
    """Parse using the distilled (student) model."""
    model_dir: Path = arguments.model_dir
    sentence: str = arguments.sentence

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    # Use the new DistilledSemanticParser
    distilled_parser = DistilledSemanticParser.from_pretrained(str(model_dir))

    try:
        atoms = distilled_parser.parse(sentence)
        for atom in atoms:
            print(atom)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _get_sentences(arguments: argparse.Namespace) -> list[str]:
    """Get sentences from input file, command line, or defaults."""
    if arguments.input and Path(arguments.input).exists():
        print(f"  Reading sentences from: {arguments.input}")
        sentences = Path(arguments.input).read_text(encoding="utf-8").splitlines()
        return [s.strip() for s in sentences if s.strip()]

    if arguments.sentences:
        print(f"  Using {len(arguments.sentences)} sentences from command line")
        return [s.strip() for s in arguments.sentences if s.strip()]

    print("  No input provided. Using default sentences.")
    return [
        "Dogs are animals.",
        "Birds can fly.",
        "The dog has fur.",
        "Rain causes flooding.",
        "Cats are mammals.",
        "The wheel is part of the car.",
        "Mary is happy.",
        "The cup is on the table.",
        "Smoking causes lung cancer.",
        "Children need education.",
        "The heart is part of the body.",
        "Fear causes stress.",
        "Fish can swim.",
        "The tree has leaves.",
        "Peter is sad.",
        "The book is on the shelf.",
        "Exercise improves health.",
        "Wolves are predators.",
        "The engine is part of the car.",
        "Ben loves animals.",
    ]


def _run_e2e(arguments: argparse.Namespace) -> None:
    """End-to-end pipeline: teacher → dataset → student → inference."""
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE")
    print("=" * 70)

    output_dir: Path = arguments.output_dir
    method: str = arguments.method
    force: bool = arguments.force
    skip_training: bool = arguments.skip_training

    # ── Get sentences ─────────────────────────────────────────────────────────

    sentences = _get_sentences(arguments)
    print(f"  Sentences: {len(sentences)}")

    # ── Step 1: Generate structured JSON from teacher ──────────────────────

    print("\nStep 1: Teacher generates structured JSON")
    structured_file = Path("data/teacher_structured.jsonl")
    structured_file.parent.mkdir(parents=True, exist_ok=True)

    if structured_file.exists() and not force:
        print(f"  Using existing structured data: {structured_file}")
    else:
        print(f"  Generating structured data for {len(sentences)} sentences...")
        parser = build_reference_semantic_parser()
        builder = SemanticDatasetBuilder(parser, include_metta=True)
        accepted, rejected = builder.generate(sentences)
        builder.write_jsonl(accepted, structured_file)
        rejected_path = structured_file.with_name(
            f"{structured_file.stem}.rejected{structured_file.suffix}"
        )
        builder.write_rejected_jsonl(rejected, rejected_path)
        print(f"  Accepted: {len(accepted)} -> {structured_file}")
        print(f"  Rejected: {len(rejected)} -> {rejected_path}")

    # ── Step 2: Convert to training pairs ──────────────────────────────────

    print("\nStep 2: Convert to training pairs")
    pairs_file = Path("data/train_pairs.jsonl")
    if pairs_file.exists() and not force:
        print(f"  Using existing pairs: {pairs_file}")
    else:
        if not structured_file.exists():
            raise FileNotFoundError(f"Structured file not found: {structured_file}")
        pairs = structured_to_pairs(
            str(structured_file),
            str(pairs_file),
            include_confidence=True,
        )
        print(f"  Generated {len(pairs)} pairs -> {pairs_file}")

    # ── Step 3: Train student model ─────────────────────────────────────────

    print("\nStep 3: Train student model")
    model_dir = output_dir
    if skip_training:
        print("  Skipping training (--skip-training)")
    elif model_dir.exists() and not force:
        print(f"  Using existing model: {model_dir}")
    else:
        from parser.student.train import train

        train(
            train_file=str(pairs_file),
            output_dir=str(model_dir),
            method=method,
        )

    # ── Step 4: Run inference ──────────────────────────────────────────────

    print("\nStep 4: Inference with distilled parser")
    if arguments.test_sentence:
        test_sentences = [arguments.test_sentence]
    else:
        test_sentences = sentences[:5] if len(sentences) >= 5 else sentences

    if not model_dir.exists():
        print(f"  Model not found: {model_dir}")
        return

    # Use the new DistilledSemanticParser
    distilled_parser = DistilledSemanticParser.from_pretrained(str(model_dir))

    print(f"\n  Model: {model_dir}")
    print(f"  Test sentences: {len(test_sentences)}")
    print()

    valid_count = 0
    for sentence in test_sentences:
        try:
            atoms = distilled_parser.parse(sentence)
            metta = " ".join(str(atom) for atom in atoms)
            status = "OK"
            valid_count += 1
        except Exception as e:
            metta = f"Error: {e}"
            status = "FAIL"

        print(f"  [{status}] '{sentence[:40]:40s}' -> {metta}")

    print(f"\nResults: {valid_count}/{len(test_sentences)} valid")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Model saved to: {model_dir}")
    print(
        f"    distilled_parser = DistilledSemanticParser.from_pretrained('{model_dir}')"
    )
    print("    result = parser.parse('Dogs are animals.')")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return its process exit status."""
    arguments = build_argument_parser().parse_args(argv)

    try:
        if arguments.command == "parse":
            _run_parse(arguments)
        elif arguments.command == "build-dataset":
            _run_dataset(arguments)
        elif arguments.command == "convert-to-pairs":
            _run_convert_to_pairs(arguments)
        elif arguments.command == "train-student":
            _run_train_student(arguments)
        elif arguments.command == "distill":
            _run_distill(arguments)
        elif arguments.command == "e2e":
            _run_e2e(arguments)
        else:
            print(f"Unknown command: {arguments.command}", file=sys.stderr)
            return 1
    except (ModelGenerationError, OSError, SemanticParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    main()
