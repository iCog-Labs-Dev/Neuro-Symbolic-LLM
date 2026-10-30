"""One semantic-parser CLI for teacher extraction, distillation and inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from parser.semantic.dataset import (
    split_pairs,
    structured_to_pairs,
    validate_student_pair,
    verify_split_overlap,
    write_pairs_jsonl,
)
from parser.semantic.model_config import build_reference_semantic_parser
from parser.semantic.schema import SEMANTIC_CONTRACT_VERSION, SemanticParseResult
from parser.semantic.semantic_parser import (
    DistilledSemanticParser,
    ModelGenerationError,
    SemanticParseError,
)


def add_catalog_arguments(arguments: argparse.ArgumentParser) -> None:
    ontology = arguments.add_mutually_exclusive_group()
    ontology.add_argument(
        "--ontology-bundle",
        type=Path,
        help="versioned ontology manifest (default: semantic-extraction-v2)",
    )
    ontology.add_argument(
        "--ontology-module",
        action="append",
        dest="ontology_modules",
        metavar="NAME",
        help="select an ontology module; repeat for dependencies",
    )


def add_document_export_arguments(arguments: argparse.ArgumentParser) -> None:
    """Register source, output, and operational export options."""
    arguments.add_argument(
        "--input",
        required=True,
        type=Path,
        help="UTF-8 text with one sentence per line, or a JSONL source/pilot manifest",
    )
    arguments.add_argument(
        "--output", required=True, type=Path, help="accepted parse-result JSONL"
    )
    arguments.add_argument(
        "--rejected-output",
        type=Path,
        help="rejected records (default: <output-stem>.rejected.jsonl)",
    )
    arguments.add_argument(
        "--metadata-output",
        type=Path,
        help="producer metadata and derived views (default: <output-stem>.metadata.jsonl)",
    )
    arguments.add_argument(
        "--student-pairs-output",
        type=Path,
        help="optional validated source/document supervision pairs",
    )
    arguments.add_argument(
        "--include-metta",
        action="store_true",
        help="include derived MeTTa expressions in the metadata sidecar",
    )
    arguments.add_argument(
        "--asserted-only",
        action="store_true",
        help="render asserted-only MeTTa in the sidecar; parse results remain complete",
    )
    arguments.add_argument(
        "--resume", action="store_true", help="resume a matching interrupted export"
    )
    arguments.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="checkpoint/report interval (default: 100 records)",
    )
    arguments.add_argument(
        "--corpus-id", help="namespace for sentence or manifest inputs"
    )
    add_document_target_arguments(arguments)


def add_document_target_arguments(arguments: argparse.ArgumentParser) -> None:
    arguments.add_argument(
        "--model-revision", help="model revision recorded in document provenance"
    )
    arguments.add_argument(
        "--require-structured-output",
        action="store_true",
        help="require provider-supported JSON-schema decoding",
    )
    add_catalog_arguments(arguments)


def command_catalog(arguments: argparse.Namespace):
    from parser.ontology.catalog import load_manifest_catalog, load_ontology_catalog

    if getattr(arguments, "ontology_modules", None):
        return load_ontology_catalog(modules=arguments.ontology_modules)
    manifest = getattr(arguments, "ontology_bundle", None)
    return (
        load_manifest_catalog(manifest)[0]
        if manifest is not None
        else load_manifest_catalog()[0]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the semantic-parser command-line interface."""
    parser = argparse.ArgumentParser(
        prog="semantic-parser",
        description="Convert source text into validated semantic parse results.",
        epilog="Use 'semantic-parser <command> --help' for command options.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="model configuration YAML (default: project configuration)",
    )
    parser.add_argument(
        "--profile", help="named model profile (default: configured active profile)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-dataset", "export-documents"):
        command = commands.add_parser(
            name, help="generate validated parse results and optional student pairs"
        )
        add_document_export_arguments(command)
        command.add_argument(
            "--target",
            choices=(SEMANTIC_CONTRACT_VERSION,),
            default=SEMANTIC_CONTRACT_VERSION,
            help="semantic contract version of the exported results",
        )
    parse = commands.add_parser(
        "parse", help="parse one sentence using the configured model"
    )
    parse.add_argument("sentence", help="natural-language sentence to parse")
    parse.add_argument(
        "--context",
        default="",
        help="reference-only background for resolving participants",
    )
    parse.add_argument(
        "--format",
        choices=("json", "metta"),
        default="json",
        help="parse-result JSON or its derived MeTTa view",
    )
    add_catalog_arguments(parse)
    convert = commands.add_parser(
        "convert-to-pairs",
        help="convert parse-result JSONL to student training pairs",
    )
    convert.add_argument(
        "--input",
        required=True,
        type=Path,
        help="parse-result JSONL; adjacent metadata is retained",
    )
    convert.add_argument(
        "--output", required=True, type=Path, help="output input/label JSONL"
    )
    convert.add_argument(
        "--metadata-input",
        type=Path,
        help="explicit producer sidecar when not adjacent to input",
    )
    convert.add_argument(
        "--max-per-text",
        type=int,
        default=1,
        help="deprecated compatibility option; full document labels are retained",
    )
    convert.add_argument(
        "--include-confidence",
        action="store_true",
        help="include the first assertion's confidence as pair metadata",
    )
    add_catalog_arguments(convert)
    split = commands.add_parser(
        "split-pairs", help="create source-disjoint training, validation, and test sets"
    )
    split.add_argument(
        "--input", required=True, type=Path, help="validated student pairs JSONL"
    )
    split.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for train/validation/test JSONL",
    )
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    add_catalog_arguments(split)
    training = commands.add_parser(
        "train-student", help="train the student on validated document pairs"
    )
    training.add_argument(
        "--train-file", required=True, type=Path, help="training pairs JSONL"
    )
    training.add_argument(
        "--val-file", type=Path, help="optional separate validation pairs"
    )
    training.add_argument(
        "--training-config", type=Path, help="student training settings YAML"
    )
    training.add_argument(
        "--output-dir",
        type=Path,
        help="checkpoint directory (default: training configuration)",
    )
    training.add_argument(
        "--method",
        choices=("lora", "full"),
        default="full",
        help="training method (default: full)",
    )
    student = commands.add_parser(
        "distill", help="parse using a compatible document-trained student"
    )
    student.add_argument("sentence", help="natural-language sentence")
    student.add_argument("--context", default="", help="reference-only background")
    student.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/semantic_document_parser"),
        help="compatible checkpoint directory",
    )
    student.add_argument("--format", choices=("json", "metta"), default="json")
    add_catalog_arguments(student)
    e2e = commands.add_parser(
        "e2e",
        help="teacher extraction, source-disjoint splits, student training, and held-out evaluation",
    )
    sources = e2e.add_mutually_exclusive_group()
    sources.add_argument("--input", type=Path, help="sentence TXT or source JSONL")
    sources.add_argument("--sentences", nargs="+", help="sentences supplied directly")
    e2e.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/semantic_document_parser"),
        help="run directory; existing valid splits may be reused",
    )
    e2e.add_argument(
        "--training-config", type=Path, help="student training settings YAML"
    )
    e2e.add_argument("--method", choices=("lora", "full"), default="full")
    e2e.add_argument(
        "--test-sentence",
        help="additional explicit demonstration after held-out evaluation",
    )
    e2e.add_argument(
        "--skip-training",
        action="store_true",
        help="evaluate an existing compatible checkpoint without training",
    )
    e2e.add_argument(
        "--force",
        action="store_true",
        help="regenerate in a new child run directory without overwriting existing artifacts",
    )
    e2e.add_argument(
        "--min-acceptance",
        type=float,
        default=0.0,
        help="required accepted/input fraction; rejected records remain quarantined (default: 0)",
    )
    e2e.add_argument(
        "--resume",
        action="store_true",
        help="resume interrupted extraction before splitting",
    )
    e2e.add_argument(
        "--ontology-bundle",
        type=Path,
        help="manifest shared by extraction, training, and evaluation",
    )
    return parser


def _print_document(result: SemanticParseResult, output_format: str) -> None:
    """Print a parse result as JSON or as validated MeTTa expressions."""
    if output_format == "json":
        print(result.model_dump_json())
        return
    from parser.semantic.metta_renderer import render_metta, validate_rendered_metta

    expressions = render_metta(result)
    validate_rendered_metta(expressions)
    for expression in expressions:
        print(expression)


def _run_parse(arguments: argparse.Namespace) -> None:
    parser = build_reference_semantic_parser(
        arguments.config,
        profile_name=arguments.profile,
        catalog=command_catalog(arguments),
    )
    _print_document(
        parser.parse(arguments.sentence, context=arguments.context), arguments.format
    )


def _run_convert_to_pairs(arguments: argparse.Namespace) -> None:
    pairs = structured_to_pairs(
        str(arguments.input),
        str(arguments.output),
        max_per_text=arguments.max_per_text,
        include_confidence=arguments.include_confidence,
        catalog=command_catalog(arguments),
        metadata_file=arguments.metadata_input,
    )
    print(f"Student pairs: {len(pairs)} -> {arguments.output}")


def _run_split_pairs(arguments: argparse.Namespace) -> None:
    """Split student pairs into train, validation, and test sets."""

    input_path: Path = arguments.input
    output_dir: Path = arguments.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    if (
        len({path.resolve() for path in (input_path, train_path, val_path, test_path)})
        != 4
    ):
        raise ValueError("Input and split-output paths must differ")
    for path in (train_path, val_path, test_path):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing split: {path}")

    pairs = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                pair = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at " f"{input_path}:{line_number}: " f"{error.msg}"
                ) from error
            if not isinstance(pair, dict) or not isinstance(pair.get("input", ""), str):
                raise ValueError(
                    f"Invalid pair at "
                    f"{input_path}:{line_number}: "
                    "expected an object with a string input"
                )

            try:
                validate_student_pair(pair, catalog=command_catalog(arguments))
            except ValueError as error:
                raise ValueError(
                    f"Invalid pair at {input_path}:{line_number}: {error}"
                ) from error
            pairs.append(pair)

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        test_ratio=arguments.test_ratio,
        seed=arguments.seed,
    )

    verify_split_overlap(train_pairs, val_pairs, test_pairs)

    write_pairs_jsonl(train_pairs, train_path)
    write_pairs_jsonl(val_pairs, val_path)
    write_pairs_jsonl(test_pairs, test_path)

    print(f"Train:      {len(train_pairs)} -> {train_path}")
    print(f"Validation: {len(val_pairs)} -> {val_path}")
    print(f"Test:       {len(test_pairs)} -> {test_path}")


def _run_train_student(arguments: argparse.Namespace) -> None:
    """Train student model."""
    from parser.student.train import train

    print("Training student model...")
    print(f"  Training file: {arguments.train_file}")
    print(f"  Method: {arguments.method}")
    print(f"  Output: {arguments.output_dir}")

    training_options = {}
    if arguments.training_config is not None:
        training_options["config_path"] = str(arguments.training_config)
    train(
        train_file=str(arguments.train_file),
        val_file=str(arguments.val_file) if arguments.val_file else None,
        output_dir=str(arguments.output_dir) if arguments.output_dir else None,
        method=arguments.method,
        **training_options,
    )


def _checkpoint_error(model_dir: Path) -> str | None:
    """Check local checkpoint files without loading model weights into memory."""
    if not model_dir.is_dir():
        return "model directory is missing or is not a directory"

    try:
        from transformers import AutoConfig, AutoTokenizer

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

        if not isinstance(config, dict) or not config:
            return "config.json must contain a model configuration object"
        AutoConfig.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )

        AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )

        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ):
            candidate = model_dir / name

            if not candidate.exists():
                continue

            if name.endswith(".index.json"):
                index = json.loads(candidate.read_text(encoding="utf-8"))

                weight_map = (
                    index.get("weight_map") if isinstance(index, dict) else None
                )

                if not isinstance(weight_map, dict) or not weight_map:
                    return f"{name} has no valid weight map"

                shards = list(weight_map.values())

                if any(not isinstance(shard, str) or not shard for shard in shards):
                    return f"{name} contains invalid shard names"

                files = [model_dir / shard for shard in set(shards)]

            else:
                files = [candidate]

            for path in files:
                if not path.is_file():
                    return f"missing weight file: " f"{path.name}"

                with path.open("rb") as file:
                    if not file.read(1):
                        return f"empty weight file: " f"{path.name}"

            return None

    except Exception as error:
        return f"unreadable checkpoint: {error}"

    return "no model weights or weight index found"


def _require_checkpoint(model_dir: Path) -> None:
    error = _checkpoint_error(model_dir)

    if error is not None:
        raise ValueError(
            "A complete model checkpoint is required " f"at {model_dir}: {error}"
        )


def _run_distill(arguments: argparse.Namespace) -> None:
    _require_checkpoint(arguments.model_dir)
    parser = DistilledSemanticParser.from_pretrained(
        str(arguments.model_dir), catalog=command_catalog(arguments)
    )
    _print_document(
        parser.parse(arguments.sentence, context=arguments.context), arguments.format
    )


def _metta_preview(result: SemanticParseResult) -> str:
    """Return the MeTTa view of a result, or why it only has a JSON form."""
    from parser.semantic.metta_renderer import MettaRenderError, render_metta

    try:
        return " ".join(render_metta(result)) or "(no expressions)"
    except MettaRenderError as error:
        return f"(JSON only: {error})"


def _read_evaluation_pairs(path: Path, catalog) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
                validate_student_pair(pair, catalog=catalog)
            except ValueError as error:
                raise ValueError(f"Invalid pair at {path}:{number}: {error}") from error
            pairs.append(pair)
    return pairs


def _run_e2e(arguments: argparse.Namespace) -> int:
    """Reuse the extraction, split, training, and checkpoint interfaces."""
    from datetime import datetime, timezone

    from parser.semantic.catalog_document_cli import run_export
    from parser.semantic.catalog_student import validate_catalog_student_pair
    from parser.student.train import train

    if not 0 <= arguments.min_acceptance <= 1:
        raise ValueError("--min-acceptance must be in [0, 1]")
    if arguments.force and (arguments.skip_training or arguments.resume):
        raise ValueError("--force cannot be combined with --skip-training or --resume")
    root = arguments.output_dir
    if arguments.force and root.exists():
        root = root / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
        print(f"Fresh run: {root}")
    catalog = command_catalog(arguments)
    train_file, val_file, test_file = (
        root / "splits" / name
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    )
    cached = all(path.is_file() for path in (train_file, val_file, test_file))
    model_dir = root / "student"
    if arguments.skip_training:
        if not (model_dir / "training_metadata.json").exists():
            model_dir = root
        _require_checkpoint(model_dir)
    if cached and (arguments.input is not None or arguments.sentences):
        raise ValueError(
            "Existing splits are not regenerated from new input; use --force"
        )
    if cached:
        print(f"Using existing splits in {root / 'splits'}")
    else:
        print("Step 1: Teacher extraction")
        documents, pairs = root / "documents.jsonl", root / "pairs.jsonl"
        completed_export = arguments.resume and all(
            path.is_file() for path in (documents, pairs, root / "rejected.jsonl")
        )
        if not completed_export:
            if root.exists() and not arguments.resume:
                raise ValueError(
                    "An existing run requires valid cached splits, --resume, or --force"
                )
            if arguments.input is None and not arguments.sentences:
                raise ValueError("Supply --input or --sentences for extraction")
            root.mkdir(parents=True, exist_ok=True)
            input_path = arguments.input
            if input_path is None:
                input_path = root / "sources.jsonl"
                content = "".join(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    for text in arguments.sentences
                )
                if input_path.exists():
                    if (
                        not arguments.resume
                        or input_path.read_text(encoding="utf-8") != content
                    ):
                        raise ValueError(
                            "Direct source inputs differ from the existing run"
                        )
                else:
                    input_path.write_text(content, encoding="utf-8")
            status = run_export(
                argparse.Namespace(
                    config=arguments.config,
                    profile=arguments.profile,
                    input=input_path,
                    output=documents,
                    rejected_output=root / "rejected.jsonl",
                    student_pairs_output=pairs,
                    ontology_bundle=arguments.ontology_bundle,
                    ontology_modules=None,
                    model_revision=None,
                    require_structured_output=False,
                    corpus_id=None,
                    resume=arguments.resume,
                    progress_every=100,
                )
            )
            if status == 1:
                raise ValueError(
                    "Extraction did not complete; inspect the error and resume the run"
                )
        accepted = sum(
            1 for line in pairs.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        rejected = sum(
            1
            for line in (root / "rejected.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if not accepted or accepted / (accepted + rejected) < arguments.min_acceptance:
            raise ValueError(
                "Accepted data does not meet --min-acceptance or contains no examples"
            )
        print("\nStep 2: Split train / validation / test")
        _run_split_pairs(
            argparse.Namespace(
                input=pairs,
                output_dir=root / "splits",
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
                ontology_bundle=arguments.ontology_bundle,
                ontology_modules=None,
            )
        )
    train_pairs, val_pairs, test_pairs = (
        _read_evaluation_pairs(path, catalog)
        for path in (train_file, val_file, test_file)
    )
    verify_split_overlap(train_pairs, val_pairs, test_pairs)
    if not train_pairs or not test_pairs:
        raise ValueError("Need nonempty source-disjoint training and test sets")
    print("\nStep 3: Train student model")
    if arguments.skip_training:
        print(f"  Skipping training; using {model_dir}")
    else:
        from parser.ontology.catalog import load_manifest_catalog
        from parser.student.train import load_config

        training_config = (
            load_config(str(arguments.training_config))
            if arguments.training_config
            else load_config()
        )
        manifest = training_config["data"].get("ontology_bundle_manifest")
        if not manifest:
            raise ValueError("Training configuration requires an ontology manifest")
        training_catalog, _ = load_manifest_catalog(Path(manifest))
        if (
            training_catalog.versions != catalog.versions
            or training_catalog.signatures() != catalog.signatures()
        ):
            raise ValueError(
                "Training and extraction must use the same ontology manifest"
            )
        options = (
            {"config_path": str(arguments.training_config)}
            if arguments.training_config
            else {}
        )
        model_dir = Path(
            train(
                train_file=str(train_file),
                val_file=str(val_file),
                method=arguments.method,
                output_dir=str(model_dir),
                **options,
            )
        )
    print("\nStep 4: Evaluate on the held-out test set")
    student = DistilledSemanticParser.from_pretrained(str(model_dir), catalog=catalog)
    valid = exact = total = 0
    for pair in test_pairs:
        _, _, expected = validate_catalog_student_pair(pair, catalog=catalog)
        total += 1
        sentence = pair["input"]
        try:
            result = student.parse(sentence, context=pair.get("reference_context", ""))
        except (ValueError, RuntimeError) as error:
            print(f"  [FAIL] {sentence[:40]!r} -> {error}")
            continue
        valid += 1
        matches = result.extraction == expected.extraction
        exact += int(matches)
        verdict = "MATCH" if matches else "MISMATCH"
        print(f"  [{verdict}] {sentence[:40]!r} -> {_metta_preview(result)}")
    print(f"\nResults: {valid}/{total} valid")
    print(f"Exact extraction match: {exact}/{total} ({exact / total:.1%})")
    print("Exact matching is ID-sensitive; it is not a semantic-equivalence score.")
    if arguments.test_sentence:
        _print_document(student.parse(arguments.test_sentence), "json")
    print(f"\nModel: {model_dir}")
    return 0 if valid == total else 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.command in ("build-dataset", "export-documents"):
            from parser.semantic.catalog_document_cli import run_export

            return run_export(arguments)
        handlers = {
            "parse": _run_parse,
            "distill": _run_distill,
            "convert-to-pairs": _run_convert_to_pairs,
            "split-pairs": _run_split_pairs,
            "train-student": _run_train_student,
        }
        if arguments.command == "e2e":
            return _run_e2e(arguments)
        handlers[arguments.command](arguments)
    except (ModelGenerationError, OSError, SemanticParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
