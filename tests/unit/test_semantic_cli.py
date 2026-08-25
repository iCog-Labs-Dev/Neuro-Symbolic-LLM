"""Tests for the semantic-parser command-line interface."""

from __future__ import annotations

from pathlib import Path

from parser.grammar.atomese import parse_atom
from parser.semantic import cli
from parser.semantic.schema import SemanticParseResult


class FakeParser:
    provider_name = "fake"
    model_name = "fake-model"
    prompt_version = "2.0.0"

    def parse(self, sentence: str, context: str = ""):
        del sentence, context
        return [parse_atom("(Has dog fur)")]

    def generate_structured(self, sentence: str) -> SemanticParseResult:
        return SemanticParseResult.model_validate(
            {
                "assertions": [
                    {
                        "predicate": "Has",
                        "relation": None,
                        "arguments": [
                            {"value": "dog", "role": "owner"},
                            {"value": "fur", "role": "possessed"},
                        ],
                        "fallback": False,
                        "polarity": "positive",
                        "confidence": 0.99,
                        "source_span": sentence,
                        "alternatives": [],
                    }
                ]
            }
        )

    def render_metta(self, result: SemanticParseResult) -> list[str]:
        del result
        return ["(Has dog fur)"]

    def validate_rendered_metta(self, expressions: list[str]):
        return [parse_atom(expression) for expression in expressions]


def test_parse_command_prints_metta(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "build_reference_semantic_parser",
        lambda *args, **kwargs: FakeParser(),
    )

    exit_code = cli.main(["parse", "A dog has fur."])

    assert exit_code == 0
    assert capsys.readouterr().out == "(Has dog fur)\n"


def test_dataset_command_rejects_input_output_collision(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "sentences.txt"
    input_path.write_text("A dog has fur.\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "build-dataset",
            "--input",
            str(input_path),
            "--output",
            str(input_path),
        ]
    )

    assert exit_code == 1
    assert "paths must differ" in capsys.readouterr().err


def test_dataset_command_rejects_shared_output_paths(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "sentences.txt"
    output_path = tmp_path / "dataset.jsonl"
    input_path.write_text("A dog has fur.\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "build-dataset",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--rejected-output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert "paths must differ" in capsys.readouterr().err


def test_dataset_command_writes_both_outputs(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "build_reference_semantic_parser",
        lambda *args, **kwargs: FakeParser(),
    )
    input_path = tmp_path / "sentences.txt"
    output_path = tmp_path / "dataset.jsonl"
    input_path.write_text("A dog has fur.\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "build-dataset",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--include-metta",
        ]
    )

    assert exit_code == 0
    assert '"predicate": "Has"' in output_path.read_text(encoding="utf-8")
    rejected_path = tmp_path / "dataset.rejected.jsonl"
    assert rejected_path.read_text(encoding="utf-8") == ""
    assert "Accepted: 1" in capsys.readouterr().out


def test_help_explains_command_options(capsys) -> None:
    parser = cli.build_argument_parser()

    try:
        parser.parse_args(["build-dataset", "--help"])
    except SystemExit as error:
        assert error.code == 0

    help_output = capsys.readouterr().out
    assert "one sentence per line" in help_output
    assert "accepted structured records" in help_output
    assert "derived MeTTa expressions" in help_output
