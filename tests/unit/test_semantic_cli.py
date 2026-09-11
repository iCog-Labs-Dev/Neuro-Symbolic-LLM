"""Tests for the semantic-parser command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("filename", ["train.jsonl", "validation.jsonl", "test.jsonl"])
def test_split_rejects_input_output_collision(tmp_path, capsys, filename):
    source = tmp_path / filename
    content = '{"input": "hello", "label": "{}"}\n'
    source.write_text(content)
    assert (
        cli.main(["split-pairs", "--input", str(source), "--output-dir", str(tmp_path)])
        == 1
    )
    assert "paths must differ" in capsys.readouterr().err
    assert source.read_text() == content


@pytest.mark.parametrize("content", ["null", '{"input": null}', "{broken"])
def test_split_reports_invalid_line_without_writing(tmp_path, capsys, content):
    source = tmp_path / "pairs.jsonl"
    source.write_text("\n" + content + "\n")
    output = tmp_path / "splits"
    assert (
        cli.main(["split-pairs", "--input", str(source), "--output-dir", str(output)])
        == 1
    )
    assert f"{source}:2" in capsys.readouterr().err
    assert not output.exists()


def test_split_command_writes_disjoint_files(tmp_path):
    source = tmp_path / "pairs.jsonl"
    pairs = [{"input": str(i), "label": "{}"} for i in range(10)]
    source.write_text("\n".join(json.dumps(pair) for pair in pairs + pairs))
    output = tmp_path / "splits"
    assert (
        cli.main(["split-pairs", "--input", str(source), "--output-dir", str(output)])
        == 0
    )
    splits = [
        [json.loads(line) for line in (output / name).read_text().splitlines()]
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    ]
    assert [len(split) for split in splits] == [8, 1, 1]
    assert sorted(pair["input"] for split in splits for pair in split) == [
        str(i) for i in range(10)
    ]


def test_e2e_trains_on_splits_and_uses_held_out_inputs(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "teacher_structured.jsonl").write_text("", encoding="utf-8")
    pairs = [{"input": str(i), "label": "{}"} for i in range(20)]
    (data / "train_pairs.jsonl").write_text(
        "\n".join(json.dumps(pair) for pair in pairs), encoding="utf-8"
    )
    calls = []
    inferred = []

    def train(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_dir"]).mkdir()

    monkeypatch.setitem(
        sys.modules, "parser.student.train", SimpleNamespace(train=train)
    )
    monkeypatch.setattr(
        cli,
        "DistilledSemanticParser",
        SimpleNamespace(
            from_pretrained=lambda _: SimpleNamespace(
                parse=lambda sentence: inferred.append(sentence) or []
            )
        ),
    )
    assert cli.main(["e2e", "--output-dir", str(tmp_path / "model")]) == 0
    assert calls[0]["train_file"] == "data/splits/train.jsonl"
    assert calls[0]["val_file"] == "data/splits/validation.jsonl"
    test_pairs = [
        json.loads(line)
        for line in (data / "splits/test.jsonl").read_text().splitlines()
    ]
    assert inferred == [pair["input"] for pair in test_pairs]


def test_e2e_missing_input_returns_error(tmp_path, capsys):
    assert cli.main(["e2e", "--input", str(tmp_path / "missing.txt")]) == 1
    assert "missing.txt" in capsys.readouterr().err


def test_convert_rejects_input_output_collision(tmp_path, capsys):
    source = tmp_path / "pairs.jsonl"
    source.write_text("original", encoding="utf-8")
    assert (
        cli.main(["convert-to-pairs", "--input", str(source), "--output", str(source)])
        == 1
    )
    assert source.read_text() == "original"
    assert "paths must differ" in capsys.readouterr().err
