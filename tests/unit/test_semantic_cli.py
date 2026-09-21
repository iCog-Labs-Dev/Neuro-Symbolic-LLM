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


def write_checkpoint(path):
    """Save a tiny usable checkpoint without network access."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "dog": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=2, n_embd=8, n_layer=1, n_head=1, n_positions=16)
    )
    model.save_pretrained(path)


@pytest.mark.parametrize("existing", ["missing", "empty", "complete"])
def test_e2e_trains_on_splits_and_uses_held_out_inputs(monkeypatch, tmp_path, existing):
    import sys
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "teacher_structured.jsonl").write_text("", encoding="utf-8")
    label = FakeParser().generate_structured("Dog").model_dump_json()
    pairs = [{"input": str(i), "label": label} for i in range(20)]
    (data / "train_pairs.jsonl").write_text(
        "\n".join(json.dumps(pair) for pair in pairs), encoding="utf-8"
    )
    calls = []
    inferred = []
    model_dir = tmp_path / "model"
    if existing == "empty":
        model_dir.mkdir()
    elif existing == "complete":
        write_checkpoint(model_dir)

    def train(**kwargs):
        calls.append(kwargs)
        write_checkpoint(Path(kwargs["output_dir"]))

    monkeypatch.setitem(
        sys.modules, "parser.student.train", SimpleNamespace(train=train)
    )
    monkeypatch.setattr(
        cli,
        "DistilledSemanticParser",
        SimpleNamespace(
            from_pretrained=lambda _: SimpleNamespace(
                parse=lambda sentence: inferred.append(sentence)
                or FakeParser().parse(sentence)
            )
        ),
    )
    assert cli.main(["e2e", "--output-dir", str(tmp_path / "model")]) == 0
    if existing == "complete":
        assert calls == []
    else:
        assert len(calls) == 1
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


@pytest.mark.parametrize(
    "damage",
    [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "empty_weights",
        "bad_config",
        "bad_tokenizer",
    ],
)
def test_checkpoint_rejects_missing_or_unreadable_files(tmp_path, damage):
    write_checkpoint(tmp_path)
    if damage == "empty_weights":
        (tmp_path / "model.safetensors").write_bytes(b"")
    elif damage == "bad_config":
        (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    elif damage == "bad_tokenizer":
        (tmp_path / "tokenizer.json").write_text("null", encoding="utf-8")
    else:
        (tmp_path / damage).unlink()
    assert cli._checkpoint_error(tmp_path) is not None


def test_checkpoint_requires_every_indexed_shard(tmp_path):
    write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").rename(tmp_path / "shard-1.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight_a": "shard-1.safetensors",
                    "weight_b": "shard-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    assert "shard-2.safetensors" in cli._checkpoint_error(tmp_path)
    (tmp_path / "shard-2.safetensors").write_bytes(
        (tmp_path / "shard-1.safetensors").read_bytes()
    )
    assert cli._checkpoint_error(tmp_path) is None


@pytest.mark.parametrize(
    "index", [None, {}, {"weight_map": {}}, {"weight_map": {"a": None}}]
)
def test_checkpoint_rejects_malformed_index(tmp_path, index):
    write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    assert cli._checkpoint_error(tmp_path) is not None


def test_checkpoint_accepts_pytorch_weights(tmp_path):
    write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    import torch

    torch.save({"weight": torch.ones(1)}, tmp_path / "pytorch_model.bin")
    assert cli._checkpoint_error(tmp_path) is None


def test_e2e_skip_training_rejects_empty_checkpoint_before_teacher(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    assert cli.main(["e2e", "--skip-training", "--output-dir", str(model)]) == 1
    assert "complete model checkpoint is required" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_distill_rejects_empty_checkpoint(tmp_path, capsys):
    assert cli.main(["distill", "Dog", "--model-dir", str(tmp_path)]) == 1
    assert "complete model checkpoint is required" in capsys.readouterr().err


@pytest.fixture
def cached_e2e(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    splits = tmp_path / "data/splits"
    splits.mkdir(parents=True)
    for name in ("teacher_structured.jsonl", "train_pairs.jsonl"):
        (splits.parent / name).write_text("", encoding="utf-8")
    paths = [
        splits / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    ]
    for index, path in enumerate(paths):
        path.write_text(
            json.dumps(
                {
                    "input": str(index),
                    "label": FakeParser().generate_structured("Dog").model_dump_json(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    model_checks = []
    monkeypatch.setattr(
        cli, "_checkpoint_error", lambda path: model_checks.append(path)
    )
    monkeypatch.setattr(
        cli,
        "DistilledSemanticParser",
        SimpleNamespace(from_pretrained=lambda _: FakeParser()),
    )
    return paths, model_checks


@pytest.mark.parametrize("left,right", [(0, 1), (0, 2), (1, 2)])
def test_cached_splits_reject_overlap_before_model_use(cached_e2e, capsys, left, right):
    paths, model_checks = cached_e2e
    paths[right].write_text(
        json.dumps({"input": f" {left} ", "label": "{}"}), encoding="utf-8"
    )
    assert cli.main(["e2e"]) == 1
    assert "overlap" in capsys.readouterr().err
    assert model_checks == []


@pytest.mark.parametrize(
    "bad_record",
    [
        "{broken",
        "null",
        "[]",
        '{"input": null}',
        '{"input": "Dog", "label": null}',
        '{"input": " ", "label": "{}"}',
    ],
)
def test_cached_splits_report_invalid_record_location(cached_e2e, capsys, bad_record):
    paths, model_checks = cached_e2e
    paths[1].write_text("\n" + bad_record + "\n", encoding="utf-8")
    assert cli.main(["e2e"]) == 1
    assert "validation.jsonl:2" in capsys.readouterr().err
    assert model_checks == []


def test_valid_cached_splits_are_reused_without_changes(cached_e2e, capsys):
    paths, model_checks = cached_e2e
    original = [path.read_bytes() for path in paths]
    assert cli.main(["e2e"]) == 0
    assert "Using existing dataset splits" in capsys.readouterr().out
    assert model_checks
    assert [path.read_bytes() for path in paths] == original


@pytest.mark.parametrize(
    "prediction, matches",
    [
        ("(Has dog fur)", True),
        ("(Has fur dog)", False),
        ("(Not (Has dog fur))", False),
        ("(Inheritance dog animal)", False),
    ],
)
def test_e2e_reports_relation_accuracy(
    cached_e2e, monkeypatch, capsys, prediction, matches
):
    monkeypatch.setattr(
        FakeParser, "parse", lambda self, sentence: [parse_atom(prediction)]
    )
    assert cli.main(["e2e"]) == 0
    output = capsys.readouterr().out
    assert "Results: 1/1 valid" in output
    assert f"Exact relation match: {int(matches)}/1" in output
    assert ("[MATCH]" if matches else "[MISMATCH]") in output


def test_e2e_prediction_failure_does_not_report_completion(
    cached_e2e, monkeypatch, capsys
):
    def fail(self, sentence):
        raise ValueError("bad prediction")

    monkeypatch.setattr(FakeParser, "parse", fail)
    assert cli.main(["e2e"]) == 1
    output = capsys.readouterr()
    assert "Exact relation match: 0/1" in output.out
    assert "PIPELINE COMPLETE" not in output.out
    assert "prediction failures" in output.err


def test_e2e_rejects_invalid_expected_label(cached_e2e, capsys):
    paths, _ = cached_e2e
    paths[2].write_text(json.dumps({"input": "Dog", "label": "{}"}), encoding="utf-8")
    assert cli.main(["e2e"]) == 1
    assert "Invalid test label" in capsys.readouterr().err


def test_e2e_match_ignores_assertion_order_and_confidence(
    cached_e2e, monkeypatch, capsys
):
    import copy

    paths, _ = cached_e2e
    target = FakeParser().generate_structured("Dog").model_dump(mode="json")
    second = copy.deepcopy(target["assertions"][0])
    second["arguments"][0]["value"] = "cat"
    second["confidence"] = 0.5
    target["assertions"].append(second)
    paths[2].write_text(
        json.dumps({"input": "Dog and cat", "label": json.dumps(target)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        FakeParser,
        "parse",
        lambda self, sentence: [
            parse_atom("(Has cat fur)"),
            parse_atom("(Has dog fur)"),
        ],
    )
    assert cli.main(["e2e"]) == 0
    assert "Exact relation match: 1/1" in capsys.readouterr().out


def test_e2e_rejects_empty_test_set(cached_e2e, capsys):
    paths, _ = cached_e2e
    paths[2].write_text("", encoding="utf-8")
    assert cli.main(["e2e"]) == 1
    assert "No test examples" in capsys.readouterr().err
