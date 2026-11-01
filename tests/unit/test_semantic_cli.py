"""Tests for semantic-parser CLI routing and local checkpoint validation."""

import json
import sys
from types import ModuleType

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import cli, model_config
from parser.semantic.backends import CallableBackend
from parser.semantic.dataset import verify_split_overlap
from parser.semantic.schema import SEMANTIC_CONTRACT_VERSION
from tests.semantic_fixtures import pair, target
from tests.semantic_fixtures import result as fixture_result


@pytest.fixture
def backend(monkeypatch):
    calls = []

    def generate(*, prompt, model):
        calls.append(prompt)
        source = prompt.rsplit("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]
        return json.dumps(target(source))

    monkeypatch.setattr(
        model_config, "create_backend", lambda profile: CallableBackend(generate)
    )
    monkeypatch.setattr(
        cli, "command_catalog", lambda arguments: load_ontology_catalog()
    )
    return calls


def test_parse_defaults_to_current_parse_result_json(backend, capsys):
    assert cli.main(["parse", "Mira left."]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["contract_version"] == SEMANTIC_CONTRACT_VERSION
    assert result["document"]["source_text"] == "Mira left."
    assert result["extraction"]["assertions"][0]["predicate"] == "Event"
    assert len(backend) == 1


def test_parse_metta_is_a_derived_view(backend, capsys):
    assert cli.main(["parse", "Mira left.", "--format", "metta"]) == 0

    assert capsys.readouterr().out.strip() == "(Event e1)"
    assert len(backend) == 1


@pytest.mark.parametrize("command", ["build-dataset", "export-documents"])
def test_document_commands_delegate_with_current_target(tmp_path, monkeypatch, command):
    source = tmp_path / "sources.txt"
    output = tmp_path / "results.jsonl"
    source.write_text("Mira left.\n", encoding="utf-8")
    captured = []
    exporter = ModuleType("parser.semantic.catalog_document_cli")
    exporter.run_export = lambda arguments: captured.append(arguments) or 0
    monkeypatch.setitem(sys.modules, exporter.__name__, exporter)

    assert (
        cli.main(
            [
                command,
                "--input",
                str(source),
                "--output",
                str(output),
                "--student-pairs-output",
                str(tmp_path / "pairs.jsonl"),
                "--rejected-output",
                str(tmp_path / "rejected.jsonl"),
            ]
        )
        == 0
    )

    assert len(captured) == 1
    assert captured[0].target == SEMANTIC_CONTRACT_VERSION
    assert captured[0].input == source
    assert captured[0].output == output


def test_convert_to_pairs_uses_current_parse_result_jsonl(tmp_path, backend):
    source = tmp_path / "results.jsonl"
    output = tmp_path / "pairs.jsonl"
    source.write_text(fixture_result().model_dump_json() + "\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "convert-to-pairs",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    converted = json.loads(output.read_text(encoding="utf-8"))
    assert converted["contract_version"] == SEMANTIC_CONTRACT_VERSION
    assert converted["input"] == "Mira left."
    assert json.loads(converted["label"])["assertions"][0]["predicate"] == "Event"
    assert backend == []


def test_split_validates_pairs_and_writes_source_disjoint_sets(tmp_path, backend):
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        "".join(json.dumps(pair(f"Source {index}.")) + "\n" for index in range(10)),
        encoding="utf-8",
    )
    output = tmp_path / "splits"

    assert (
        cli.main(["split-pairs", "--input", str(source), "--output-dir", str(output)])
        == 0
    )

    splits = [
        [
            json.loads(line)
            for line in (output / name).read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    ]
    assert [len(records) for records in splits] == [8, 1, 1]
    verify_split_overlap(*splits)
    assert backend == []


@pytest.mark.parametrize("record", [None, [], {"input": "Dog", "label": "{}"}])
def test_split_refuses_malformed_or_old_pairs_before_writing(tmp_path, backend, record):
    source = tmp_path / "pairs.jsonl"
    source.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "splits"

    assert (
        cli.main(["split-pairs", "--input", str(source), "--output-dir", str(output)])
        == 1
    )
    assert not output.exists()
    assert backend == []


def test_unsupported_target_contract_is_not_an_active_choice():
    with pytest.raises(SystemExit):
        cli.build_argument_parser().parse_args(
            [
                "build-dataset",
                "--input",
                "source.txt",
                "--output",
                "docs.jsonl",
                "--target",
                "semantic-document-v1",
            ]
        )


def test_train_command_passes_current_training_controls(monkeypatch, tmp_path):
    from parser.student import train as training

    options = []
    monkeypatch.setattr(training, "train", lambda **kwargs: options.append(kwargs))
    assert (
        cli.main(
            [
                "train-student",
                "--train-file",
                str(tmp_path / "train.jsonl"),
                "--val-file",
                str(tmp_path / "validation.jsonl"),
                "--method",
                "lora",
            ]
        )
        == 0
    )
    assert options[0]["method"] == "lora"
    assert options[0]["val_file"].endswith("validation.jsonl")
    assert options[0]["output_dir"] is None


def _write_checkpoint(path):
    """Write a small local Transformers checkpoint without network access."""
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
def test_checkpoint_validation_rejects_missing_or_unreadable_files(tmp_path, damage):
    _write_checkpoint(tmp_path)
    if damage == "empty_weights":
        (tmp_path / "model.safetensors").write_bytes(b"")
    elif damage == "bad_config":
        (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    elif damage == "bad_tokenizer":
        (tmp_path / "tokenizer.json").write_text("null", encoding="utf-8")
    else:
        (tmp_path / damage).unlink()

    assert cli._checkpoint_error(tmp_path) is not None


def test_checkpoint_validation_requires_every_indexed_shard(tmp_path):
    _write_checkpoint(tmp_path)
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
def test_checkpoint_validation_rejects_malformed_index(tmp_path, index):
    _write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    assert cli._checkpoint_error(tmp_path) is not None


def test_checkpoint_validation_accepts_pytorch_weights(tmp_path):
    _write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    import torch

    torch.save({"weight": torch.ones(1)}, tmp_path / "pytorch_model.bin")

    assert cli._checkpoint_error(tmp_path) is None


def test_distill_requires_a_real_checkpoint(tmp_path):
    assert cli.main(["distill", "--model-dir", str(tmp_path), "Mira left."]) == 1


def test_e2e_requires_explicit_sources(tmp_path, backend, monkeypatch):
    exporter = ModuleType("parser.semantic.catalog_document_cli")
    exporter.run_export = lambda arguments: 0
    monkeypatch.setitem(sys.modules, exporter.__name__, exporter)
    assert cli.main(["e2e", "--output-dir", str(tmp_path / "run")]) == 1
    assert backend == []
