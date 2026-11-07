"""Occurrence selection, short sources and configuration-driven preparation."""

import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.prepare_alignment_labels import (
    collect_examples,
    exact_paths,
    parse_args,
    select_examples,
)
from stages.stage_A.a4.data import load_data, padded_source_blocks


class Tokenizer:
    pad_token_id = None
    eos_token_id = 0

    def __call__(self, text, return_offsets_mapping=False):
        pieces = list(re.finditer(r"\S+", text))
        result = {"input_ids": [int(piece.group()) for piece in pieces]}
        if return_offsets_mapping:
            result["offset_mapping"] = [piece.span() for piece in pieces]
        return result


def occurrence_file(tmp_path, patterns):
    path = tmp_path / "exact.jsonl"
    rows = []
    for template, start, end, text in patterns:
        rows.append(
            {
                "pattern": template,
                "exact_bindings": [
                    {
                        "corpus_id": "c",
                        "document_id": "a",
                        "evidence": [
                            {
                                "source_spans": [
                                    {"start": start, "end": end, "text": text}
                                ]
                            }
                        ],
                    }
                ],
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_short_document_retains_multiple_template_pairs_without_duplicate_rows(
    tmp_path,
):
    source = {("c", "a"): {"source_text": "1 2 3"}}
    path = occurrence_file(
        tmp_path, [("A", 0, 3, "1 2"), ("B", 0, 3, "1 2"), ("A", 0, 3, "1 2")]
    )
    rows, encoded, counts = collect_examples(
        source, Tokenizer(), [path], {"A", "B"}, lambda row: row["pattern"], 128
    )
    assert len(rows) == 2
    assert {row["template_id"] for row in rows} == {"A", "B"}
    assert all(row["token_index"] == 1 and row["window_end"] == 3 for row in rows)
    assert encoded["c", "a"]["input_ids"] == [1, 2, 3]
    assert counts["duplicate_position_template_pairs"] == 1


def test_occurrence_window_crosses_old_fixed_block_boundary(tmp_path):
    source = {("c", "a"): {"source_text": "1 2 3 4 5 6"}}
    path = occurrence_file(tmp_path, [("A", 6, 9, "4 5")])
    rows, _, _ = collect_examples(
        source, Tokenizer(), [path], {"A"}, lambda row: row["pattern"], 4
    )
    assert rows[0]["window_start"] == 1
    assert rows[0]["token_index"] == 3


def test_wide_evidence_is_reported_not_truncated(tmp_path):
    source = {("c", "a"): {"source_text": "1 2 3 4 5"}}
    path = occurrence_file(tmp_path, [("A", 0, 9, "1 2 3 4 5")])
    rows, _, counts = collect_examples(
        source, Tokenizer(), [path], {"A"}, lambda row: row["pattern"], 4
    )
    assert not rows
    assert counts["evidence_wider_than_window"] == 1


def test_changed_evidence_is_rejected(tmp_path):
    path = occurrence_file(tmp_path, [("A", 0, 3, "wrong")])
    with pytest.raises(ValueError, match="does not match"):
        collect_examples(
            {("c", "a"): {"source_text": "1 2 3"}},
            Tokenizer(),
            [path],
            {"A"},
            lambda row: row["pattern"],
            4,
        )


def test_selection_is_deterministic_and_has_no_replacement():
    rows = [
        {"template_id": name, "occurrence": index}
        for name in ("A", "B", "C")
        for index in range(4)
    ]
    selected = select_examples(rows, 10, 5, 3)
    assert selected == select_examples(rows, 10, 5, 3)
    assert (
        len(selected)
        == len({(row["template_id"], row["occurrence"]) for row in selected})
        == 10
    )
    assert {row["template_id"] for row in selected[:3]} == {"A", "B", "C"}


def test_selection_fails_below_minimum_and_reports_all_if_below_target():
    rows = [{"template_id": "A", "occurrence": index} for index in range(6)]
    assert len(select_examples(rows, 10, 5, 0)) == 6
    with pytest.raises(ValueError, match="minimum"):
        select_examples(rows, 10, 7, 0)


def test_shard_directory_excludes_failed_shards(tmp_path):
    for name, status in (
        ("documents-00000", "completed"),
        ("documents-00001", "failed"),
    ):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps({"status": status}))
        (folder / "exact.jsonl").write_text("{}\n")
    assert exact_paths([tmp_path]) == [
        (tmp_path / "documents-00000/exact.jsonl").resolve()
    ]


def test_evaluation_keeps_short_documents_and_tails(tmp_path):
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join(json.dumps({"source_text": text}) for text in ("1 2", "3 4 5 6 7 8"))
    )
    blocks, masks, _, metadata = padded_source_blocks(path, Tokenizer(), 4, 10, 10)
    assert blocks.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6], [6, 7, 8, 0]]
    assert masks.sum(axis=1).tolist() == [2, 4, 3]
    assert metadata["shape"] == [3, 4]


def test_config_uses_training_sequence_length_and_resolves_relative_paths(tmp_path):
    training = tmp_path / "training.yaml"
    training.write_text("training:\n  seq_len: 16\n  model: gpt2\n  seed: 2\n")
    config = tmp_path / "prepare.yaml"
    config.write_text(
        "\n".join(
            [
                "training_config: training.yaml",
                "documents: documents.jsonl",
                "exact: evidence",
                "bundle: bundle.json",
                "patterns: patterns.jsonl",
                "embeddings: embeddings.jsonl",
                "output: prepared.json",
                "target_examples: 10000",
                "minimum_examples: 5000",
            ]
        )
    )
    args = parse_args(["--config", str(config)])
    assert args.seq_len == 16 and args.seed == 2
    assert args.sources == tmp_path / "documents.jsonl"
    assert args.exact == [tmp_path / "evidence"]


def test_config_rejects_divergent_sequence_lengths(tmp_path):
    (tmp_path / "training.yaml").write_text(
        "training:\n  seq_len: 128\n  model: gpt2\n"
    )
    config = tmp_path / "prepare.yaml"
    config.write_text("training_config: training.yaml\nseq_len: 64\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


@pytest.fixture
def prepared(tmp_path):
    sources = tmp_path / "sources.jsonl"
    sources.write_text(
        json.dumps({"corpus_id": "c", "document_id": "a", "source_text": "1 2 3"})
        + "\n"
    )
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        json.dumps({"corpus_id": "c", "document_id": "b", "source_text": "4 5"}) + "\n"
    )
    arrays = tmp_path / "pairs.npz"
    np.savez_compressed(
        arrays,
        input_ids=np.array([[1, 2, 3, 0]] * 2, dtype=np.int32),
        token_mask=np.array([[True, True, True, False]] * 2),
        targets=np.array([[1, 0], [2, 1]], dtype=np.int32),
    )
    labels = []
    for template, position, start, end, text in (
        ("A", 1, 0, 3, "1 2"),
        ("B", 2, 2, 5, "2 3"),
    ):
        labels.append(
            {
                "corpus_id": "c",
                "document_id": "a",
                "template_id": template,
                "token_index": position,
                "window_start": 0,
                "window_end": 3,
                "source_spans": [{"start": start, "end": end, "text": text}],
            }
        )
    manifest = {
        "contract_version": "alignment-occurrences-v2",
        "seq_len": 4,
        "source_sha256": {"sources": hashlib.sha256(sources.read_bytes()).hexdigest()},
        "arrays_file": arrays.name,
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "bank_snapshot_sha256": "bank",
        "labels": labels,
        "bank_source_text_sha256": [hashlib.sha256(b"1 2 3").hexdigest()],
    }
    path = tmp_path / "pairs.json"
    args = SimpleNamespace(
        train_sources=sources,
        validation_sources=validation,
        alignment_labels=path,
        skip_test=True,
        max_train_rows=10,
        max_train_blocks=10,
        max_eval_rows=10,
        max_eval_blocks=10,
        minimum_examples=2,
    )
    bank = SimpleNamespace(
        input_snapshot_sha256="bank",
        records=[SimpleNamespace(template_id=name) for name in ("A", "B")],
    )
    return args, bank, manifest


def admit_prepared(prepared):
    args, bank, manifest = prepared
    args.alignment_labels.write_text(json.dumps(manifest))
    return load_data(args, Tokenizer(), bank, SimpleNamespace(seq_len=4, batch_size=1))


def test_occurrence_manifest_loads_short_contexts_and_labels(prepared):
    data = admit_prepared(prepared)
    assert data.train.shape == (2, 4)
    assert data.train_mask.sum(axis=1).tolist() == [3, 3]
    assert data.targets.tolist() == [[1, 0], [2, 1]]
    assert data.validation_mask.tolist() == [[True, True, False, False]]


@pytest.mark.parametrize(
    "mutation", ["bank", "arrays", "source", "minimum", "span", "padding", "overlap"]
)
def test_occurrence_admission_rejects_mismatched_inputs(prepared, mutation):
    args, _, manifest = prepared
    if mutation == "bank":
        manifest["bank_snapshot_sha256"] = "wrong"
    elif mutation == "arrays":
        manifest["arrays_sha256"] = "wrong"
    elif mutation == "source":
        manifest["source_sha256"]["sources"] = "wrong"
    elif mutation == "minimum":
        args.minimum_examples = 3
    elif mutation == "span":
        manifest["labels"][0]["source_spans"][0]["text"] = "wrong"
    elif mutation == "padding":
        manifest["labels"][0]["token_index"] = 3
    else:
        args.validation_sources.write_text(args.train_sources.read_text())
    with pytest.raises(ValueError):
        admit_prepared(prepared)
