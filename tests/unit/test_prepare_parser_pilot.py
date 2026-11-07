"""Tests for parser pilot-corpus preparation."""

from __future__ import annotations

import json
import random

import pytest

from scripts.prepare_parser_pilot import (
    build_manifest,
    collect_candidates,
    iter_source_texts,
    normalize_text,
    split_sentences,
    write_manifest,
)


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  Alice   opened\n the door. ") == "Alice opened the door."


def test_split_sentences() -> None:
    assert split_sentences("Alice opened the door. Bob entered the room.") == [
        "Alice opened the door.",
        "Bob entered the room.",
    ]


def test_collect_candidates_deduplicates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter(
            [
                "Alice opened the door.",
                "Alice opened the door.",
                "Bob entered the room.",
            ]
        ),
    )

    source = {
        "name": "fake",
        "domain": "general",
        "kind": "hf_field",
        "count": 2,
    }

    selected = collect_candidates(
        source,
        global_seen=set(),
        rng=random.Random(42),
        min_chars=5,
        max_chars=200,
    )

    assert sorted(selected) == [
        "Alice opened the door.",
        "Bob entered the room.",
    ]


def test_build_manifest_preserves_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter(
            [
                f"{source['name']} example one.",
                f"{source['name']} example two.",
            ]
        ),
    )

    config = {
        "seed": 42,
        "min_chars": 5,
        "max_chars": 200,
        "sources": [
            {
                "name": "proofwriter",
                "domain": "logical_reasoning",
                "kind": "hf_field",
                "count": 1,
            },
            {
                "name": "magpie",
                "domain": "idioms",
                "kind": "hf_field",
                "count": 1,
            },
        ],
    }

    manifest = build_manifest(config)

    assert len(manifest) == 2

    assert {record["source_dataset"] for record in manifest} == {
        "proofwriter",
        "magpie",
    }

    assert {record["source_domain"] for record in manifest} == {
        "logical_reasoning",
        "idioms",
    }


def test_build_manifest_rejects_short_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter(
            [
                "Only one usable example.",
            ]
        ),
    )

    config = {
        "sources": [
            {
                "name": "fake",
                "domain": "general",
                "kind": "hf_field",
                "count": 2,
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match="produced only 1",
    ):
        build_manifest(config)


def test_write_manifest_writes_jsonl(
    tmp_path,
) -> None:
    output = tmp_path / "pilot.jsonl"

    records = [
        {
            "text": "Alice opened the door.",
            "source_dataset": "fake",
            "source_domain": "general",
        }
    ]

    write_manifest(
        records,
        output,
    )

    row = json.loads(output.read_text(encoding="utf-8").strip())

    assert row == records[0]


def test_sentence_split_produces_individual_candidates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(["Anne is blue. Anne is red. " "All red things are nice."]),
    )

    source = {
        "kind": "hf_field",
        "sentence_split": True,
    }

    assert list(iter_source_texts(source)) == [
        "Anne is blue.",
        "Anne is red.",
        "All red things are nice.",
    ]


def test_drop_questions_after_sentence_split(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(
            [
                "Amy has 10 apples. "
                "Amy gives away 2 apples. "
                "How many apples remain?"
            ]
        ),
    )

    source = {
        "kind": "hf_field",
        "sentence_split": True,
        "drop_questions": True,
    }

    assert list(iter_source_texts(source)) == [
        "Amy has 10 apples.",
        "Amy gives away 2 apples.",
    ]


def test_questions_are_preserved_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(["Where is Alice?"]),
    )

    source = {
        "kind": "hf_field",
    }

    assert list(iter_source_texts(source)) == ["Where is Alice?"]


def test_drop_questions_only_removes_interrogatives(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(
            [
                "John has 20 marbles. "
                "He gives 5 marbles to Alice. "
                "How many marbles remain? "
                "Alice keeps the marbles."
            ]
        ),
    )

    source = {
        "kind": "hf_field",
        "sentence_split": True,
        "drop_questions": True,
    }

    assert list(iter_source_texts(source)) == [
        "John has 20 marbles.",
        "He gives 5 marbles to Alice.",
        "Alice keeps the marbles.",
    ]


def test_sentence_split_without_question_filter_keeps_questions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(["John has 20 marbles. " "How many marbles remain?"]),
    )

    source = {
        "kind": "hf_field",
        "sentence_split": True,
    }

    assert list(iter_source_texts(source)) == [
        "John has 20 marbles.",
        "How many marbles remain?",
    ]


def test_sentence_split_normalizes_whitespace(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(["Anne   is blue.\n" "Bob   is red."]),
    )

    source = {
        "kind": "hf_field",
        "sentence_split": True,
    }

    assert list(iter_source_texts(source)) == [
        "Anne is blue.",
        "Bob is red.",
    ]
