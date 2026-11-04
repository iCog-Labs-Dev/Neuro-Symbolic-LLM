"""Tests for the refactored parser pilot-corpus preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_parser_pilot import (
    build_argument_parser,
    collect_candidates,
    deduplicate_records,
    is_usable_text,
    iter_source_texts,
    load_config,
    normalize_text,
    prepare_sources,
    split_sentences,
    write_by_domain,
    write_jsonl,
    write_mixed,
    write_separate,
)


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Read generated JSONL records for assertions."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_record(
    text: str,
    source: str,
    domain: str,
) -> dict[str, str]:
    return {
        "text": text,
        "source_dataset": source,
        "source_domain": domain,
    }


@pytest.fixture
def prepared_sources() -> dict[str, list[dict[str, str]]]:
    """Small corpus with two sources sharing a domain."""
    return {
        "rocstories": [
            make_record(
                "Alice opened the door.",
                "rocstories",
                "narrative",
            ),
            make_record(
                "Bob entered the room.",
                "rocstories",
                "narrative",
            ),
        ],
        "spgc": [
            make_record(
                "Carol walked through the garden.",
                "spgc",
                "narrative",
            ),
            make_record(
                "Alice opened the door.",
                "spgc",
                "narrative",
            ),
        ],
        "proofwriter": [
            make_record(
                "All birds are animals.",
                "proofwriter",
                "logical_reasoning",
            ),
        ],
    }


# ---------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  Alice   opened\n the door. ") == "Alice opened the door."


def test_split_sentences() -> None:
    assert split_sentences("Alice opened the door. Bob entered the room.") == [
        "Alice opened the door.",
        "Bob entered the room.",
    ]


def test_split_sentences_empty_text() -> None:
    assert split_sentences("   ") == []


def test_is_usable_text() -> None:
    assert is_usable_text("Alice entered.", 5, 100)
    assert not is_usable_text("Hi", 5, 100)
    assert not is_usable_text("123456", 5, 100)
    assert not is_usable_text("A" * 101, 5, 100)


# ---------------------------------------------------------------------
# Source preprocessing
# ---------------------------------------------------------------------


def test_sentence_split_produces_individual_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(["Anne is blue. Anne is red. All red things are nice."]),
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
    monkeypatch: pytest.MonkeyPatch,
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


def test_questions_preserved_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot._iter_hf_field",
        lambda source: iter(["Where is Alice?"]),
    )

    assert list(iter_source_texts({"kind": "hf_field"})) == ["Where is Alice?"]


def test_unsupported_source_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported source kind"):
        list(iter_source_texts({"kind": "unknown"}))


# ---------------------------------------------------------------------
# Local narrative passages
# ---------------------------------------------------------------------


def test_local_text_dir_extracts_passages(
    tmp_path: Path,
) -> None:
    books = tmp_path / "books"
    books.mkdir()

    (books / "story.txt").write_text(
        "Alice entered the garden.\n\n"
        "Bob followed her quietly.\n\n"
        "They discovered a hidden door.",
        encoding="utf-8",
    )

    source = {
        "kind": "local_text_dir",
        "path": str(books),
        "glob": "*.txt",
        "max_chars": 60,
    }

    passages = list(iter_source_texts(source))

    assert passages
    assert all(len(text) <= 60 for text in passages)
    assert any("Alice entered" in text for text in passages)
    assert any("hidden door" in text for text in passages)


def test_local_text_dir_missing_directory(
    tmp_path: Path,
) -> None:
    source = {
        "kind": "local_text_dir",
        "path": str(tmp_path / "missing"),
    }

    with pytest.raises(FileNotFoundError):
        list(iter_source_texts(source))


# ---------------------------------------------------------------------
# Candidate sampling
# ---------------------------------------------------------------------


def test_collect_candidates_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
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
        seed=42,
        min_chars=5,
        max_chars=200,
    )

    assert sorted(selected) == [
        "Alice opened the door.",
        "Bob entered the room.",
    ]


def test_collect_candidates_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter([f"Example number {i}." for i in range(20)]),
    )

    source = {
        "name": "fake",
        "kind": "hf_field",
        "count": 5,
    }

    first = collect_candidates(source, seed=42, min_chars=5, max_chars=200)
    second = collect_candidates(source, seed=42, min_chars=5, max_chars=200)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_collect_candidates_rejects_short_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter(["Only one usable example."]),
    )

    source = {
        "name": "fake",
        "kind": "hf_field",
        "count": 2,
    }

    with pytest.raises(ValueError, match="requested 2 examples"):
        collect_candidates(source, seed=42, min_chars=5, max_chars=200)


def test_collect_candidates_respects_max_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter(["First valid example.", "Second valid example."]),
    )

    source = {
        "name": "fake",
        "kind": "hf_field",
        "count": 2,
        "max_scan": 1,
    }

    with pytest.raises(ValueError, match="found only 1"):
        collect_candidates(source, seed=42, min_chars=5, max_chars=200)


# ---------------------------------------------------------------------
# Source preparation and provenance
# ---------------------------------------------------------------------


def test_prepare_sources_preserves_provenance(
    monkeypatch: pytest.MonkeyPatch,
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

    prepared = prepare_sources(config)

    assert set(prepared) == {"proofwriter", "magpie"}
    assert len(prepared["proofwriter"]) == 1
    assert len(prepared["magpie"]) == 1

    assert prepared["proofwriter"][0]["source_dataset"] == "proofwriter"
    assert prepared["proofwriter"][0]["source_domain"] == "logical_reasoning"

    assert prepared["magpie"][0]["source_dataset"] == "magpie"
    assert prepared["magpie"][0]["source_domain"] == "idioms"


def test_prepare_sources_skips_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.prepare_parser_pilot.iter_source_texts",
        lambda source: iter(["A valid example sentence."]),
    )

    config = {
        "sources": [
            {
                "name": "enabled",
                "domain": "general",
                "kind": "hf_field",
                "count": 1,
            },
            {
                "name": "disabled",
                "domain": "general",
                "kind": "hf_field",
                "count": 1,
                "enabled": False,
            },
        ]
    }

    prepared = prepare_sources(config)

    assert list(prepared) == ["enabled"]


def test_prepare_sources_rejects_missing_fields() -> None:
    config = {
        "sources": [
            {
                "name": "incomplete",
                "kind": "hf_field",
                "count": 1,
            }
        ]
    }

    with pytest.raises(ValueError, match="missing 'domain'"):
        prepare_sources(config)


# ---------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------


def test_deduplicate_records_preserves_first_source() -> None:
    records = [
        make_record("Alice opened the door.", "rocstories", "narrative"),
        make_record("alice opened the door.", "spgc", "narrative"),
        make_record("Bob entered the room.", "spgc", "narrative"),
    ]

    result = deduplicate_records(records)

    assert len(result) == 2
    assert result[0]["source_dataset"] == "rocstories"
    assert result[1]["source_dataset"] == "spgc"


# ---------------------------------------------------------------------
# Output modes
# ---------------------------------------------------------------------


def test_write_separate(
    prepared_sources: dict[str, list[dict[str, str]]],
    tmp_path: Path,
) -> None:
    counts = write_separate(prepared_sources, tmp_path)

    assert counts == {
        "rocstories": 2,
        "spgc": 2,
        "proofwriter": 1,
    }

    for name, records in prepared_sources.items():
        path = tmp_path / "separate" / f"{name}.jsonl"
        assert path.exists()
        assert read_jsonl(path) == records


def test_write_by_domain(
    prepared_sources: dict[str, list[dict[str, str]]],
    tmp_path: Path,
) -> None:
    counts = write_by_domain(prepared_sources, tmp_path, seed=42)

    assert counts == {
        "narrative": 3,
        "logical_reasoning": 1,
    }

    narrative = read_jsonl(tmp_path / "by_domain" / "narrative.jsonl")

    assert len(narrative) == 3
    assert {r["source_dataset"] for r in narrative} == {
        "rocstories",
        "spgc",
    }
    assert all(r["source_domain"] == "narrative" for r in narrative)


def test_write_mixed(
    prepared_sources: dict[str, list[dict[str, str]]],
    tmp_path: Path,
) -> None:
    counts = write_mixed(prepared_sources, tmp_path, seed=42)

    assert counts == {"mixed": 4}

    records = read_jsonl(tmp_path / "mixed" / "pilot_mixed.jsonl")

    assert len(records) == 4
    assert {r["source_domain"] for r in records} == {
        "narrative",
        "logical_reasoning",
    }


def test_write_mixed_is_reproducible(
    prepared_sources: dict[str, list[dict[str, str]]],
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_mixed(prepared_sources, first, seed=42)
    write_mixed(prepared_sources, second, seed=42)

    assert read_jsonl(first / "mixed" / "pilot_mixed.jsonl") == read_jsonl(
        second / "mixed" / "pilot_mixed.jsonl"
    )


def test_write_jsonl_creates_parent_directories(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "pilot.jsonl"
    records = [make_record("Alice opened the door.", "fake", "general")]

    write_jsonl(records, output)

    assert read_jsonl(output) == records


# ---------------------------------------------------------------------
# Configuration and CLI
# ---------------------------------------------------------------------


def test_load_config_rejects_missing_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty sources list"):
        load_config(path)


def test_cli_defaults_to_separate() -> None:
    parser = build_argument_parser()
    args = parser.parse_args([])

    assert args.mode == "separate"


def test_cli_accepts_all_modes() -> None:
    parser = build_argument_parser()

    for mode in ("separate", "by_domain", "mixed", "all"):
        assert parser.parse_args(["--mode", mode]).mode == mode
