"""Prepare reproducible parser pilot corpora in separate, domain, or mixed mode."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
DEFAULT_CONFIG = Path("configs/parser_config/pilot_corpus.yaml")
DEFAULT_OUTPUT = Path("data/parser_corpus")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]


def is_usable_text(
    text: str,
    min_chars: int,
    max_chars: int,
) -> bool:
    return min_chars <= len(text) <= max_chars and any(
        character.isalpha() for character in text
    )


def _dataset_kwargs(source: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "split": source.get("split", "train"),
        "streaming": bool(source.get("streaming", False)),
    }

    if source.get("config"):
        kwargs["name"] = source["config"]

    if source.get("data_files"):
        kwargs["data_files"] = source["data_files"]

    return kwargs


def _iter_hf_field(source: dict[str, Any]) -> Iterator[str]:
    dataset = load_dataset(source["dataset"], **_dataset_kwargs(source))
    field = source["field"]

    for row in dataset:
        value = row.get(field)
        if isinstance(value, str):
            yield value


def _iter_hf_list_field(source: dict[str, Any]) -> Iterator[str]:
    dataset = load_dataset(source["dataset"], **_dataset_kwargs(source))
    field = source["field"]

    for row in dataset:
        values = row.get(field)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    yield value


def _iter_delimited(source: dict[str, Any]) -> Iterator[str]:
    path = Path(source["path"])
    delimiter = source.get("delimiter", "\t")
    field = source["field"]

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        for row in reader:
            value = row.get(field)
            if isinstance(value, str):
                yield value


def _iter_jsonl_field(source: dict[str, Any]) -> Iterator[str]:
    path = Path(source["path"])
    field = source["field"]

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error

            if isinstance(record, dict):
                value = record.get(field)
                if isinstance(value, str):
                    yield value


def _iter_local_text_dir(source: dict[str, Any]) -> Iterator[str]:
    """Read local text files and yield paragraph-based passages.

    Books are not treated as single examples. Paragraphs are packed into
    passages up to the configured max_chars limit.
    """
    root = Path(source["path"])
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Text directory does not exist: {root}")

    pattern = source.get("glob", "**/*.txt")
    max_chars = int(source.get("max_chars", 600))

    for path in sorted(root.glob(pattern)):
        paragraphs = [
            normalize_text(p)
            for p in path.read_text(encoding="utf-8", errors="replace").split("\n\n")
        ]

        current: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            if not paragraph:
                continue

            # Split exceptionally long paragraphs into sentence-like units.
            pieces = (
                split_sentences(paragraph)
                if len(paragraph) > max_chars
                else [paragraph]
            )

            for piece in pieces:
                if len(piece) > max_chars:
                    continue

                added_length = len(piece) + (1 if current else 0)

                if current and current_length + added_length > max_chars:
                    yield " ".join(current)
                    current = []
                    current_length = 0

                current.append(piece)
                current_length += len(piece) + (1 if len(current) > 1 else 0)

        if current:
            yield " ".join(current)


def iter_source_texts(source: dict[str, Any]) -> Iterator[str]:
    kind = source["kind"]

    loaders = {
        "hf_field": _iter_hf_field,
        "hf_list_field": _iter_hf_list_field,
        "delimited": _iter_delimited,
        "jsonl_field": _iter_jsonl_field,
        "local_text_dir": _iter_local_text_dir,
    }

    if kind not in loaders:
        raise ValueError(f"Unsupported source kind: {kind!r}")

    iterator = loaders[kind](source)
    sentence_split = bool(source.get("sentence_split", False))
    drop_questions = bool(source.get("drop_questions", False))

    for raw_text in iterator:
        if sentence_split:
            candidates = split_sentences(raw_text)
        else:
            normalized = normalize_text(raw_text)
            candidates = [normalized] if normalized else []

        for candidate in candidates:
            if drop_questions and candidate.endswith("?"):
                continue
            yield candidate


def collect_candidates(
    source: dict[str, Any],
    *,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Sample one source independently and reproducibly."""
    requested = int(source["count"])
    if requested <= 0:
        return []

    source_seed = seed + sum(ord(c) for c in source["name"])
    rng = random.Random(source_seed)

    max_scan = int(source.get("max_scan", max(requested * 50, 1000)))
    candidates: list[str] = []
    seen: set[str] = set()

    for index, raw_text in enumerate(iter_source_texts(source)):
        if index >= max_scan:
            break

        text = normalize_text(raw_text)
        if not is_usable_text(text, min_chars, max_chars):
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        candidates.append(text)

    if len(candidates) < requested:
        raise ValueError(
            f"{source['name']}: requested {requested} examples, "
            f"but found only {len(candidates)} usable unique examples "
            f"after scanning {max_scan} candidates."
        )

    rng.shuffle(candidates)
    return candidates[:requested]


def prepare_sources(
    config: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Prepare every configured dataset independently."""
    seed = int(config.get("seed", 42))
    global_min = int(config.get("min_chars", 20))
    global_max = int(config.get("max_chars", 600))

    prepared: dict[str, list[dict[str, str]]] = {}

    for source in config["sources"]:
        if not source.get("enabled", True):
            continue

        for required in ("name", "domain", "kind", "count"):
            if required not in source:
                raise ValueError(f"Source configuration missing {required!r}: {source}")

        name = source["name"]
        domain = source["domain"]

        print(f"Preparing {name} [{domain}] ...")

        min_chars = int(source.get("min_chars", global_min))
        max_chars = int(source.get("max_chars", global_max))

        texts = collect_candidates(
            source,
            seed=seed,
            min_chars=min_chars,
            max_chars=max_chars,
        )

        prepared[name] = [
            {
                "text": text,
                "source_dataset": name,
                "source_domain": domain,
            }
            for text in texts
        ]

        print(f"  Prepared {len(texts)} examples")

    return prepared


def deduplicate_records(
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Deduplicate combined corpora while retaining source provenance."""
    seen: set[str] = set()
    result = []

    for record in records:
        key = record["text"].casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(record)

    return result


def write_jsonl(
    records: list[dict[str, str]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_separate(
    prepared: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> dict[str, int]:
    counts = {}

    for name, records in prepared.items():
        path = output_dir / "separate" / f"{name}.jsonl"
        write_jsonl(records, path)
        counts[name] = len(records)

    return counts


def write_by_domain(
    prepared: dict[str, list[dict[str, str]]],
    output_dir: Path,
    seed: int,
) -> dict[str, int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for records in prepared.values():
        for record in records:
            grouped[record["source_domain"]].append(record)

    counts = {}

    for domain, records in sorted(grouped.items()):
        records = deduplicate_records(records)
        random.Random(seed + sum(ord(c) for c in domain)).shuffle(records)

        path = output_dir / "by_domain" / f"{domain}.jsonl"
        write_jsonl(records, path)
        counts[domain] = len(records)

    return counts


def write_mixed(
    prepared: dict[str, list[dict[str, str]]],
    output_dir: Path,
    seed: int,
) -> dict[str, int]:
    records = [
        record for source_records in prepared.values() for record in source_records
    ]

    records = deduplicate_records(records)
    random.Random(seed).shuffle(records)

    write_jsonl(records, output_dir / "mixed" / "pilot_mixed.jsonl")

    return {"mixed": len(records)}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    if not isinstance(config.get("sources"), list) or not config["sources"]:
        raise ValueError("Configuration must contain a nonempty sources list.")

    return config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare parser pilot datasets.")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--mode",
        choices=["separate", "by_domain", "mixed", "all"],
        default="separate",
        help="Preparation mode. Default: separate.",
    )

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    config = load_config(args.config)

    prepared = prepare_sources(config)
    seed = int(config.get("seed", 42))

    modes = ["separate", "by_domain", "mixed"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "separate":
            counts = write_separate(prepared, args.output_dir)
        elif mode == "by_domain":
            counts = write_by_domain(prepared, args.output_dir, seed)
        else:
            counts = write_mixed(prepared, args.output_dir, seed)

        print(f"\n{mode.upper()} preparation:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")

    print(f"\nOutput directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
