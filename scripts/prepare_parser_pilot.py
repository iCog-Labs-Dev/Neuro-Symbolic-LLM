"""Prepare a multi-source raw-text manifest for semantic-parser distillation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize whitespace without changing lexical content."""
    return _WHITESPACE_RE.sub(" ", value).strip()


def split_sentences(text: str) -> list[str]:
    """Split document text into conservative sentence-like units."""
    normalized = normalize_text(text)

    if not normalized:
        return []

    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(normalized)
        if sentence.strip()
    ]


def is_usable_text(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
) -> bool:
    """Return whether text is suitable for the parser pilot."""
    length = len(text)

    if length < min_chars or length > max_chars:
        return False

    if not any(character.isalpha() for character in text):
        return False

    return True


def _iter_hf_field(source: dict[str, Any]) -> Iterator[str]:
    dataset_name = source["dataset"]
    split = source.get("split", "train")
    config_name = source.get("config")
    field = source["field"]
    streaming = bool(source.get("streaming", False))

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
    }

    if "data_files" in source:
        kwargs["data_files"] = source["data_files"]

    if config_name is not None:
        kwargs["name"] = config_name

    dataset = load_dataset(
        dataset_name,
        **kwargs,
    )

    for row in dataset:
        value = row.get(field)

        if isinstance(value, str):
            yield value


def _iter_hf_list_field(source: dict[str, Any]) -> Iterator[str]:
    dataset_name = source["dataset"]
    split = source.get("split", "train")
    config_name = source.get("config")
    field = source["field"]
    streaming = bool(source.get("streaming", False))

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
    }

    if "data_files" in source:
        kwargs["data_files"] = source["data_files"]

    if config_name is not None:
        kwargs["name"] = config_name

    dataset = load_dataset(
        dataset_name,
        **kwargs,
    )

    for row in dataset:
        values = row.get(field)

        if not isinstance(values, list):
            continue

        for value in values:
            if isinstance(value, str):
                yield value


def _iter_delimited(source: dict[str, Any]) -> Iterator[str]:
    path = Path(source["path"])
    field = source["field"]
    delimiter = source.get("delimiter", "\t")

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
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                continue

            value = record.get(field)

            if isinstance(value, str):
                yield value


def iter_source_texts(source: dict[str, Any]) -> Iterator[str]:
    """Yield preprocessed text candidates from one configured source."""
    kind = source["kind"]

    if kind == "hf_field":
        iterator = _iter_hf_field(source)
    elif kind == "hf_list_field":
        iterator = _iter_hf_list_field(source)
    elif kind == "delimited":
        iterator = _iter_delimited(source)
    elif kind == "jsonl_field":
        iterator = _iter_jsonl_field(source)
    else:
        raise ValueError(f"Unsupported source kind: {kind!r}")

    split_into_sentences = bool(source.get("sentence_split", False))
    drop_questions = bool(source.get("drop_questions", False))

    for text in iterator:
        if split_into_sentences:
            candidates = split_sentences(text)
        else:
            normalized = normalize_text(text)
            candidates = [normalized] if normalized else []

        for candidate in candidates:
            if drop_questions and candidate.rstrip().endswith("?"):
                continue

            yield candidate


def collect_candidates(
    source: dict[str, Any],
    *,
    global_seen: set[str],
    rng: random.Random,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Collect a deterministic deduplicated sample for one source."""
    requested = int(source["count"])

    if requested <= 0:
        return []

    max_scan = int(source.get("max_scan", max(requested * 50, 1000)))

    candidates: list[str] = []
    local_seen: set[str] = set()

    for index, raw_text in enumerate(iter_source_texts(source)):
        if index >= max_scan:
            break

        text = normalize_text(raw_text)

        if not is_usable_text(
            text,
            min_chars=min_chars,
            max_chars=max_chars,
        ):
            continue

        dedup_key = text.casefold()

        if dedup_key in local_seen or dedup_key in global_seen:
            continue

        local_seen.add(dedup_key)
        candidates.append(text)

    if len(candidates) < requested:
        raise ValueError(
            f"Source {source['name']!r} produced only "
            f"{len(candidates)} usable unique examples; "
            f"{requested} requested"
        )

    rng.shuffle(candidates)

    selected = candidates[:requested]

    for text in selected:
        global_seen.add(text.casefold())

    return selected


def build_manifest(config: dict[str, Any]) -> list[dict[str, str]]:
    """Build the complete provenance-tagged pilot manifest."""
    seed = int(config.get("seed", 42))
    min_chars = int(config.get("min_chars", 20))
    max_chars = int(config.get("max_chars", 600))

    rng = random.Random(seed)
    global_seen: set[str] = set()
    manifest: list[dict[str, str]] = []

    sources = config.get("sources")

    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must contain a nonempty 'sources' list")

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source configuration must be an object")

        if not source.get("enabled", True):
            continue

        for required in ("name", "domain", "kind", "count"):
            if required not in source:
                raise ValueError(f"Source is missing required field {required!r}")
        print(f"Loading source: {source['name']}")
        selected = collect_candidates(
            source,
            global_seen=global_seen,
            rng=rng,
            min_chars=min_chars,
            max_chars=max_chars,
        )

        for text in selected:
            manifest.append(
                {
                    "text": text,
                    "source_dataset": str(source["name"]),
                    "source_domain": str(source["domain"]),
                }
            )

    rng.shuffle(manifest)

    return manifest


def write_manifest(
    records: Iterable[dict[str, str]],
    output_path: Path,
) -> None:
    """Write the pilot corpus manifest as UTF-8 JSONL."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML configuration root."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Pilot corpus config must contain a mapping")

    return data


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a provenance-tagged semantic-parser pilot corpus."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/parser_config/pilot_corpus.yaml"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/parser_corpus/pilot_1000.jsonl"),
    )

    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()

    config = load_config(arguments.config)

    manifest = build_manifest(config)

    write_manifest(
        manifest,
        arguments.output,
    )

    print(f"Wrote {len(manifest)} examples -> " f"{arguments.output}")

    counts: dict[tuple[str, str], int] = {}

    for record in manifest:
        key = (
            record["source_dataset"],
            record["source_domain"],
        )
        counts[key] = counts.get(key, 0) + 1

    for (dataset, domain), count in sorted(counts.items()):
        print(f"  {dataset}: {count} " f"({domain})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
