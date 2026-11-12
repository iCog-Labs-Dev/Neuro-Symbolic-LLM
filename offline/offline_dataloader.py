"""Configuration-driven offline dataset preparation.

Supported source kinds:
    hf_field       : Hugging Face dataset with a text field.
    hf_list_field  : Hugging Face dataset with a list-valued field.
    delimited      : Local CSV/TSV file.
    jsonl_field    : Local JSONL file.
    local_text_dir : Local directory containing text files.
    spgc           : Original Standardized Project Gutenberg Corpus.

Selection:
    dataset name : Select one dataset.
    domain name  : Select all datasets in that domain.
    all          : Select all configured datasets.

Output modes:
    separate  : One JSONL file per dataset.
    by_domain : Combine datasets belonging to each domain.
    mixed     : Combine all selected datasets.
    all       : Generate all three output modes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import random
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = REPO_ROOT / "configs" / "offline_data.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "offline_corpus"

WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")

SUPPORTED_KINDS = {
    "hf_field",
    "hf_list_field",
    "delimited",
    "jsonl_field",
    "local_text_dir",
    "spgc",
}

SUPPORTED_MODES = {
    "separate",
    "by_domain",
    "mixed",
    "all",
}

_SPGC_METADATA_CACHE: dict[int, dict[str, Any]] | None = None


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def resolve_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""

    path = Path(path)

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing lexical content."""

    return WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split normalized text into sentence-like units."""

    text = normalize_text(text)

    if not text:
        return []

    return [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]


def is_usable_text(
    text: str,
    min_chars: int,
    max_chars: int,
) -> bool:
    """Check whether a passage satisfies the configured constraints."""

    return min_chars <= len(text) <= max_chars and any(
        character.isalpha() for character in text
    )


# ---------------------------------------------------------------------
# Configuration loading and validation
# ---------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate offline_data.yaml."""

    path = resolve_path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Offline dataset configuration not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError("Offline configuration must be a YAML mapping.")

    sources = config.get("sources")

    if not isinstance(sources, list) or not sources:
        raise ValueError("Configuration must contain a nonempty sources list.")

    selection_config = config.get("selection", {})

    if not isinstance(selection_config, dict):
        raise ValueError("selection must be a YAML mapping.")

    names: set[str] = set()

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every source must be a YAML mapping.")

        for field in ("name", "domain", "kind", "count"):
            if field not in source:
                raise ValueError(f"Source configuration missing {field!r}: {source}")

        name = source["name"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every source must have a nonempty name.")

        if name in names:
            raise ValueError(f"Duplicate dataset name: {name!r}")

        names.add(name)

        if source["kind"] not in SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported source kind {source['kind']!r} "
                f"for dataset {name!r}. "
                f"Supported kinds: {sorted(SUPPORTED_KINDS)}"
            )

        count = source["count"]

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                f"Dataset {name!r}: count must be a " "nonnegative integer."
            )

        if not isinstance(source["domain"], str) or not source["domain"].strip():
            raise ValueError(f"Dataset {name!r}: domain must be nonempty.")

    default_selection = selection_config.get("default", "all")

    if not isinstance(default_selection, str) or not default_selection.strip():
        raise ValueError("selection.default must be a nonempty string.")

    return config


# ---------------------------------------------------------------------
# Dataset and domain selection
# ---------------------------------------------------------------------


def select_sources(
    config: dict[str, Any],
    selection: str | None = None,
) -> list[dict[str, Any]]:
    """Select datasets by name, domain, or all.

    Exact dataset-name matching takes precedence over domain matching.
    """

    sources = config["sources"]

    selected_name = selection or config.get("selection", {}).get("default", "all")

    if not isinstance(selected_name, str) or not selected_name.strip():
        raise ValueError("Dataset selection must be a nonempty string.")

    selected_name = selected_name.strip()

    if selected_name == "all":
        return list(sources)

    by_name = [source for source in sources if source["name"] == selected_name]

    if by_name:
        return by_name

    by_domain = [source for source in sources if source["domain"] == selected_name]

    if by_domain:
        return by_domain

    available_names = [source["name"] for source in sources]

    available_domains = sorted({source["domain"] for source in sources})

    raise ValueError(
        f"Unknown dataset or domain: {selected_name!r}.\n"
        f"Available datasets: {available_names}\n"
        f"Available domains: {available_domains}"
    )


# ---------------------------------------------------------------------
# Hugging Face dataset loading
# ---------------------------------------------------------------------


def dataset_kwargs(
    source: dict[str, Any],
) -> dict[str, Any]:
    """Build Hugging Face load_dataset arguments from YAML."""

    kwargs: dict[str, Any] = {
        "split": source.get("split", "train"),
        "streaming": bool(source.get("streaming", False)),
    }

    if source.get("config"):
        kwargs["name"] = source["config"]

    if source.get("data_files"):
        kwargs["data_files"] = source["data_files"]

    if source.get("revision"):
        kwargs["revision"] = source["revision"]

    return kwargs


def iter_hf_field(
    source: dict[str, Any],
) -> Iterator[str]:
    """Read text from a Hugging Face dataset field."""

    dataset = load_dataset(
        source["dataset"],
        **dataset_kwargs(source),
    )

    field = source["field"]

    for row in dataset:
        value = row.get(field)

        if isinstance(value, str):
            yield value


def iter_hf_list_field(
    source: dict[str, Any],
) -> Iterator[str]:
    """Read individual strings from a list-valued dataset field."""

    dataset = load_dataset(
        source["dataset"],
        **dataset_kwargs(source),
    )

    field = source["field"]

    for row in dataset:
        values = row.get(field)

        if isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    yield value


# ---------------------------------------------------------------------
# Local file loading
# ---------------------------------------------------------------------


def iter_delimited(
    source: dict[str, Any],
) -> Iterator[str]:
    """Read a configured field from a local CSV/TSV file."""

    path = resolve_path(source["path"])

    delimiter = source.get("delimiter", "\t")
    field = source["field"]

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter=delimiter,
        )

        if field not in (reader.fieldnames or []):
            raise ValueError(f"Field {field!r} not found in {path}.")

        for row in reader:
            value = row.get(field)

            if isinstance(value, str):
                yield value


def iter_jsonl_field(
    source: dict[str, Any],
) -> Iterator[str]:
    """Read a configured field from a local JSONL file."""

    path = resolve_path(source["path"])
    field = source["field"]

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
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


def iter_local_text_dir(
    source: dict[str, Any],
) -> Iterator[str]:
    """Read local text files and produce paragraph-based passages."""

    root = resolve_path(source["path"])

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Text directory does not exist: {root}")

    pattern = source.get("glob", "**/*.txt")
    max_chars = int(source.get("max_chars", 600))

    for path in sorted(root.glob(pattern)):
        paragraphs = [
            normalize_text(paragraph)
            for paragraph in PARAGRAPH_BOUNDARY_RE.split(
                path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
        ]

        yield from pack_paragraphs(
            paragraphs,
            max_chars=max_chars,
        )


# ---------------------------------------------------------------------
# Shared paragraph and passage processing
# ---------------------------------------------------------------------


def pack_paragraphs(
    paragraphs: list[str],
    *,
    max_chars: int,
) -> Iterator[str]:
    """Combine paragraphs into passages up to max_chars."""

    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)

        if not paragraph:
            continue

        pieces = (
            split_sentences(paragraph) if len(paragraph) > max_chars else [paragraph]
        )

        for piece in pieces:
            piece = normalize_text(piece)

            if not piece or len(piece) > max_chars:
                continue

            added_length = len(piece) + (1 if current else 0)

            if current and current_length + added_length > max_chars:
                yield " ".join(current)

                current = []
                current_length = 0

            current.append(piece)
            current_length += added_length

    if current:
        yield " ".join(current)


# ---------------------------------------------------------------------
# Original SPGC: metadata-first, selective book acquisition
# ---------------------------------------------------------------------


def prepare_spgc_repository(
    source: dict[str, Any],
) -> Path:
    """Clone SPGC tools and prepare metadata without syncing all books."""

    cache_dir = resolve_path(source.get("cache_dir", "data/raw/spgc"))

    repository_dir = cache_dir / "gutenberg"

    repository_url = source.get(
        "repository",
        "https://github.com/pgcorpus/gutenberg.git",
    )

    revision = source.get("revision", "master")

    if not shutil.which("git"):
        raise RuntimeError("Git is required for SPGC acquisition.")

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not repository_dir.exists():
        print(
            "[SPGC] Cloning acquisition tools...",
            flush=True,
        )

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                revision,
                repository_url,
                str(repository_dir),
            ],
            check=True,
        )

    if not (repository_dir / "src" / "metadataparser.py").is_file():
        raise FileNotFoundError(f"Invalid SPGC repository: {repository_dir}")

    for relative in (
        "data/.mirror",
        "data/raw",
        "data/text",
        "data/tokens",
        "data/counts",
        "metadata",
        "metadata/bookshelves_html",
    ):
        (repository_dir / relative).mkdir(
            parents=True,
            exist_ok=True,
        )

    _load_spgc_metadata(repository_dir)

    return repository_dir


def _load_spgc_metadata(
    repository_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Load and cache official Gutenberg RDF metadata.

    Only the metadata archive is downloaded here.
    No full-text book synchronization is performed.
    """

    global _SPGC_METADATA_CACHE

    if _SPGC_METADATA_CACHE is not None:
        return _SPGC_METADATA_CACHE

    parser_path = repository_dir / "src" / "metadataparser.py"

    metadata_dir = repository_dir / "metadata"
    rdf_path = metadata_dir / "rdf-files.tar.bz2"
    csv_path = metadata_dir / "metadata.csv"

    spec = importlib.util.spec_from_file_location(
        "_spgc_metadata_parser",
        parser_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load the official SPGC metadata parser.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The official parser downloads the RDF catalog only if missing.
    # It does not download the Gutenberg book-text mirror.
    metadata = module.readmetadata(
        str(rdf_path),
        update=False,
    )

    if not metadata:
        raise RuntimeError("The SPGC RDF metadata catalog is empty.")

    fields = [
        "id",
        "title",
        "author",
        "authoryearofbirth",
        "authoryearofdeath",
        "language",
        "downloads",
        "subjects",
        "type",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for book_id, record in metadata.items():
            subjects = record.get("subjects") or []
            languages = record.get("language") or []

            writer.writerow(
                {
                    "id": f"PG{book_id}",
                    "title": record.get("title") or "",
                    "author": record.get("author") or "",
                    "authoryearofbirth": (record.get("authoryearofbirth") or ""),
                    "authoryearofdeath": (record.get("authoryearofdeath") or ""),
                    "language": repr(list(languages)),
                    "downloads": record.get("downloads") or "",
                    "subjects": repr(sorted(subjects)),
                    "type": record.get("type") or "",
                }
            )

    _SPGC_METADATA_CACHE = {
        int(book_id): record for book_id, record in metadata.items()
    }

    return _SPGC_METADATA_CACHE


def find_existing_spgc_books(
    mirror_dir: Path,
    selected_ids: list[str],
) -> dict[str, Path]:
    """Find selected book IDs already present in the Gutenberg mirror."""

    wanted = {str(book_id).removeprefix("PG") for book_id in selected_ids}

    existing: dict[str, Path] = {}
    priorities: dict[str, int] = {}

    patterns = (
        (re.compile(r"^(\d+)-0\.txt$"), 0),
        (
            re.compile(r"^pg(\d+)\.txt\.utf8$", re.IGNORECASE),
            1,
        ),
        (
            re.compile(r"^pg(\d+)\.txt\.utf-8$", re.IGNORECASE),
            1,
        ),
    )

    # Index the existing mirror, but retain only selected book IDs.
    for path in mirror_dir.rglob("*"):
        if not path.is_file():
            continue

        for pattern, base_priority in patterns:
            match = pattern.fullmatch(path.name)

            if not match:
                continue

            book_id = match.group(1)

            if book_id not in wanted:
                break

            priority = base_priority + (10 if "old" in path.parts else 0)

            if book_id not in existing or priority < priorities[book_id]:
                existing[book_id] = path
                priorities[book_id] = priority

            break

    return existing


def download_selected_spgc_books(
    repository_dir: Path,
    selected_ids: list[str],
) -> None:
    """Reuse selected books and download only missing selected IDs.

    Progress measures completed missing books, not the entire Gutenberg
    catalog or bytes transferred.

    If all selected books already exist, no book download is performed.
    """

    import urllib.request

    raw_dir = repository_dir / "data" / "raw"
    mirror_dir = repository_dir / "data" / ".mirror"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_mirror = find_existing_spgc_books(
        mirror_dir,
        selected_ids,
    )

    missing: list[str] = []
    available = 0

    for book_id in selected_ids:
        raw_path = raw_dir / f"{book_id}_raw.txt"

        if raw_path.is_file() and raw_path.stat().st_size > 0:
            available += 1
            continue

        mirror_path = existing_mirror.get(book_id.removeprefix("PG"))

        if mirror_path is not None:
            try:
                raw_path.hardlink_to(mirror_path)
            except OSError:
                shutil.copy2(
                    mirror_path,
                    raw_path,
                )

            available += 1
            continue

        missing.append(book_id)

    print(
        f"[SPGC] Selected books: {len(selected_ids)}",
        flush=True,
    )

    print(
        f"[SPGC] Already available: {available}",
        flush=True,
    )

    print(
        f"[SPGC] Missing books: {len(missing)}",
        flush=True,
    )

    if not missing:
        print(
            "[DOWNLOAD_SKIPPED] All selected books are available.",
            flush=True,
        )
        return

    metadata = _load_spgc_metadata(repository_dir)

    completed = 0
    last_percent = 0

    print(
        "[DOWNLOAD_PROGRESS] 0%",
        flush=True,
    )

    for book_id in missing:
        numeric_id = int(book_id.removeprefix("PG"))

        record = metadata.get(numeric_id)

        if record is None:
            raise ValueError(f"Missing SPGC metadata for {book_id}.")

        formats = record.get("formats") or {}
        candidates: list[tuple[int, str]] = []

        for mime, url in formats.items():
            mime_lower = str(mime).lower()

            if not mime_lower.startswith("text/plain"):
                continue

            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue

            priority = 0 if "utf-8" in mime_lower else 1

            candidates.append((priority, url))

        candidates.sort()

        # Fallback URLs for standard Gutenberg plain-text editions.
        candidates.extend(
            [
                (
                    2,
                    f"https://www.gutenberg.org/cache/epub/"
                    f"{numeric_id}/pg{numeric_id}.txt.utf8",
                ),
                (
                    3,
                    f"https://www.gutenberg.org/cache/epub/"
                    f"{numeric_id}/pg{numeric_id}.txt",
                ),
                (
                    4,
                    f"https://www.gutenberg.org/files/"
                    f"{numeric_id}/{numeric_id}-0.txt",
                ),
            ]
        )

        raw_path = raw_dir / f"{book_id}_raw.txt"

        temporary_path = raw_dir / (f"{book_id}_raw.txt.part")

        errors: list[str] = []
        downloaded = False

        for _, url in candidates:
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": ("NeuroSymbolicLLM/1.0 research")},
                )

                with (
                    urllib.request.urlopen(request, timeout=90) as response,
                    temporary_path.open("wb") as destination,
                ):
                    shutil.copyfileobj(
                        response,
                        destination,
                        length=1024 * 1024,
                    )

                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError("Downloaded file is empty.")

                temporary_path.replace(raw_path)
                downloaded = True
                break

            except Exception as error:
                errors.append(f"{url}: {error}")

                if temporary_path.exists():
                    temporary_path.unlink()

        if not downloaded:
            raise RuntimeError(
                f"Could not download selected book {book_id}: " + "; ".join(errors)
            )

        completed += 1

        percent = int(completed / len(missing) * 100)

        if percent != last_percent:
            print(
                f"[DOWNLOAD_PROGRESS] {percent}%",
                flush=True,
            )

            last_percent = percent

    print(
        "[DOWNLOAD_COMPLETE]",
        flush=True,
    )


def parse_spgc_metadata_value(
    value: Any,
) -> list[str]:
    """Parse list/set metadata values stored in the SPGC CSV."""

    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]

    if isinstance(parsed, str):
        return [parsed]

    if isinstance(parsed, list | tuple | set):
        return [str(item) for item in parsed if item is not None]

    return []


def load_spgc_cleaner(repository_dir: Path):
    """Load the official SPGC header/footer cleaning function."""

    cleaner_path = repository_dir / "src" / "cleanup.py"

    if not cleaner_path.is_file():
        raise FileNotFoundError(f"SPGC cleaner not found: {cleaner_path}")

    spec = importlib.util.spec_from_file_location(
        "_spgc_cleanup",
        cleaner_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SPGC cleaner: {cleaner_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.strip_headers


def select_spgc_books(
    metadata_path: Path,
    source: dict[str, Any],
    *,
    seed: int,
) -> list[str]:
    """Select books using SPGC language and subject metadata."""

    language_filter = str(source.get("language", "en")).casefold()

    subject_filter = source.get(
        "subject",
        "Fiction",
    )

    subject_filters = (
        [subject_filter] if isinstance(subject_filter, str) else subject_filter
    )

    subject_filters = [str(item).casefold() for item in subject_filters]

    books: list[str] = []

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            languages = parse_spgc_metadata_value(row.get("language"))

            subjects = parse_spgc_metadata_value(row.get("subjects"))

            is_english = any(item.casefold() == language_filter for item in languages)

            is_fiction = any(
                term in subject.casefold()
                for subject in subjects
                for term in subject_filters
            )

            if is_english and is_fiction:
                book_id = row.get("id")

                if book_id:
                    books.append(str(book_id))

    if not books:
        raise ValueError(
            "No SPGC books matched the configured language "
            f"{language_filter!r} and subjects {subject_filters!r}."
        )

    # Deduplicate while preserving order.
    books = list(dict.fromkeys(books))

    random.Random(seed).shuffle(books)

    max_books = source.get("max_books")

    if max_books is not None:
        if (
            not isinstance(max_books, int)
            or isinstance(max_books, bool)
            or max_books < 1
        ):
            raise ValueError("SPGC max_books must be a positive integer.")

        books = books[:max_books]

    return books


def iter_spgc(
    source: dict[str, Any],
    *,
    seed: int,
) -> Iterator[str]:
    """Acquire selected SPGC books and yield cleaned passages."""

    repository_dir = prepare_spgc_repository(source)

    metadata_path = repository_dir / "metadata" / "metadata.csv"

    raw_dir = repository_dir / "data" / "raw"
    text_dir = repository_dir / "data" / "text"

    text_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    books = select_spgc_books(
        metadata_path,
        source,
        seed=seed,
    )

    download_selected_spgc_books(
        repository_dir,
        books,
    )

    strip_headers = load_spgc_cleaner(repository_dir)

    max_chars = int(source.get("max_chars", 600))

    processed_books = 0
    missing_books = 0

    for book_id in books:
        raw_path = raw_dir / f"{book_id}_raw.txt"
        text_path = text_dir / f"{book_id}_text.txt"

        if not raw_path.is_file():
            missing_books += 1
            continue

        if text_path.is_file():
            clean_text = text_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        else:
            raw_text = raw_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            clean_text = strip_headers(raw_text)

            text_path.write_text(
                clean_text,
                encoding="utf-8",
            )

        processed_books += 1

        paragraphs = PARAGRAPH_BOUNDARY_RE.split(clean_text)

        yield from pack_paragraphs(
            paragraphs,
            max_chars=max_chars,
        )

        if processed_books % 50 == 0:
            print(
                f"[SPGC] Processed {processed_books}/" f"{len(books)} selected books."
            )

    if processed_books == 0:
        raise RuntimeError("No selected SPGC books were available for processing.")

    if missing_books:
        print(f"[SPGC] Missing raw books: {missing_books}")


# ---------------------------------------------------------------------
# Unified source iterator
# ---------------------------------------------------------------------


def iter_source_texts(
    source: dict[str, Any],
    *,
    seed: int,
) -> Iterator[str]:
    """Load any configured source and yield text candidates."""

    loaders = {
        "hf_field": lambda: iter_hf_field(source),
        "hf_list_field": lambda: iter_hf_list_field(source),
        "delimited": lambda: iter_delimited(source),
        "jsonl_field": lambda: iter_jsonl_field(source),
        "local_text_dir": lambda: iter_local_text_dir(source),
        "spgc": lambda: iter_spgc(source, seed=seed),
    }

    kind = source["kind"]

    if kind not in loaders:
        raise ValueError(f"Unsupported source kind: {kind!r}")

    sentence_split = bool(source.get("sentence_split", False))

    drop_questions = bool(source.get("drop_questions", False))

    for raw_text in loaders[kind]():
        if sentence_split:
            candidates = split_sentences(raw_text)
        else:
            normalized = normalize_text(raw_text)
            candidates = [normalized] if normalized else []

        for candidate in candidates:
            if drop_questions and candidate.rstrip().endswith("?"):
                continue

            yield candidate


# ---------------------------------------------------------------------
# Candidate collection and sampling
# ---------------------------------------------------------------------


def collect_candidates(
    source: dict[str, Any],
    *,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Collect usable unique examples and sample reproducibly."""

    requested = int(source["count"])

    if requested == 0:
        return []

    source_seed = seed + sum(ord(character) for character in source["name"])

    rng = random.Random(source_seed)

    max_scan = int(
        source.get(
            "max_scan",
            max(requested * 50, 1000),
        )
    )

    if max_scan < 1:
        raise ValueError(f"{source['name']}: max_scan must be positive.")

    candidates: list[str] = []
    seen: set[str] = set()

    for index, raw_text in enumerate(iter_source_texts(source, seed=seed)):
        if index >= max_scan:
            break

        text = normalize_text(raw_text)

        if not is_usable_text(
            text,
            min_chars,
            max_chars,
        ):
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


# ---------------------------------------------------------------------
# Dataset preparation and provenance
# ---------------------------------------------------------------------


def prepare_sources(
    config: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Prepare selected datasets independently."""

    seed = int(config.get("seed", 42))

    global_min = int(config.get("min_chars", 20))

    global_max = int(config.get("max_chars", 600))

    prepared: dict[str, list[dict[str, str]]] = {}

    for source in sources:
        name = source["name"]
        domain = source["domain"]

        min_chars = int(source.get("min_chars", global_min))

        max_chars = int(source.get("max_chars", global_max))

        if min_chars < 0 or max_chars < min_chars:
            raise ValueError(f"Invalid character limits for {name!r}.")

        print("\n" + "-" * 64)
        print(f"Preparing dataset: {name}")
        print(f"Domain: {domain}")
        print(f"Source type: {source['kind']}")
        print("-" * 64)

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

        print(f"Prepared passages: {len(texts)}")

    return prepared


# ---------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------


def deduplicate_records(
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Deduplicate records while preserving first-source provenance."""

    seen: set[str] = set()
    result: list[dict[str, str]] = []

    for record in records:
        key = record["text"].casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(record)

    return result


# ---------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------


def write_jsonl(
    records: list[dict[str, str]],
    path: Path,
) -> None:
    """Write prepared records to UTF-8 JSONL."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_separate(
    prepared: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> dict[str, int]:
    """Write one output file per dataset."""

    counts: dict[str, int] = {}

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
    """Combine datasets belonging to the same domain."""

    grouped: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for records in prepared.values():
        for record in records:
            grouped[record["source_domain"]].append(record)

    counts: dict[str, int] = {}

    for domain, records in sorted(grouped.items()):
        records = deduplicate_records(records)

        domain_seed = seed + sum(ord(character) for character in domain)

        random.Random(domain_seed).shuffle(records)

        path = output_dir / "by_domain" / f"{domain}.jsonl"

        write_jsonl(records, path)

        counts[domain] = len(records)

    return counts


def write_mixed(
    prepared: dict[str, list[dict[str, str]]],
    output_dir: Path,
    seed: int,
) -> dict[str, int]:
    """Combine all selected datasets into one corpus."""

    records = [
        record for source_records in prepared.values() for record in source_records
    ]

    records = deduplicate_records(records)

    random.Random(seed).shuffle(records)

    path = output_dir / "mixed" / "offline_mixed.jsonl"

    write_jsonl(records, path)

    return {"mixed": len(records)}


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare configured offline datasets.")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Offline dataset YAML configuration.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for prepared datasets.",
    )

    parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help=(
            "Dataset name, domain name, or 'all'. "
            "Defaults to selection.default in offline_data.yaml."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default="separate",
        help="Output organization mode.",
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = build_argument_parser().parse_args()

    config = load_config(args.config)

    sources = select_sources(
        config,
        selection=args.selection,
    )

    output_dir = resolve_path(args.output_dir)

    seed = int(config.get("seed", 42))

    selected_name = args.selection or config.get("selection", {}).get("default", "all")

    print("=" * 72)
    print("OFFLINE DATASET PREPARATION")
    print("=" * 72)

    print(f"Configuration: {resolve_path(args.config)}")

    print(f"Selection: {selected_name}")

    print(f"Selected datasets: {len(sources)}")

    for source in sources:
        print(f"  - {source['name']} [{source['domain']}] " f"({source['kind']})")

    prepared = prepare_sources(
        config,
        sources,
    )

    modes = ["separate", "by_domain", "mixed"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "separate":
            counts = write_separate(
                prepared,
                output_dir,
            )

        elif mode == "by_domain":
            counts = write_by_domain(
                prepared,
                output_dir,
                seed,
            )

        else:
            counts = write_mixed(
                prepared,
                output_dir,
                seed,
            )

        print(f"\n{mode.upper()} OUTPUT")

        for name, count in sorted(counts.items()):
            print(f"  {name}: {count} passages")

    total = sum(len(records) for records in prepared.values())

    print("\n" + "=" * 72)
    print("OFFLINE DATASET PREPARATION COMPLETED")
    print("=" * 72)

    print(f"Datasets prepared: {len(prepared)}")

    print(f"Total passages: {total}")

    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
