"""Export intact source contexts and annotation sidecars from JSONL snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_snapshot(
    snapshot: Path,
    output: Path,
    *,
    name: str,
    dataset: str,
    revision: str,
    split: str,
    text_field: str,
    parent_field: str | None = None,
    split_group_field: str | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Group complete parent texts without exposing annotations to the parser.

    Repeated parent rows retain every annotation in the sidecar. Text is never
    normalized, sentence-split or truncated. Oversized parents retain metadata
    and an exclusion reason. Declared split groups support later leakage audits;
    this function does not invent train/validation/test assignments.
    """
    start = time.perf_counter()
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError("Source name must be a lowercase identifier")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation or test")
    if any(not value.strip() for value in (dataset, revision, text_field)):
        raise ValueError("Dataset, revision and text field must be nonempty")
    if max_chars is not None and (
        isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1
    ):
        raise ValueError("Character limit must be a positive integer")
    if output.exists():
        raise FileExistsError(output)
    groups: dict[str, dict[str, Any]] = {}
    row_count = 0
    with snapshot.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {number} must be an object")
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Row {number} has no source text")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            parent = row.get(parent_field) if parent_field else text_hash
            if (
                isinstance(parent, bool)
                or not isinstance(parent, str | int)
                or not str(parent).strip()
            ):
                raise ValueError(f"Row {number} has no scalar parent identity")
            parent_key = json.dumps(parent, ensure_ascii=False)
            group = groups.setdefault(
                parent_key,
                {
                    "parent_id": parent,
                    "source_text": text,
                    "source_sha256": text_hash,
                    "split_groups": [],
                    "annotations": [],
                },
            )
            if group["source_text"] != text:
                raise ValueError(f"Parent {parent!r} has inconsistent source text")
            split_group = row.get(split_group_field) if split_group_field else parent
            if (
                isinstance(split_group, bool)
                or not isinstance(split_group, str | int)
                or not str(split_group).strip()
            ):
                raise ValueError(f"Row {number} has no scalar split group")
            if split_group not in group["split_groups"]:
                group["split_groups"].append(split_group)
            group["annotations"].append({"snapshot_line": number, "record": row})
            row_count += 1
    if not groups:
        raise ValueError("Snapshot contains no records")
    output.mkdir(parents=True, exist_ok=False)
    accepted = excluded = 0
    with (
        (output / "sources.jsonl").open("x", encoding="utf-8", newline="\n") as sources,
        (output / "metadata.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as metadata,
        (output / "excluded.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as failures,
    ):
        for parent_key, group in groups.items():
            identity = json.dumps(
                [dataset, revision, split, parent_key], ensure_ascii=False
            ).encode("utf-8")
            document_id = f"{name}-{hashlib.sha256(identity).hexdigest()}"
            too_long = max_chars is not None and len(group["source_text"]) > max_chars
            source_record = {
                "document_id": document_id,
                "corpus_id": f"{name}-{split}",
                "source_text": group["source_text"],
            }
            record = {
                "document_id": document_id,
                "dataset": dataset,
                "revision": revision,
                "split": split,
                "included": not too_long,
                **group,
            }
            metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
            if too_long:
                failures.write(
                    json.dumps(
                        {
                            "document_id": document_id,
                            "reason": "source_exceeds_character_limit",
                            "characters": len(group["source_text"]),
                            "limit": max_chars,
                        }
                    )
                    + "\n"
                )
                excluded += 1
            else:
                sources.write(json.dumps(source_record, ensure_ascii=False) + "\n")
                accepted += 1
    manifest = {
        "status": "passed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "revision": revision,
        "split": split,
        "snapshot_sha256": file_digest(snapshot),
        "script_sha256": file_digest(Path(__file__)),
        "input_rows": row_count,
        "parent_count": len(groups),
        "included_parents": accepted,
        "excluded_parents": excluded,
        "policy": {
            "text_field": text_field,
            "parent_field": parent_field,
            "split_group_field": split_group_field,
            "max_chars": max_chars,
            "text_transformed": False,
            "annotations_in_parser_input": False,
        },
        "files": {
            name: file_digest(output / name)
            for name in ("sources.jsonl", "metadata.jsonl", "excluded.jsonl")
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for field in ("name", "dataset", "revision", "split", "text-field"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--parent-field")
    parser.add_argument("--split-group-field")
    parser.add_argument("--max-chars", type=int)
    args = vars(parser.parse_args())
    print(json.dumps(prepare_snapshot(**args), indent=2))


if __name__ == "__main__":
    main()
