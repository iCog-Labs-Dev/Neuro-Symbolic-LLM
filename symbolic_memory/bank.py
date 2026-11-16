"""Associate miner pattern records with offline embeddings by ID."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hybrid_miner.contracts import PatternRecord

from residual.symbolic_head.contracts.template_record import (
    TemplateRecord,
    validate_template_record,
)


@dataclass(frozen=True)
class TemplateBank:
    """Template records and SHA-256 fingerprints of their source artifacts."""

    records: tuple[TemplateRecord, ...]
    patterns_sha256: str
    embeddings_sha256: str
    patterns: tuple[PatternRecord, ...] = ()

    @property
    def vector_dim(self) -> int:
        return int(self.records[0].key_vector.shape[0])

    @property
    def input_snapshot_sha256(self) -> str:
        """Hash the pattern and embedding artifact fingerprints as one identity."""
        identity = json.dumps(
            {
                "patterns_sha256": self.patterns_sha256,
                "embeddings_sha256": self.embeddings_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _objects(payload: bytes, name: str) -> Iterator[tuple[int, dict[str, Any]]]:
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{name}:{number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{number}: expected a JSON object")
        yield number, row


def _patterns(payload: bytes, name: str) -> tuple[PatternRecord, ...]:
    result: list[PatternRecord] = []
    seen: set[str] = set()
    for number, row in _objects(payload, name):
        try:
            pattern = PatternRecord.model_validate(row)
            if pattern.id in seen:
                raise ValueError(f"duplicate pattern id {pattern.id!r}")
        except ValueError as error:
            raise ValueError(f"{name}:{number}: {error}") from error
        seen.add(pattern.id)
        result.append(pattern)
    if not result:
        raise ValueError(f"{name}: no pattern records")
    return tuple(result)


def read_patterns(path: Path) -> tuple[PatternRecord, ...]:
    """Parse PatternRecord JSONL using the miner's record contract."""
    return _patterns(path.read_bytes(), path.name)


def _vector(row: dict[str, Any], field: str) -> np.ndarray:
    value = row.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty numeric vector")
    if any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        raise ValueError(
            f"{field} must contain numbers, not strings, booleans or arrays"
        )
    with np.errstate(over="raise", invalid="raise"):
        vector: np.ndarray = np.asarray(value, dtype=np.float32)
    return vector


def load_bank(patterns_path: Path, embeddings_path: Path) -> TemplateBank:
    """Join pattern and embedding JSONL artifacts by ID.

    Vectors retain their supplied values as float32 arrays. Producer metadata
    is preserved without projection or normalization.
    """
    patterns_bytes = patterns_path.read_bytes()
    embeddings_bytes = embeddings_path.read_bytes()
    patterns = _patterns(patterns_bytes, patterns_path.name)
    by_id = {pattern.id: pattern for pattern in patterns}
    records: dict[str, TemplateRecord] = {}
    dimension: int | None = None
    for number, row in _objects(embeddings_bytes, embeddings_path.name):
        try:
            identifier = row.get("pattern_id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("pattern_id must be a nonempty string")
            if identifier not in by_id:
                raise ValueError(f"unknown embedding pattern_id {identifier!r}")
            if identifier in records:
                raise ValueError(f"duplicate embedding pattern_id {identifier!r}")
            key = _vector(row, "key")
            value = _vector(row, "value")
            if dimension is None:
                dimension = len(key)
            declared_dimension = row.get("embedding_dim", dimension)
            if (
                isinstance(declared_dimension, bool)
                or not isinstance(declared_dimension, int)
                or declared_dimension != dimension
            ):
                raise ValueError(
                    "embedding_dim disagrees with the bank vector dimension"
                )
            identifier, expression, key, value = validate_template_record(
                identifier, by_id[identifier].canonical_term, key, value, dimension
            )
            key.setflags(write=False)
            value.setflags(write=False)
            records[identifier] = TemplateRecord(identifier, expression, key, value)
        except (ValueError, OverflowError, FloatingPointError) as error:
            raise ValueError(f"{embeddings_path.name}:{number}: {error}") from error
    missing = set(by_id) - records.keys()
    if missing:
        raise ValueError(
            f"{embeddings_path.name}: missing embeddings for {sorted(missing)}"
        )
    return TemplateBank(
        records=tuple(records[pattern.id] for pattern in patterns),
        patterns_sha256=hashlib.sha256(patterns_bytes).hexdigest(),
        embeddings_sha256=hashlib.sha256(embeddings_bytes).hexdigest(),
        patterns=patterns,
    )
