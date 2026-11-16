"""Source JSONL token blocks shared by training and firing-label preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def source_blocks(
    path: Path, tokenizer: Any, seq_len: int, max_rows: int, max_blocks: int
) -> tuple[np.ndarray, set[str], dict[str, Any]]:
    """Tokenize whole source rows independently and discard incomplete tails."""
    if seq_len < 2 or max_rows < 1 or max_blocks < 1:
        raise ValueError("Source block limits must be positive and seq_len >= 2")
    blocks: list[np.ndarray] = []
    text_hashes: set[str] = set()
    selected: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open(encoding="utf-8-sig") as stream:
        for row_number, line in enumerate(stream, 1):
            digest.update(line.encode("utf-8"))
            if (
                not line.strip()
                or len(selected) >= max_rows
                or len(blocks) >= max_blocks
            ):
                continue
            row = json.loads(line)
            text = row["source_text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{row_number}: empty source text")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in text_hashes:
                continue
            ids = np.asarray(tokenizer(text)["input_ids"], dtype=np.int32)
            count = min(len(ids) // seq_len, max_blocks - len(blocks))
            if not count:
                continue
            blocks.extend(ids[: count * seq_len].reshape(count, seq_len))
            text_hashes.add(text_hash)
            selected.append(
                {
                    "row": row_number,
                    "text_sha256": text_hash,
                    "blocks": count,
                    "document_id": row.get("document_id"),
                    "block_start": len(blocks) - count,
                }
            )
    if not blocks:
        raise ValueError(f"{path}: no complete token blocks")
    array = np.stack(blocks)
    return (
        array,
        text_hashes,
        {
            "path": str(path),
            "decoded_source_sha256": digest.hexdigest(),
            "selected": selected,
            "shape": list(array.shape),
            "token_blocks_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "boundary_policy": "independent documents; incomplete token tails discarded",
        },
    )
