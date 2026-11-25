"""Token blocks and source-derived alignment targets for Stage A4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stages.source_data import source_blocks


@dataclass(frozen=True)
class A4Data:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray | None
    targets: np.ndarray
    provenance: dict[str, Any]
    train_mask: np.ndarray | None = None
    validation_mask: np.ndarray | None = None
    test_mask: np.ndarray | None = None

    def batch_labels(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return token positions and bank indices for selected training blocks."""
        targets = self.targets[indices]
        positions = np.column_stack((np.arange(len(indices)), targets[:, 0])).astype(
            np.int32
        )
        return positions, targets[:, 1]


def load_data(args: Any, tokenizer: Any, bank: Any, training: Any) -> A4Data:
    """Validate split separation, bank identity and complete firing-label coverage."""
    payload = args.alignment_labels.read_bytes()
    manifest = json.loads(payload)
    if (
        isinstance(manifest, dict)
        and manifest.get("contract_version") == "alignment-occurrences-v2"
    ):
        return _load_occurrences(args, tokenizer, bank, training, manifest, payload)
    arrays, hashes, provenance = {}, {}, {}
    splits = (
        ("train", "validation") if args.skip_test else ("train", "validation", "test")
    )
    for split in splits:
        arrays[split], hashes[split], provenance[split] = source_blocks(
            getattr(args, f"{split}_sources"),
            tokenizer,
            training.seq_len,
            args.max_train_rows if split == "train" else args.max_eval_rows,
            args.max_train_blocks if split == "train" else args.max_eval_blocks,
        )
        if len(arrays[split]) < training.batch_size:
            raise ValueError(
                f"{split}: too few blocks for batch_size={training.batch_size}"
            )
    if any(
        hashes[a] & hashes[b] for i, a in enumerate(splits) for b in splits[i + 1 :]
    ):
        raise ValueError("Identical source text occurs in multiple splits")
    payload = args.alignment_labels.read_bytes()
    labels = json.loads(payload)
    if not isinstance(labels, dict) or not isinstance(labels.get("labels"), list):
        raise ValueError("Alignment manifest must contain a labels list")
    if labels.get("contract_version") != "alignment-labels-v1":
        raise ValueError("Unsupported alignment-label manifest version")
    if labels.get("bank_snapshot_sha256") != bank.input_snapshot_sha256:
        raise ValueError("Alignment labels belong to another bank")
    if labels.get("token_blocks_sha256") != provenance["train"]["token_blocks_sha256"]:
        raise ValueError("Alignment labels belong to different training blocks")
    lookup = {record.template_id: index for index, record in enumerate(bank.records)}
    targets: np.ndarray = np.full((len(arrays["train"]), 2), -1, dtype=np.int32)
    for label in labels.get("labels", []):
        if not isinstance(label, dict) or not {
            "block_index",
            "token_index",
            "template_id",
        }.issubset(label):
            raise ValueError(
                "Alignment labels require block, token and template fields"
            )
        block, token = label["block_index"], label["token_index"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (block, token)
        ) or not (0 <= block < len(targets) and 0 <= token < training.seq_len):
            raise ValueError("Invalid alignment-label position")
        if targets[block, 0] != -1:
            raise ValueError("Duplicate alignment label for a training block")
        if label["template_id"] not in lookup:
            raise ValueError("Alignment label references an unknown template")
        targets[block] = token, lookup[label["template_id"]]
    missing = int(np.sum(targets[:, 0] < 0))
    if missing:
        raise ValueError(
            f"{missing} training blocks lack firing labels; supply a fully labelled training subset"
        )
    minimum = getattr(args, "minimum_examples", 1)
    if len(targets) < minimum:
        raise ValueError(
            f"Prepared data has {len(targets)} examples; requires {minimum}"
        )
    provenance["alignment"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "labelled_blocks": len(targets),
        "selection_policy": labels.get("selection_policy"),
    }
    return A4Data(
        arrays["train"], arrays["validation"], arrays.get("test"), targets, provenance
    )


def padded_source_blocks(path, tokenizer, seq_len, max_rows, max_blocks):
    """Keep independent causal document chunks, including right-padded tails."""
    if seq_len < 2 or max_rows < 1 or max_blocks < 1:
        raise ValueError("Invalid source block limits")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    blocks, masks, selected, hashes = [], [], [], set()
    payload = path.read_bytes()
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["source_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{number}: empty source text")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in hashes or len(selected) >= max_rows or len(blocks) >= max_blocks:
            continue
        tokens = tokenizer(text)["input_ids"]
        if len(tokens) < 2:
            continue
        hashes.add(digest)
        selected.append({"document_id": row.get("document_id"), "text_sha256": digest})
        for start in range(0, len(tokens) - 1, seq_len - 1):
            chunk = tokens[start : start + seq_len]
            if len(chunk) < 2 or len(blocks) >= max_blocks:
                break
            blocks.append(chunk + [pad] * (seq_len - len(chunk)))
            masks.append([True] * len(chunk) + [False] * (seq_len - len(chunk)))
    if not blocks:
        raise ValueError(f"{path}: no next-token evaluation examples")
    return (
        np.asarray(blocks, dtype=np.int32),
        np.asarray(masks, dtype=bool),
        hashes,
        {
            "path": str(path),
            "shape": [len(blocks), seq_len],
            "selected": selected,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "boundary_policy": "independent documents; one-token chunk overlap; right padding; masked next-token targets",
        },
    )


def _load_occurrences(args, tokenizer, bank, training, manifest, payload) -> A4Data:
    if manifest.get("bank_snapshot_sha256") != bank.input_snapshot_sha256:
        raise ValueError("Alignment examples belong to another bank")
    if manifest.get("seq_len") != training.seq_len:
        raise ValueError("Prepared sequence length differs from training")
    source_payload = args.train_sources.read_bytes()
    if hashlib.sha256(source_payload).hexdigest() != manifest.get(
        "source_sha256", {}
    ).get("sources"):
        raise ValueError("Prepared training sources changed")
    name = manifest.get("arrays_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Prepared arrays must be adjacent to their manifest")
    arrays_path = args.alignment_labels.parent / name
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != manifest.get(
        "arrays_sha256"
    ):
        raise ValueError("Prepared training arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"input_ids", "token_mask", "targets"}:
            raise ValueError("Invalid occurrence array fields")
        ids, mask, targets = (
            archive[key] for key in ("input_ids", "token_mask", "targets")
        )
    labels = manifest.get("labels")
    count = len(ids)
    if (
        ids.dtype != np.int32
        or ids.shape != (count, training.seq_len)
        or mask.dtype != np.bool_
        or mask.shape != ids.shape
        or targets.dtype != np.int32
        or targets.shape != (count, 2)
        or not isinstance(labels, list)
        or len(labels) != count
        or np.any(mask[:, 1:] & ~mask[:, :-1])
        or np.any(mask.sum(axis=1) < 2)
    ):
        raise ValueError("Invalid right-padded occurrence arrays")
    minimum = getattr(args, "minimum_examples", 5000)
    if count < minimum or count < training.batch_size:
        raise ValueError(
            f"Prepared data has {count} examples; requires {minimum} and one full batch"
        )
    if count > args.max_train_blocks:
        raise ValueError(
            "Prepared examples exceed --max-train-blocks; do not silently truncate"
        )
    documents = {}
    train_hashes = set()
    for line in source_payload.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = (row["corpus_id"], row["document_id"])
        if identity in documents:
            raise ValueError("Duplicate training source identity")
        text = row["source_text"]
        documents[identity] = (text, tokenizer(text, return_offsets_mapping=True))
        train_hashes.add(hashlib.sha256(text.encode()).hexdigest())
    if len(documents) > args.max_train_rows:
        raise ValueError("Prepared source count exceeds --max-train-rows")
    lookup = {record.template_id: index for index, record in enumerate(bank.records)}
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    seen = set()
    for index, label in enumerate(labels):
        identity = (label["corpus_id"], label["document_id"])
        if identity not in documents or label["template_id"] not in lookup:
            raise ValueError("Occurrence refers to an unknown source or template")
        text, encoded = documents[identity]
        start, end, position = (
            label["window_start"],
            label["window_end"],
            label["token_index"],
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (start, end, position)
        ) or not (
            0 <= start < end <= len(encoded["input_ids"])
            and end - start <= training.seq_len
            and 0 <= position < end - start
        ):
            raise ValueError("Invalid occurrence window or label position")
        expected = encoded["input_ids"][start:end]
        if (
            not np.array_equal(
                ids[index], expected + [pad] * (training.seq_len - len(expected))
            )
            or not np.array_equal(
                mask[index], np.arange(training.seq_len) < len(expected)
            )
            or not np.array_equal(
                targets[index], [position, lookup[label["template_id"]]]
            )
        ):
            raise ValueError(
                "Prepared occurrence differs from its source/tokenizer/target"
            )
        spans = label["source_spans"]
        if not spans or any(
            not (0 <= span["start"] < span["end"] <= len(text))
            or text[span["start"] : span["end"]] != span["text"]
            for span in spans
        ):
            raise ValueError("Occurrence evidence differs from source text")
        first = min(span["start"] for span in spans)
        last = max(span["end"] for span in spans)
        offsets = encoded["offset_mapping"]
        covered = [
            i
            for i, (left, right) in enumerate(offsets)
            if right > first and left < last
        ]
        if not covered or min(covered) < start or max(covered) != start + position:
            raise ValueError("Label must follow complete evidence inside its window")
        key = (identity, start, position, label["template_id"])
        if key in seen:
            raise ValueError("Duplicate occurrence-template training pair")
        seen.add(key)
    provenance = {
        "train": {
            "path": str(args.train_sources),
            "shape": list(ids.shape),
            "documents": len(documents),
        },
        "alignment": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "examples": count,
            "distinct_templates": len(set(targets[:, 1].tolist())),
            "selection_policy": manifest.get("selection_policy"),
            "coverage": manifest.get("coverage"),
        },
    }
    split_arrays, split_masks = {}, {}
    occupied = train_hashes | set(manifest.get("bank_source_text_sha256", []))
    for split in ("validation",) if args.skip_test else ("validation", "test"):
        values, valid, hashes, metadata = padded_source_blocks(
            getattr(args, f"{split}_sources"),
            tokenizer,
            training.seq_len,
            args.max_eval_rows,
            args.max_eval_blocks,
        )
        if len(values) < training.batch_size or hashes & occupied:
            raise ValueError(
                f"{split}: too few examples or source overlap with training/bank/another split"
            )
        occupied.update(hashes)
        split_arrays[split], split_masks[split], provenance[split] = (
            values,
            valid,
            metadata,
        )
    return A4Data(
        ids,
        split_arrays["validation"],
        split_arrays.get("test"),
        targets,
        provenance,
        mask,
        split_masks["validation"],
        split_masks.get("test"),
    )
