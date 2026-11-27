"""Token blocks and source-derived alignment targets for Stage A4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from stages.source_data import source_blocks

ROCSTORIES_TASK_IDS = (1, 2, 3)
ROCSTORIES_EVAL_GROUPS = tuple(
    f"rocstories_held_out_group_{task_id}" for task_id in ROCSTORIES_TASK_IDS
)
HELD_OUT_GROUP_SCOPE = "fixed_held_out_groups"
HELD_OUT_GROUP_INTERPRETATION = (
    "These metrics describe three fixed held-out evaluation groups from the "
    "same ROCStories corpus. Each group is paired with the training step of "
    "the same number for metric calculation. The groups do not represent "
    "independent semantic domains."
)


def _batch_labels(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.column_stack((np.arange(len(targets)), targets[:, 0])).astype(
        np.int32
    )
    return positions, targets[:, 1]


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
        return _batch_labels(self.targets[indices])


@dataclass(frozen=True)
class AlignmentTask:
    """Training arrays and per-row occurrence provenance for one source task."""

    name: str
    epochs: int
    train_blocks: np.ndarray
    train_mask: np.ndarray
    targets: np.ndarray
    labels: tuple[dict[str, Any], ...]

    def batch_inputs(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions, targets = _batch_labels(self.targets[indices])
        return self.train_blocks[indices], positions, targets, self.train_mask[indices]


@dataclass(frozen=True)
class SequentialAlignmentData:
    """ROCStories tasks, shared splits and fixed held-out groups."""

    tasks: tuple[AlignmentTask, ...]
    validation: np.ndarray
    validation_mask: np.ndarray
    test: np.ndarray | None
    test_mask: np.ndarray | None
    provenance: dict[str, Any]
    task_eval: dict[str, tuple[np.ndarray, np.ndarray]]


def load_data(
    args: Any, tokenizer: Any, bank: Any, training: Any
) -> A4Data | SequentialAlignmentData:
    """Validate split separation, bank identity and complete firing-label coverage."""
    payload = args.alignment_labels.read_bytes()
    manifest = json.loads(payload)
    sequential = getattr(args, "data_mode", None) == "sequential-aligned"
    if (
        isinstance(manifest, dict)
        and manifest.get("contract_version") == "alignment-occurrences-v2"
    ):
        return _load_occurrences(
            args, tokenizer, bank, training, manifest, payload, sequential=sequential
        )
    if sequential:
        raise ValueError(
            "Sequential alignment requires an alignment-occurrences-v2 manifest"
        )
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
    payload = path.read_bytes()
    rows = (
        (number, json.loads(line))
        for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1)
        if line.strip()
    )
    return _padded_source_rows(
        path, payload, rows, tokenizer, seq_len, max_rows, max_blocks
    )


def _padded_source_rows(path, payload, rows, tokenizer, seq_len, max_rows, max_blocks):
    if seq_len < 2 or max_rows < 1 or max_blocks < 1:
        raise ValueError("Invalid source block limits")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    blocks, masks, selected, hashes = [], [], [], set()
    for number, row in rows:
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
        block_start = len(blocks)
        for start in range(0, len(tokens) - 1, seq_len - 1):
            chunk = tokens[start : start + seq_len]
            if len(chunk) < 2 or len(blocks) >= max_blocks:
                break
            blocks.append(chunk + [pad] * (seq_len - len(chunk)))
            masks.append([True] * len(chunk) + [False] * (seq_len - len(chunk)))
        selected.append(
            {
                "document_id": row.get("document_id"),
                "text_sha256": digest,
                "block_start": block_start,
                "blocks": len(blocks) - block_start,
            }
        )
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


def _source_identity(row):
    identity = (row.get("corpus_id"), row.get("document_id"))
    if any(
        part is None or isinstance(part, str) and not part.strip() for part in identity
    ):
        raise ValueError("Occurrence sources and labels require a source identity")
    return identity


def _partition_validation_groups(blocks, mask, metadata):
    """Assign selected validation documents in hash order, keeping all chunks together."""
    documents = sorted(metadata["selected"], key=lambda row: row["text_sha256"])
    group_count = len(ROCSTORIES_EVAL_GROUPS)
    if len(documents) < group_count:
        raise ValueError(
            "Sequential-aligned validation requires at least three distinct "
            "held-out documents within the evaluation limits"
        )

    task_eval, provenance = {}, {}
    for index, name in enumerate(ROCSTORIES_EVAL_GROUPS):
        selected = documents[index::group_count]
        indices = [
            block_index
            for row in selected
            for block_index in range(
                row["block_start"], row["block_start"] + row["blocks"]
            )
        ]
        task_eval[name] = (blocks[indices], mask[indices])
        provenance[name] = {
            "path": metadata["path"],
            "source_sha256": metadata["source_sha256"],
            "block_index_scope": "shared_validation",
            "selected": selected,
            "shape": list(task_eval[name][0].shape),
        }
    return task_eval, provenance


def _load_task_eval(path, tokenizer, training, args, occupied, train_identities):
    """Load explicitly assigned groups and verify held-out provenance before packing."""
    payload = path.read_bytes()
    groups = {task_id: [] for task_id in ROCSTORIES_TASK_IDS}
    identities = set(train_identities)
    hashes = set(occupied)
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id not in groups
        ):
            raise ValueError(f"{path}:{number}: invalid held-out task ID")
        identity = _source_identity(row)
        text = row.get("source_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{number}: empty source text")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if identity in identities or digest in hashes:
            raise ValueError(
                f"{path}:{number}: held-out group source overlaps another source"
            )
        identities.add(identity)
        hashes.add(digest)
        groups[task_id].append((number, row))

    if not all(groups.values()):
        raise ValueError("Held-out sources must contain all three ROCStories task IDs")

    task_eval, provenance = {}, {}
    for name, rows in zip(ROCSTORIES_EVAL_GROUPS, groups.values(), strict=True):
        blocks, mask, _, metadata = _padded_source_rows(
            path,
            payload,
            rows,
            tokenizer,
            training.seq_len,
            args.max_eval_rows,
            args.max_eval_blocks,
        )
        task_eval[name] = (blocks, mask)
        provenance[name] = metadata
    return task_eval, provenance


def _load_occurrences(
    args, tokenizer, bank, training, manifest, payload, *, sequential
) -> A4Data | SequentialAlignmentData:
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
    arrays_payload = arrays_path.read_bytes()
    if hashlib.sha256(arrays_payload).hexdigest() != manifest.get("arrays_sha256"):
        raise ValueError("Prepared training arrays changed")
    with np.load(BytesIO(arrays_payload), allow_pickle=False) as archive:
        if set(archive.files) != {"input_ids", "token_mask", "targets"}:
            raise ValueError("Invalid occurrence array fields")
        ids, mask, targets = (
            archive[key] for key in ("input_ids", "token_mask", "targets")
        )
    labels = manifest.get("labels")
    count = ids.shape[0] if ids.ndim else 0
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
    if np.any(targets[:, 1] < 0) or np.any(targets[:, 1] >= len(bank.records)):
        raise ValueError("Occurrence target index is outside the bank")
    minimum = getattr(args, "minimum_examples", 5000)
    required_examples = minimum if sequential else max(minimum, training.batch_size)
    if count < required_examples:
        raise ValueError(
            f"Prepared data has {count} examples; requires at least "
            f"{required_examples}"
        )
    if count > args.max_train_blocks:
        raise ValueError(
            "Prepared examples exceed --max-train-blocks; do not silently truncate"
        )
    documents = {}
    document_tasks = {}
    task_rows: dict[int, list[int]] = {task_id: [] for task_id in ROCSTORIES_TASK_IDS}
    task_documents: dict[int, set[tuple[Any, Any]]] = {
        task_id: set() for task_id in ROCSTORIES_TASK_IDS
    }
    train_hashes = set()
    for line in source_payload.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = _source_identity(row)
        if identity in documents:
            raise ValueError("Duplicate training source identity")
        text = row["source_text"]
        if sequential:
            task_id = row.get("task_id")
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id not in ROCSTORIES_TASK_IDS
            ):
                raise ValueError(
                    f"Invalid ROCStories task ID for source {identity}: {task_id!r}"
                )
            document_tasks[identity] = task_id
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
        if not isinstance(label, dict):
            raise ValueError("Occurrence labels must be objects")
        identity = _source_identity(label)
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
        if sequential:
            task_id = document_tasks[identity]
            task_rows[task_id].append(index)
            task_documents[task_id].add(identity)
    provenance: dict[str, Any] = {
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
    if sequential:
        if not all(task_rows.values()) or sum(map(len, task_rows.values())) != count:
            raise ValueError(
                "Every occurrence must be assigned to exactly one of the three ROCStories tasks"
            )
        tasks = tuple(
            AlignmentTask(
                name=f"rocstories_task_{task_id}",
                epochs=training.epochs,
                train_blocks=ids[rows],
                train_mask=mask[rows],
                targets=targets[rows],
                labels=tuple(labels[row] for row in rows),
            )
            for task_id, rows in task_rows.items()
        )
        provenance.update(
            validation_scope=HELD_OUT_GROUP_SCOPE,
            task_order=[task.name for task in tasks],
            task_mapping=(
                "Original ROCStories task_id joined by (corpus_id, document_id). "
                "A3 runtime tasks used shuffled grouping."
            ),
            alignment_counts={
                task.name: {
                    "task_id": task_id,
                    "examples": len(task.train_blocks),
                    "source_documents": len(task_documents[task_id]),
                }
                for task_id, task in zip(ROCSTORIES_TASK_IDS, tasks, strict=True)
            },
            source_fingerprints={
                **manifest["source_sha256"],
                **{split: provenance[split]["source_sha256"] for split in split_arrays},
                "alignment_manifest": provenance["alignment"]["sha256"],
                "alignment_arrays": manifest["arrays_sha256"],
            },
        )
        task_eval_path = getattr(args, "task_eval_sources", None)
        if task_eval_path is None:
            task_eval, task_eval_provenance = _partition_validation_groups(
                split_arrays["validation"],
                split_masks["validation"],
                provenance["validation"],
            )
            assignment = (
                "Round robin over selected validation documents sorted by text SHA-256"
            )
        else:
            task_eval, task_eval_provenance = _load_task_eval(
                task_eval_path, tokenizer, training, args, occupied, documents
            )
            assignment = "Explicit task_id from separate held-out source file"
        provenance["task_eval"] = task_eval_provenance
        provenance["source_fingerprints"]["task_eval"] = task_eval_provenance[
            ROCSTORIES_EVAL_GROUPS[0]
        ]["source_sha256"]
        provenance["continual_evaluation"] = {
            "scope": HELD_OUT_GROUP_SCOPE,
            "interpretation": HELD_OUT_GROUP_INTERPRETATION,
            "group_order": list(ROCSTORIES_EVAL_GROUPS),
            "assignment": assignment,
        }
        return SequentialAlignmentData(
            tasks,
            split_arrays["validation"],
            split_masks["validation"],
            split_arrays.get("test"),
            split_masks.get("test"),
            provenance,
            task_eval,
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
