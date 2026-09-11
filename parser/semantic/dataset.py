"""parser/semantic/dataset.py

Validated JSONL dataset generation for semantic parser distillation.

This file provides:
  1. Structured JSON generation from teacher (ReferenceSemanticParser)
  2. Conversion to clean (text, metta_expr) pairs with confidence filtering

"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from parser.semantic.normalization import normalize_semantic_result
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
)
from parser.semantic.student_prompt import build_student_prompt

# ── Dataset records ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DistillationRecord:
    """One accepted text-to-structured-JSON teacher example."""

    text: str
    target: dict[str, Any]
    teacher_provider: str
    teacher_model: str
    prompt_version: str
    metta: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """One source sentence rejected by the reference pipeline."""

    text: str
    error: str


# ── Dataset generator ──────────────────────────────────────────────────────────


class SemanticDatasetBuilder:
    """Build validated distillation data with a ReferenceSemanticParser."""

    def __init__(
        self,
        parser: ReferenceSemanticParser,
        *,
        include_metta: bool = False,
    ) -> None:
        """Store the parser used to label source sentences."""
        self._parser = parser
        self._include_metta = include_metta

    def generate(
        self,
        sentences: Iterable[str],
    ) -> tuple[list[DistillationRecord], list[RejectedRecord]]:
        """Parse sentences and separate accepted from rejected examples."""
        accepted = []
        rejected = []

        for sentence in sentences:
            text = sentence.strip()
            if not text:
                rejected.append(
                    RejectedRecord(
                        text=sentence,
                        error="The sentence cannot be empty",
                    )
                )
                continue

            try:
                result = self._parser.generate_structured(text)
                expressions = self._parser.render_metta(result)
                self._parser.validate_rendered_metta(expressions)
            except (ModelGenerationError, SemanticParseError, ValueError) as error:
                rejected.append(RejectedRecord(text=text, error=str(error)))
                continue

            accepted.append(
                DistillationRecord(
                    text=text,
                    target=result.model_dump(mode="json"),
                    teacher_provider=self._parser.provider_name,
                    teacher_model=self._parser.model_name,
                    prompt_version=self._parser.prompt_version,
                    metta=tuple(expressions) if self._include_metta else None,
                )
            )

        return accepted, rejected

    @staticmethod
    def write_jsonl(
        records: Iterable[DistillationRecord],
        output_path: str | Path,
    ) -> None:
        """Write accepted records as UTF-8 JSON Lines."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            for record in records:
                payload = asdict(record)
                if record.metta is None:
                    payload.pop("metta")
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def write_rejected_jsonl(
        records: Iterable[RejectedRecord],
        output_path: str | Path,
    ) -> None:
        """Write rejected records as UTF-8 JSON Lines."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


# ── Student Dataset for training ─────────────────────────────────────────────


def validate_student_pair(pair: Any) -> None:
    """Reject malformed input/label records before tokenization or splitting."""
    if not isinstance(pair, dict) or any(
        not isinstance(pair.get(key), str) or not pair[key].strip()
        for key in ("input", "label")
    ):
        raise ValueError("input and label must be nonempty strings")
    if not isinstance(json.loads(pair["label"]), dict):
        raise ValueError("label must contain a JSON object")


class StudentDataset(Dataset):
    """Prepares (input, json_label) pairs for student model training."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 256):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("Student training requires an EOS token")
        self.examples = []
        self.oversized = 0
        skipped = 0

        for pair in pairs:
            try:
                validate_student_pair(pair)
            except ValueError:
                skipped += 1
                continue
            text = pair.get("input", "").strip()
            json_label = pair.get("label", "").strip()

            prompt_text = build_student_prompt(text)
            # Encode the boundary separately: prompt tokens must match inference,
            # and every target token must contribute to the loss.
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(json_label, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_length:
                self.oversized += 1
                continue

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )

        if self.oversized:
            print(
                f"  Skipped {self.oversized} oversized examples (max_length={max_length})"
            )

        if skipped > 0:
            print(f"  Skipped {skipped} invalid examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


@dataclass(frozen=True)
class StudentBatchCollator:
    """Right-pad inputs while excluding padding from attention and loss."""

    pad_token_id: int

    def __call__(
        self, examples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        padding = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        return {
            key: pad_sequence(
                [example[key] for example in examples],
                batch_first=True,
                padding_value=value,
            )
            for key, value in padding.items()
        }


# ── Converter: Structured → Pairs (for student training) ─────────────────────


def structured_to_pairs(
    input_file: str,
    output_file: str,
    max_per_text: int = 1,
    include_confidence: bool = False,
) -> list[dict]:
    """Convert teacher structured output to (input, json_label) pairs."""
    source, destination = Path(input_file), Path(output_file)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("Input and output paths must differ")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    total = 0
    valid = 0
    skipped = 0

    with (
        open(input_file, encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                skipped += 1
                continue
            text = record["text"].strip()
            if not text:
                skipped += 1
                continue

            # Get structured output
            structured = (
                record.get("target")
                or record.get("structured_output")
                or record.get("structured")
            )
            if not structured:
                skipped += 1
                continue

            try:
                result = SemanticParseResult.model_validate(structured)
            except Exception:
                skipped += 1
                continue

            try:
                normalized = normalize_semantic_result(result)
            except Exception:
                skipped += 1
                continue

            # Get confidence
            confidence = (
                normalized.assertions[0].confidence if normalized.assertions else 1.0
            )

            # Save the ENTIRE JSON as the label (NOT MeTTa)
            json_label = json.dumps(structured, ensure_ascii=False)

            record_out = {
                "input": text,
                "label": json_label,  # ← JSON, not MeTTa!
            }
            if include_confidence:
                record_out["confidence"] = confidence

            pairs.append(record_out)
            fout.write(json.dumps(record_out) + "\n")
            valid += 1

    print(f"  Total: {total}, Valid: {valid}, Skipped: {skipped}")
    return pairs


def split_pairs(
    pairs: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deduplicate inputs, shuffle, and split using largest-remainder rounding."""

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("Split ratios must be finite numbers between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Train, validation and test ratios must sum to 1.0")

    unique_pairs = []
    seen = set()

    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict) or not isinstance(pair.get("input", ""), str):
            raise ValueError(f"Pair {index} must be an object with a string input")
        text = pair.get("input", "").strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        unique_pairs.append(pair)

    rng = random.Random(seed)
    rng.shuffle(unique_pairs)

    total = len(unique_pairs)
    # Normalize tolerated floating-point error before assigning all records.
    sizes = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [math.floor(size) for size in sizes]
    order = sorted(range(3), key=lambda i: sizes[i] - counts[i], reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    train_end = counts[0]
    val_end = train_end + counts[1]

    train_pairs = unique_pairs[:train_end]
    val_pairs = unique_pairs[train_end:val_end]
    test_pairs = unique_pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


def verify_split_overlap(
    train_pairs: list[dict],
    val_pairs: list[dict],
    test_pairs: list[dict],
) -> None:
    """Ensure the same input text does not appear across dataset splits."""

    train_texts = {pair["input"].strip() for pair in train_pairs}
    val_texts = {pair["input"].strip() for pair in val_pairs}
    test_texts = {pair["input"].strip() for pair in test_pairs}

    if not train_texts.isdisjoint(val_texts):
        raise ValueError("Train and validation sets overlap")

    if not train_texts.isdisjoint(test_texts):
        raise ValueError("Train and test sets overlap")

    if not val_texts.isdisjoint(test_texts):
        raise ValueError("Validation and test sets overlap")


def write_pairs_jsonl(pairs: list[dict], output_file: str | Path) -> None:
    """Write student input-label pairs to JSONL"""

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
