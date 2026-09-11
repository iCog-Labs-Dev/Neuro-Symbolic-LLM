"""parser/semantic/dataset.py

Validated JSONL dataset generation for semantic parser distillation.

This file provides:
  1. Structured JSON generation from teacher (ReferenceSemanticParser)
  2. Conversion to clean (text, metta_expr) pairs with confidence filtering

"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from parser.semantic.normalization import normalize_semantic_result
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
)

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


class StudentDataset(Dataset):
    """Prepares (input, json_label) pairs for student model training."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 256):
        self.examples = []
        skipped = 0

        for pair in pairs:
            text = pair.get("input", "").strip()
            json_label = pair.get("label", "").strip()

            if not text or not json_label:
                skipped += 1
                continue

            # Validate JSON label
            try:
                json.loads(json_label)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt_text = (
                f"Extract triples from this sentence as JSON:\nSentence: {text}\nJSON:"
            )
            full_text = prompt_text + json_label + tokenizer.eos_token

            full_enc = tokenizer(
                full_text, truncation=True, max_length=max_length, padding=False
            )
            prompt_enc = tokenizer(
                prompt_text, truncation=True, max_length=max_length, padding=False
            )

            prompt_len = len(prompt_enc["input_ids"])
            input_ids = full_enc["input_ids"]

            # Mask prompt tokens: -100 means "ignore for loss"
            labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]
            labels = labels[:max_length]

            if all(i == -100 for i in labels):
                skipped += 1
                continue

            self.examples.append(
                {
                    "input_ids": input_ids[:max_length],
                    "attention_mask": full_enc["attention_mask"][:max_length],
                    "labels": labels,
                }
            )

        if skipped > 0:
            print(f"  Skipped {skipped} invalid examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


# ── Converter: Structured → Pairs (for student training) ─────────────────────


def structured_to_pairs(
    input_file: str,
    output_file: str,
    max_per_text: int = 1,
    include_confidence: bool = False,
) -> list[dict]:
    """Convert teacher structured output to (input, json_label) pairs."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    total = 0
    valid = 0
    skipped = 0

    with open(input_file) as fin, open(output_file, "w") as fout:
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

            text = record.get("text", "").strip()
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
