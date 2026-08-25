"""parser/semantic/dataset.py

Validated JSONL dataset generation for semantic parser distillation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
