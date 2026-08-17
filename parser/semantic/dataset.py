"""parser/semantic/dataset.py

Validated JSONL dataset generation for semantic parser distillation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from parser.semantic.semantic_parser import ReferenceSemanticParser, SemanticParseError

# ── Dataset records ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DistillationRecord:
    """One accepted text-to-MeTTa teacher example."""

    text: str
    metta: str
    teacher_provider: str
    teacher_model: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """One source sentence rejected by the reference pipeline."""

    text: str
    error: str


# ── Dataset generator ──────────────────────────────────────────────────────────


class SemanticDatasetBuilder:
    """Build validated distillation data with a ReferenceSemanticParser."""

    def __init__(self, parser: ReferenceSemanticParser) -> None:
        """Store the parser used to label source sentences."""
        self._parser = parser

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
                atoms = self._parser.parse(text)
            except (SemanticParseError, ValueError) as error:
                rejected.append(RejectedRecord(text=text, error=str(error)))
                continue

            accepted.append(
                DistillationRecord(
                    text=text,
                    metta="\n".join(map(str, atoms)),
                    teacher_provider=self._parser.provider_name,
                    teacher_model=self._parser.model_name,
                    prompt_version=self._parser.prompt_version,
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
                file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
