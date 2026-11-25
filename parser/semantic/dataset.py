"""Parse-result datasets and masked-loss student tokenization."""

from __future__ import annotations

import json
import math
import random
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.catalog_student import (
    make_student_pair,
    validate_catalog_student_pair,
)
from parser.semantic.document_metadata import (
    document_metadata,
    load_document_metadata,
    metadata_path,
    source_metadata,
)
from parser.semantic.metta_renderer import render_metta, validate_rendered_metta
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
)

# ── Dataset records ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DistillationRecord:
    """A teacher parse result with producer metadata and an optional MeTTa rendering."""

    text: str
    target: dict[str, Any]
    teacher_provider: str
    teacher_model: str
    prompt_version: str
    semantic_contract_version: str
    source_dataset: str | None = None
    source_domain: str | None = None
    metta: tuple[str, ...] | None = None
    rendering_error: str | None = None
    metta_asserted_only: bool = False

    @property
    def result(self) -> SemanticParseResult:
        return SemanticParseResult.model_validate(self.target)


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """One source sentence rejected by the reference pipeline."""

    text: str
    error: str


# ── Dataset generator ──────────────────────────────────────────────────────────


class SemanticDatasetBuilder:
    """Build validated teacher data for training the DistilledSemanticParser.

    A ReferenceSemanticParser parses each sentence; accepted results and
    rejected sentences are kept apart.
    """

    def __init__(
        self,
        parser: ReferenceSemanticParser,
        *,
        include_metta: bool = False,
        asserted_only: bool = False,
    ) -> None:
        """Store the parser and dataset-generation policy."""
        self._parser = parser
        self._include_metta = include_metta or asserted_only
        self._asserted_only = asserted_only

    def generate(
        self,
        sentences: Iterable[str],
        *,
        source_dataset: str | None = None,
        source_domain: str | None = None,
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
                result = self._parser.parse(sentence)
                if source_dataset is not None:
                    corpus_id = (
                        source_dataset
                        if not any(character.isspace() for character in source_dataset)
                        else "source-"
                        + sha256(source_dataset.encode("utf-8")).hexdigest()
                    )
                    result = SemanticParseResult.model_validate(
                        {
                            **result.model_dump(mode="json"),
                            "document": {
                                **result.document.model_dump(mode="json"),
                                "corpus_id": corpus_id,
                            },
                        }
                    )
                metta = None
                rendering_error = None
                if self._include_metta:
                    try:
                        expressions = render_metta(
                            result, asserted_only=self._asserted_only
                        )
                        validate_rendered_metta(expressions)
                        metta = tuple(expressions)
                    except ValueError as error:
                        rendering_error = str(error)
                accepted.append(
                    DistillationRecord(
                        text=sentence,
                        target=result.model_dump(mode="json"),
                        teacher_provider=self._parser.provider_name,
                        teacher_model=self._parser.model_name,
                        prompt_version=result.provenance.prompt_version,
                        semantic_contract_version=result.contract_version,
                        source_dataset=source_dataset,
                        source_domain=source_domain,
                        metta=metta,
                        rendering_error=rendering_error,
                        metta_asserted_only=self._asserted_only,
                    )
                )
            except (ModelGenerationError, SemanticParseError, ValueError) as error:
                rejected.append(RejectedRecord(text=sentence, error=str(error)))

        return accepted, rejected

    @staticmethod
    def write_jsonl(
        records: Iterable[DistillationRecord | SemanticParseResult],
        output_path: str | Path,
    ) -> None:
        """Write accepted records as UTF-8 JSON Lines."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sidecar = metadata_path(path)
        with (
            path.open("w", encoding="utf-8") as file,
            sidecar.open("w", encoding="utf-8") as metadata_file,
        ):
            for record in records:
                result = (
                    record.result if isinstance(record, DistillationRecord) else record
                )
                file.write(result.model_dump_json() + "\n")
                fields = {}
                if isinstance(record, DistillationRecord):
                    fields = {
                        name: value
                        for name in (
                            "source_dataset",
                            "source_domain",
                            "metta",
                            "rendering_error",
                            "metta_asserted_only",
                        )
                        if (value := getattr(record, name)) is not None
                    }
                metadata_file.write(
                    json.dumps(document_metadata(result, **fields), ensure_ascii=False)
                    + "\n"
                )

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


def validate_student_pair(pair: Any, *, catalog: OntologyCatalog | None = None) -> None:
    """Reject malformed input/label records before tokenization or splitting."""
    validate_catalog_student_pair(
        pair, catalog=catalog if catalog is not None else load_ontology_catalog()
    )


class StudentDataset(Dataset):
    """Prepares (input, json_label) pairs for student model training."""

    def __init__(
        self,
        pairs: list[dict],
        tokenizer,
        max_length: int | None = None,
        *,
        catalog: OntologyCatalog | None = None,
    ):
        if max_length is not None and (isinstance(max_length, bool) or max_length <= 0):
            raise ValueError("max_length must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("Student training requires an EOS token")
        self.examples = []
        self.oversized = 0
        self.required_max_length = 0
        active_catalog = catalog if catalog is not None else load_ontology_catalog()

        for pair_index, pair in enumerate(pairs, start=1):
            prompt_text, json_label, _ = validate_catalog_student_pair(
                pair, catalog=active_catalog
            )
            # Encode the boundary separately: prompt tokens must match inference,
            # and every target token must contribute to the loss.
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(json_label, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [tokenizer.eos_token_id]
            input_ids = prompt_ids + target_ids
            self.required_max_length = max(self.required_max_length, len(input_ids))
            if max_length is not None and len(input_ids) > max_length:
                raise ValueError(
                    f"Catalog pair {pair_index} needs {len(input_ids)} tokens; "
                    f"max_length is {max_length}. Targets are never truncated."
                )

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )

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
    *,
    catalog: OntologyCatalog | None = None,
    metadata_file: str | Path | None = None,
) -> list[dict]:
    """Convert parse-result JSONL into validated student pairs."""
    if max_per_text != 1:
        warnings.warn(
            "max_per_text is deprecated; full document labels are retained",
            DeprecationWarning,
            stacklevel=2,
        )
    source, destination = Path(input_file), Path(output_file)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("Input and output paths must differ")
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing output: {destination}")
    active_catalog = catalog if catalog is not None else load_ontology_catalog()
    sidecar = (
        Path(metadata_file) if metadata_file is not None else metadata_path(source)
    )
    if metadata_file is not None and not sidecar.is_file():
        raise FileNotFoundError(f"Metadata sidecar not found: {sidecar}")
    metadata = load_document_metadata(sidecar) if sidecar.exists() else None
    pairs = []
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                result = SemanticParseResult.model_validate_json(line)
                fields = {}
                if metadata is not None:
                    identity = result.document.corpus_id, result.document.document_id
                    if identity not in metadata:
                        raise ValueError(
                            "Document is missing from the metadata sidecar"
                        )
                    fields = source_metadata(result, metadata.pop(identity))
                pair = make_student_pair(
                    result,
                    source_dataset=fields.get("source_dataset"),
                    source_domain=fields.get("source_domain"),
                )
                if include_confidence:
                    assertions = result.extraction.assertions
                    pair["confidence"] = assertions[0].confidence if assertions else 1.0
                _, _, reconstructed = validate_catalog_student_pair(
                    pair, catalog=active_catalog
                )
                if (
                    reconstructed.document != result.document
                    or reconstructed.extraction != result.extraction
                ):
                    raise ValueError("Conversion would change the document")
                pairs.append(pair)
            except ValueError as error:
                raise ValueError(
                    f"Invalid document at {source}:{line_number}: {error}"
                ) from error
    if metadata:
        raise ValueError("Metadata sidecar contains unmatched documents")
    write_pairs_jsonl(pairs, destination)
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
