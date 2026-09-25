"""Canonical mined-pattern contract."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SemanticLevel = Literal["surface", "extracted", "derived", "invented"]


class PatternRecordError(ValueError):
    """Raised when a valid general record violates a stage-specific policy."""


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpan(_ContractModel):
    """Half-open source interval supporting a mined pattern."""

    document_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("document_id cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_interval(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("source span must satisfy start < end")
        return self


class PatternProvenance(_ContractModel):
    """Auditable origin metadata required for each candidate pattern."""

    corpus_id: str
    parser_version: str
    ontology_versions: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    miner_version: str | None = None
    reasoner_version: str | None = None
    model_version: str | None = None
    experiment_version: str | None = None

    @field_validator("corpus_id", "parser_version")
    @classmethod
    def validate_required_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("provenance identifiers cannot be empty")
        return value

    @field_validator(
        "miner_version",
        "reasoner_version",
        "model_version",
        "experiment_version",
    )
    @classmethod
    def validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("optional provenance identifiers cannot be empty")
        return value

    @field_validator("ontology_versions", "source_document_ids")
    @classmethod
    def validate_identifier_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if not normalized or any(not item for item in normalized):
            raise ValueError("provenance identifier lists cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("provenance identifier lists cannot contain duplicates")
        return normalized


def _validate_json_value(value: Any, context: str) -> Any:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must contain finite JSON data") from error
    return value


class PatternRecord(_ContractModel):
    """Storage-neutral canonical record for an extracted or mined pattern."""

    id: str
    canonical_term: str
    interface_variables: tuple[str, ...]
    existential_variables: tuple[str, ...]
    type_constraints: dict[str, str]
    semantic_level: SemanticLevel
    ontology_groundings: tuple[str, ...]
    semantic_consequences: tuple[str, ...] | None = None
    source_spans: tuple[SourceSpan, ...]
    parser_confidences: tuple[float, ...]
    derivation_or_definition: str | None = None
    provenance: PatternProvenance
    generator_history: tuple[str, ...]
    cheap_features: dict[str, Any]
    cheap_endpoint: Any | None = None
    expensive_endpoint: Any | None = None
    expensive_trace: Any | None = None
    controller_prediction: float | None = None
    controller_uncertainty: float | None = None
    causal_statistics: dict[str, float] | None = None
    runtime_cost: float | None = None
    status: str

    @field_validator("id", "canonical_term", "status")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required pattern text fields cannot be empty")
        return value

    @field_validator("derivation_or_definition")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("derivation_or_definition cannot be empty")
        return value

    @field_validator(
        "interface_variables",
        "existential_variables",
        "ontology_groundings",
        "semantic_consequences",
        "generator_history",
    )
    @classmethod
    def validate_unique_text_tuple(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("pattern identifier lists cannot contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("pattern identifier lists cannot contain duplicates")
        return normalized

    @field_validator("type_constraints")
    @classmethod
    def validate_type_constraints(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for variable, type_name in value.items():
            variable = variable.strip()
            type_name = type_name.strip()
            if not variable or not type_name:
                raise ValueError(
                    "type constraints require non-empty variables and types"
                )
            normalized[variable] = type_name
        return normalized

    @field_validator("parser_confidences")
    @classmethod
    def validate_parser_confidences(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
            for confidence in value
        ):
            raise ValueError("parser confidences must be finite values in [0, 1]")
        return value

    @field_validator("controller_prediction")
    @classmethod
    def validate_controller_prediction(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("controller_prediction must be finite")
        return value

    @field_validator("controller_uncertainty", "runtime_cost")
    @classmethod
    def validate_nonnegative_metric(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(
                "uncertainty and runtime cost must be finite and non-negative"
            )
        return value

    @field_validator("causal_statistics")
    @classmethod
    def validate_causal_statistics(
        cls, value: dict[str, float] | None
    ) -> dict[str, float] | None:
        if value is None:
            return None
        if any(not key.strip() for key in value):
            raise ValueError("causal statistic names cannot be empty")
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError("causal statistics must be finite")
        return value

    @field_validator(
        "cheap_features",
        "cheap_endpoint",
        "expensive_endpoint",
        "expensive_trace",
    )
    @classmethod
    def validate_json_payload(cls, value: Any, info: Any) -> Any:
        return _validate_json_value(value, info.field_name)

    @model_validator(mode="after")
    def validate_cross_field_invariants(self) -> PatternRecord:
        interface = set(self.interface_variables)
        existential = set(self.existential_variables)
        overlap = interface & existential
        if overlap:
            raise ValueError(
                f"interface and existential variables overlap: {sorted(overlap)}"
            )

        variables = interface | existential
        unconstrained_keys = set(self.type_constraints) - variables
        if unconstrained_keys:
            raise ValueError(
                "type constraints reference undeclared variables: "
                f"{sorted(unconstrained_keys)}"
            )

        span_documents = {span.document_id for span in self.source_spans}
        provenance_documents = set(self.provenance.source_document_ids)
        unknown_documents = span_documents - provenance_documents
        if unknown_documents:
            raise ValueError(
                "source spans reference documents absent from provenance: "
                f"{sorted(unknown_documents)}"
            )

        if self.semantic_level == "extracted":
            if not self.source_spans:
                raise ValueError("extracted patterns require source spans")
            if not self.parser_confidences:
                raise ValueError("extracted patterns require parser confidences")
            if not self.ontology_groundings:
                raise ValueError("extracted patterns require ontology groundings")
        return self

    def to_canonical_json(self) -> str:
        """Serialize deterministically for caches, logs, and reproducibility checks."""

        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> PatternRecord:
        """Deserialize a record produced by :meth:`to_canonical_json`."""

        return cls.model_validate_json(payload)


def validate_q1_pattern_record(record: PatternRecord) -> PatternRecord:
    """Enforce extraction-first Q1 scope without restricting the future schema."""

    violations: list[str] = []
    if record.semantic_level not in {"surface", "extracted"}:
        violations.append("semantic_level must be surface or extracted")
    if record.semantic_consequences:
        violations.append("semantic_consequences require a later semantic stage")
    if record.derivation_or_definition is not None:
        violations.append("derivation_or_definition requires a later semantic stage")
    if not record.generator_history:
        violations.append("generator_history is required for Q1 mined records")
    if record.provenance.miner_version is None:
        violations.append("miner_version is required for Q1 mined records")
    if not record.cheap_features:
        violations.append("cheap_features are required for Q1 mined records")
    if record.controller_prediction is not None:
        violations.append("controller_prediction is outside Q1 scope")
    if record.controller_uncertainty is not None:
        violations.append("controller_uncertainty is outside Q1 scope")
    if record.causal_statistics is not None:
        violations.append("causal_statistics are outside Q1 scope")
    if violations:
        raise PatternRecordError(
            "Q1 PatternRecord violations: " + "; ".join(violations)
        )
    return record
