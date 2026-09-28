"""Deterministic normalization for structured semantic output."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from parser.semantic.schema import SemanticParseResult

_WHITESPACE_RE = re.compile(r"\s+")

_TEMPORAL_ROLES = {
    "time",
    "start_time",
    "end_time",
    "time_or_event",
}


class SemanticNormalizationError(ValueError):
    """Raised when semantic values cannot be normalized safely."""


def _normalize_phrase(value: str, field: str) -> str:
    """Normalize Unicode and whitespace without changing letter case."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        raise SemanticNormalizationError(f"{field} cannot be empty")

    return normalized


def normalize_symbol(value: str, field: str = "symbol") -> str:
    """Convert a phrase into one MeTTa symbol while preserving case."""
    return _normalize_phrase(value, field).replace(" ", "_")


def normalize_argument_value(value: str, role: str) -> str:
    """Normalize an argument into a safe MeTTa symbol.

    Temporal argument values that begin with a digit are prefixed with
    ``time_`` so that values such as ``2025`` or ``10 AM`` remain valid
    MeTTa symbols.

    Non-temporal arguments are left unchanged apart from normal symbol
    normalization.
    """
    normalized = normalize_symbol(value, "argument value")

    if role.casefold() in _TEMPORAL_ROLES and normalized[0].isdigit():
        return f"time_{normalized}"

    return normalized


def normalize_semantic_result(
    result: SemanticParseResult,
    *,
    aliases: Mapping[str, str] | None = None,
    alias_types: Mapping[str, str] | None = None,
) -> SemanticParseResult:
    """Return a normalized deep copy of structured semantic output.

    Aliases are explicit rather than guessed. Keys are observed entity names and
    values are their canonical names. Entity case is preserved.
    """
    normalized_result = result.model_copy(deep=True)

    normalized_aliases = {
        _normalize_phrase(alias, "alias"): _normalize_phrase(canonical, "alias target")
        for alias, canonical in (aliases or {}).items()
    }
    normalized_types = {
        _normalize_phrase(entity, "typed entity"): _normalize_phrase(
            entity_type, "entity type"
        )
        for entity, entity_type in (alias_types or {}).items()
    }

    for assertion in normalized_result.assertions:
        if assertion.relation is not None:
            assertion.relation = normalize_symbol(
                assertion.relation.casefold(), "relation"
            )

        for argument in assertion.arguments:
            entity = _normalize_phrase(argument.value, "argument value")
            entity = normalized_aliases.get(entity, entity)
            expected_type = normalized_types.get(entity)

            if (
                expected_type is not None
                and argument.type is not None
                and _normalize_phrase(argument.type, "argument type").casefold()
                != expected_type.casefold()
            ):
                raise SemanticNormalizationError(
                    f"Alias/type conflict for {entity!r}: expected "
                    f"{expected_type!r}, got {argument.type!r}"
                )

            argument.role = _normalize_phrase(
                argument.role,
                "argument role",
            ).casefold()

            argument.value = normalize_argument_value(
                entity,
                argument.role,
            )

            if argument.type is not None:
                argument.type = _normalize_phrase(
                    argument.type,
                    "argument type",
                )

        assertion.source_span = _normalize_phrase(
            assertion.source_span,
            "source span",
        )

        assertion.alternatives = [
            _normalize_phrase(
                alternative,
                "alternative",
            )
            for alternative in assertion.alternatives
        ]

    return normalized_result
