"""Deterministic normalization for structured semantic output."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from parser.semantic.schema import SemanticParseResult

_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_PREFIX_RE = re.compile(r"^[0-9]")

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
    """Convert a phrase into one MeTTa-safe symbol.

    This performs only deterministic representation cleanup:
    - whitespace -> underscore
    - commas in numeric values are removed
    - decimal points are converted to underscores
    - percent signs are expanded to 'percent'
    """
    normalized = _normalize_phrase(value, field)

    normalized = normalized.replace(",", "")
    normalized = normalized.replace("%", "_percent")
    normalized = normalized.replace(".", "_")
    normalized = normalized.replace(" ", "_")

    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        raise SemanticNormalizationError(f"{field} cannot be empty")

    return normalized


def normalize_argument_value(value: str, role: str) -> str:
    """Normalize an argument into a safe MeTTa symbol.

    Temporal numeric values receive a ``time_`` prefix.
    Other values beginning with a digit receive a ``num_`` prefix.

    Examples:
        2025, role=time       -> time_2025
        10 AM, role=time      -> time_10_AM
        25 liters             -> num_25_liters
        1,000 employees       -> num_1000_employees
        37.5 degrees          -> num_37_5_degrees
        25%                   -> num_25_percent
    """
    normalized = normalize_symbol(value, "argument value")
    role_normalized = role.casefold()

    if normalized.startswith("time_") or normalized.startswith("num_"):
        return normalized

    if _NUMERIC_PREFIX_RE.match(normalized):
        if role_normalized in _TEMPORAL_ROLES:
            return f"time_{normalized}"

        return f"num_{normalized}"

    return normalized


def normalize_semantic_result(
    result: SemanticParseResult,
    *,
    aliases: Mapping[str, str] | None = None,
    alias_types: Mapping[str, str] | None = None,
) -> SemanticParseResult:
    """Return a normalized deep copy of structured semantic output.

    Aliases are explicit rather than guessed. Keys are observed entity names
    and values are their canonical names. Entity case is preserved.
    """
    normalized_result = result.model_copy(deep=True)

    normalized_aliases = {
        _normalize_phrase(alias, "alias"): _normalize_phrase(
            canonical,
            "alias target",
        )
        for alias, canonical in (aliases or {}).items()
    }

    normalized_types = {
        _normalize_phrase(entity, "typed entity"): _normalize_phrase(
            entity_type,
            "entity type",
        )
        for entity, entity_type in (alias_types or {}).items()
    }

    for assertion in normalized_result.assertions:
        if assertion.relation is not None:
            assertion.relation = normalize_symbol(
                assertion.relation.casefold(),
                "relation",
            )

        for argument in assertion.arguments:
            entity = _normalize_phrase(
                argument.value,
                "argument value",
            )

            entity = normalized_aliases.get(entity, entity)

            expected_type = normalized_types.get(entity)

            if (
                expected_type is not None
                and argument.type is not None
                and _normalize_phrase(
                    argument.type,
                    "argument type",
                ).casefold()
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
