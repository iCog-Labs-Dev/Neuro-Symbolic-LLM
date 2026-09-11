"""Deterministic normalization for structured semantic output."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from parser.semantic.schema import SemanticParseResult

_WHITESPACE_RE = re.compile(r"\s+")


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
            argument.value = normalize_symbol(entity, "argument value")
            argument.role = _normalize_phrase(argument.role, "argument role").casefold()
            if argument.type is not None:
                argument.type = _normalize_phrase(argument.type, "argument type")

        assertion.source_span = _normalize_phrase(assertion.source_span, "source span")
        assertion.alternatives = [
            _normalize_phrase(alternative, "alternative")
            for alternative in assertion.alternatives
        ]

    return normalized_result
