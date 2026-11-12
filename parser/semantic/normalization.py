"""Deterministic normalization for structured semantic output.

normalize_extraction_payload cleans the format of the model's raw extraction
before it is validated as a SemanticExtraction. It changes letter case,
spacing, and punctuation of names, never their meaning: quotations stay
exact, catalog names only lose surrounding whitespace, and synonyms are
merged only through explicit aliases.

restore_quote_case is separate: a quote that differs from the source only
by letter case ("students" for "Students") gets the source's own spelling
back, and every such change is returned so the parser can record it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


class SemanticNormalizationError(ValueError):
    """Raised when semantic values cannot be normalized safely."""


def restore_quote_case(
    payload: dict[str, Any], source_text: str
) -> tuple[dict[str, Any], list[str]]:
    """Give quotes that differ from the source only by letter case its spelling.

    Exact quotes are left alone. Returns the repaired copy and one note per
    changed quote; other mismatches are left for validation to reject.
    """
    lowered = source_text.lower()
    if len(lowered) != len(source_text):  # case mapping moved the offsets
        return payload, []
    repaired = deepcopy(payload)
    changes: list[str] = []

    def starts(haystack: str, needle: str) -> list[int]:
        found, start = [], haystack.find(needle)
        while start != -1:
            found.append(start)
            start = haystack.find(needle, start + 1)
        return found

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        text, occurrence = value.get("text"), value.get("occurrence", 0)
        if (
            isinstance(text, str)
            and text
            and set(value) <= {"text", "occurrence"}
            and isinstance(occurrence, int)
            and not isinstance(occurrence, bool)
            and occurrence >= 0
        ):
            if occurrence < len(starts(source_text, text)):
                return
            candidates = starts(lowered, text.lower())
            if len(text.lower()) != len(text) or occurrence >= len(candidates):
                return
            start = candidates[occurrence]
            exact = source_text[start : start + len(text)]
            value["text"] = exact
            value["occurrence"] = starts(source_text, exact).index(start)
            changes.append(f"quote {text!r} restored to source spelling {exact!r}")
            return
        for item in value.values():
            visit(item)

    visit(repaired)
    return repaired, changes


def normalize_extraction_payload(
    payload: dict[str, Any],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a normalized copy of the model's (LLM's) raw extraction.

    Entity IDs, kinds, relations, roles, and rule variables become lowercase
    snake_case names. Aliases map an observed kind or relation to its
    canonical name (for example ``vendor`` to ``supplier``); they are explicit
    rather than guessed. Values with an unexpected shape are left unchanged
    for schema validation to reject.
    """
    normalized = deepcopy(payload)
    canonical_names = {
        _normalize_name(alias, "alias"): _normalize_name(canonical, "alias target")
        for alias, canonical in (aliases or {}).items()
    }

    for entity in _dicts(normalized.get("entities")):
        _apply(entity, "id", lambda value: _normalize_name(value, "entity id"))
        _apply(
            entity,
            "kind",
            lambda value: _canonical_name(value, "kind", canonical_names),
        )
        _apply(entity, "entity_type", str.strip)

    for assertion in _dicts(normalized.get("assertions")):
        _normalize_assertion_payload(assertion, canonical_names)

    for rule in _dicts(normalized.get("rules")):
        _apply(rule, "scope", _normalize_enum)
        for variable in _dicts(rule.get("variables")):
            _apply(variable, "id", lambda value: _normalize_name(value, "variable"))
            _apply(variable, "entity_type", str.strip)
        for part in [
            *_dicts(rule.get("antecedents")),
            *_dicts(rule.get("consequents")),
        ]:
            _normalize_assertion_payload(part, canonical_names)

    return normalized


def _normalize_assertion_payload(
    assertion: dict[str, Any], canonical_names: Mapping[str, str]
) -> None:
    """Normalize one raw assertion in place; its evidence quotes are left exact."""
    _apply(assertion, "predicate", str.strip)
    _apply(
        assertion,
        "relation",
        lambda value: _canonical_name(value, "relation", canonical_names),
    )

    for argument in _dicts(assertion.get("arguments")):
        _apply(argument, "entity_id", lambda value: _normalize_name(value, "entity id"))
        _apply(argument, "role", lambda value: _normalize_name(value, "role"))

    for name in ("polarity", "factuality", "modality"):
        _apply(assertion, name, _normalize_enum)

    alternatives = assertion.get("alternatives")
    if isinstance(alternatives, list):
        assertion["alternatives"] = [
            _normalize_phrase(item, "alternative") if isinstance(item, str) else item
            for item in alternatives
        ]


def _normalize_name(value: str, field: str) -> str:
    """Convert a name into lowercase snake_case, keeping a leading ``$``.

    A name that would start with a digit gets a ``num_`` prefix, so
    ``30 days`` becomes ``num_30_days``.
    """
    name = normalize_symbol(value.casefold(), field)
    if name[0].isdigit():
        return f"num_{name}"
    return name


def _canonical_name(value: str, field: str, canonical_names: Mapping[str, str]) -> str:
    """Normalize a kind or relation, then replace it by its alias target."""
    name = _normalize_name(value, field)
    return canonical_names.get(name, name)


def _normalize_enum(value: str) -> str:
    """Normalize a fixed option such as a factuality; every option is lowercase."""
    return value.strip().casefold()


def _apply(record: dict[str, Any], key: str, normalize: Callable[[str], str]) -> None:
    """Normalize ``record[key]`` in place when it is a string."""
    value = record.get(key)
    if isinstance(value, str):
        record[key] = normalize(value)


def _dicts(value: Any) -> list[dict[str, Any]]:
    """Return the dictionaries of a list, or nothing for any other value."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_phrase(value: str, field: str) -> str:
    """Normalize Unicode and whitespace without changing letter case."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        raise SemanticNormalizationError(f"{field} cannot be empty")

    return normalized


def normalize_symbol(value: str, field: str = "symbol") -> str:
    """Convert a phrase into one MeTTa-safe symbol.

    This performs deterministic representation cleanup:
    - Unicode is normalized
    - whitespace becomes underscores
    - commas in numeric values are removed
    - decimal points become underscores
    - percent signs become ``percent``
    - apostrophes and other punctuation become separators
    - unsupported Unicode characters are transliterated where possible
    - repeated underscores are collapsed
    """
    normalized = _normalize_phrase(value, field)
    if (
        normalized.startswith("$")
        and len(normalized) > 1
        and (normalized[1].isalpha() or normalized[1] == "_")
    ):
        variable_name = normalized[1:]

        variable_name = unicodedata.normalize("NFKD", variable_name)
        variable_name = variable_name.encode("ascii", "ignore").decode("ascii")

        variable_name = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            variable_name,
        )

        variable_name = re.sub(
            r"_+",
            "_",
            variable_name,
        ).strip("_")

        if not variable_name:
            raise SemanticNormalizationError(f"{field} variable cannot be empty")

        return f"${variable_name}"
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("‘", "'")

    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")

    normalized = normalized.replace(",", "")
    normalized = normalized.replace("%", "_percent")
    normalized = normalized.replace(".", "_")
    normalized = normalized.replace("&", "_and_")

    normalized = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        normalized,
    )

    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        raise SemanticNormalizationError(f"{field} cannot be empty")

    return normalized
