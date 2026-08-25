"""Tests for deterministic structured semantic normalization."""

from __future__ import annotations

import pytest

from parser.semantic.normalization import (
    SemanticNormalizationError,
    normalize_semantic_result,
    normalize_symbol,
)
from parser.semantic.schema import SemanticParseResult


def make_result(
    *,
    value: str = "  New   York  ",
    role: str = " Location ",
    argument_type: str | None = " City ",
    relation: str = "  Located In ",
) -> SemanticParseResult:
    """Build unnormalized structured output."""
    return SemanticParseResult.model_validate(
        {
            "assertions": [
                {
                    "predicate": "Evaluation",
                    "relation": relation,
                    "arguments": [
                        {"value": value, "role": role, "type": argument_type}
                    ],
                    "fallback": True,
                    "polarity": "positive",
                    "confidence": 0.9,
                    "source_span": "  New York is a city.  ",
                    "alternatives": ["  another meaning  "],
                }
            ]
        }
    )


def test_normalizes_whitespace_and_preserves_entity_case() -> None:
    normalized = normalize_semantic_result(make_result())
    assertion = normalized.assertions[0]

    assert assertion.relation == "located_in"
    assert assertion.arguments[0].value == "New_York"
    assert assertion.arguments[0].role == "location"
    assert assertion.arguments[0].type == "City"
    assert assertion.source_span == "New York is a city."
    assert assertion.alternatives == ["another meaning"]


def test_does_not_mutate_the_model_output() -> None:
    original = make_result()

    normalize_semantic_result(original)

    assert original.assertions[0].arguments[0].value == "  New   York  "


def test_applies_only_explicit_aliases() -> None:
    normalized = normalize_semantic_result(
        make_result(value="Apple Computer", argument_type="Organization"),
        aliases={"Apple Computer": "Apple"},
        alias_types={"Apple": "Organization"},
    )

    assert normalized.assertions[0].arguments[0].value == "Apple"


def test_rejects_alias_type_conflicts() -> None:
    with pytest.raises(SemanticNormalizationError, match="Alias/type conflict"):
        normalize_semantic_result(
            make_result(value="Apple Inc.", argument_type="Fruit"),
            aliases={"Apple Inc.": "Apple"},
            alias_types={"Apple": "Organization"},
        )


def test_normalizes_compatible_unicode() -> None:
    assert normalize_symbol("Ａｐｐｌｅ Phone") == "Apple_Phone"


def test_rejects_whitespace_only_symbols() -> None:
    with pytest.raises(SemanticNormalizationError, match="cannot be empty"):
        normalize_symbol(" \t ")
