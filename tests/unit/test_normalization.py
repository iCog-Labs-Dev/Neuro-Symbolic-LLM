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


@pytest.mark.parametrize(
    "role,value,expected",
    [
        ("time", "2025", "time_2025"),
        ("time", "10 AM", "time_10_AM"),
        ("start_time", "2025", "time_2025"),
        ("end_time", "2026", "time_2026"),
        ("time_or_event", "2025", "time_2025"),
    ],
)
def test_prefixes_numeric_temporal_values(
    role: str,
    value: str,
    expected: str,
) -> None:
    """Numeric temporal values receive the time_ prefix."""
    normalized = normalize_semantic_result(
        make_result(
            value=value,
            role=role,
            argument_type=None,
            relation="occur",
        )
    )

    argument = normalized.assertions[0].arguments[0]

    assert argument.value == expected
    assert argument.role == role


def test_prefixes_non_temporal_numeric_values() -> None:
    """Numeric-leading non-temporal values receive the num_ prefix."""
    normalized = normalize_semantic_result(
        make_result(
            value="2025",
            role="patient",
            argument_type=None,
            relation="measure",
        )
    )

    assert normalized.assertions[0].arguments[0].value == "num_2025"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("25 liters", "num_25_liters"),
        ("1,000 employees", "num_1000_employees"),
        ("37.5 degrees", "num_37_5_degrees"),
        ("25%", "num_25_percent"),
        ("90 minutes", "num_90_minutes"),
        ("5 km", "num_5_km"),
        ("1M people", "num_1M_people"),
        ("10 kg", "num_10_kg"),
        ("30 days", "num_30_days"),
    ],
)
def test_normalizes_numeric_quantities(
    value: str,
    expected: str,
) -> None:
    """Quantities and measurements become MeTTa-safe symbols."""
    normalized = normalize_semantic_result(
        make_result(
            value=value,
            role="patient",
            argument_type=None,
            relation="measure",
        )
    )

    assert normalized.assertions[0].arguments[0].value == expected


def test_does_not_double_prefix_normalized_temporal_values() -> None:
    normalized = normalize_semantic_result(
        make_result(
            value="time_2025",
            role="time",
            argument_type=None,
            relation="occur",
        )
    )

    assert normalized.assertions[0].arguments[0].value == "time_2025"


def test_does_not_double_prefix_normalized_numeric_values() -> None:
    """Already-normalized numeric values are not prefixed twice."""
    normalized = normalize_semantic_result(
        make_result(
            value="num_25_liters",
            role="patient",
            argument_type=None,
            relation="measure",
        )
    )

    assert normalized.assertions[0].arguments[0].value == "num_25_liters"


def test_preserves_non_numeric_non_temporal_values() -> None:
    """Ordinary symbolic values do not receive numeric prefixes."""
    normalized = normalize_semantic_result(
        make_result(
            value="research_report",
            role="patient",
            argument_type=None,
            relation="read",
        )
    )

    assert normalized.assertions[0].arguments[0].value == "research_report"
