"""Tests for deterministic structured-semantics-to-MeTTa rendering."""

from __future__ import annotations

import pytest

from parser.semantic.metta_renderer import (
    MettaRenderError,
    render_metta,
    validate_rendered_metta,
)
from parser.semantic.schema import SemanticParseResult


def make_result(
    *,
    predicate: str,
    values: tuple[str, ...],
    roles: tuple[str, ...],
    relation: str | None = None,
    fallback: bool = False,
    polarity: str = "positive",
) -> SemanticParseResult:
    """Build structured output for renderer tests."""
    return SemanticParseResult.model_validate(
        {
            "assertions": [
                {
                    "predicate": predicate,
                    "relation": relation,
                    "arguments": [
                        {"value": value, "role": role}
                        for value, role in zip(values, roles, strict=True)
                    ],
                    "fallback": fallback,
                    "polarity": polarity,
                    "confidence": 0.99,
                    "source_span": "supporting text",
                    "alternatives": [],
                }
            ]
        }
    )


def test_renders_known_predicate() -> None:
    result = make_result(
        predicate="Has",
        values=("dog", "fur"),
        roles=("owner", "possessed"),
    )

    assert render_metta(result) == ["(Has dog fur)"]


def test_renders_evaluation_with_argument_list() -> None:
    result = make_result(
        predicate="Evaluation",
        relation="buy",
        values=("Ben", "car"),
        roles=("agent", "patient"),
        fallback=True,
    )

    assert render_metta(result) == ["(Evaluation buy (List Ben car))"]


def test_renders_negative_assertion() -> None:
    result = make_result(
        predicate="On",
        values=("cat", "chair"),
        roles=("entity", "surface"),
        polarity="negative",
    )

    assert render_metta(result) == ["(Not (On cat chair))"]


def test_preserves_symbol_case() -> None:
    result = make_result(
        predicate="LocatedIn",
        values=("Apple", "California"),
        roles=("entity", "location"),
    )

    assert render_metta(result) == ["(LocatedIn Apple California)"]


def test_renders_multiple_assertions_in_order() -> None:
    first = make_result(
        predicate="Has",
        values=("dog", "fur"),
        roles=("owner", "possessed"),
    )
    second = make_result(
        predicate="CanDo",
        values=("dog", "bark"),
        roles=("agent", "action"),
    )
    result = SemanticParseResult(assertions=[first.assertions[0], second.assertions[0]])

    assert render_metta(result) == ["(Has dog fur)", "(CanDo dog bark)"]


@pytest.mark.parametrize("unsafe_value", ["New York", "car)", "(car"])
def test_rejects_unsafe_argument_symbols(unsafe_value: str) -> None:
    result = make_result(
        predicate="Has",
        values=("company", unsafe_value),
        roles=("owner", "possessed"),
    )

    with pytest.raises(MettaRenderError, match="argument value"):
        render_metta(result)


def test_rejects_unsafe_evaluation_relation() -> None:
    result = make_result(
        predicate="Evaluation",
        relation="buy car",
        values=("Ben", "car"),
        roles=("agent", "patient"),
        fallback=True,
    )

    with pytest.raises(MettaRenderError, match="relation"):
        render_metta(result)


def test_validates_and_parses_rendered_expressions() -> None:
    atoms = validate_rendered_metta(["(Has dog fur)", "(Not (On cat chair))"])

    assert [str(atom) for atom in atoms] == [
        "(Has dog fur)",
        "(Not (On cat chair))",
    ]


def test_rejects_invalid_rendered_expression() -> None:
    with pytest.raises(MettaRenderError, match="Rendered invalid MeTTa"):
        validate_rendered_metta(["(Unknown dog fur)"])
