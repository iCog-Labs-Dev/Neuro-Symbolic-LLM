"""Regression tests for semantic factuality classification."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from parser.semantic import (
    CallableBackend,
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from parser.semantic.schema import SemanticParseResult


def assertion(
    *,
    predicate: str = "Evaluation",
    relation: str | None = "work",
    values: tuple[str, ...] = ("drug",),
    roles: tuple[str, ...] = ("agent",),
    fallback: bool = True,
    polarity: str = "positive",
    factuality: str = "asserted",
    confidence: float = 0.99,
    source_span: str = "supporting text",
) -> dict[str, Any]:
    """Build one structured assertion."""
    return {
        "predicate": predicate,
        "relation": relation,
        "arguments": [
            {
                "value": value,
                "role": role,
                "type": None,
            }
            for value, role in zip(
                values,
                roles,
                strict=True,
            )
        ],
        "fallback": fallback,
        "polarity": polarity,
        "factuality": factuality,
        "confidence": confidence,
        "source_span": source_span,
        "alternatives": [],
    }


def structured_output(
    *assertions: dict[str, Any],
) -> str:
    """Serialize fake teacher output."""
    return json.dumps(
        {
            "assertions": list(assertions),
        }
    )


def make_teacher(
    output: str,
) -> ReferenceSemanticParser:
    """Build a deterministic fake teacher parser."""

    def generate(
        *,
        prompt: str,
        model: str,
    ) -> str:
        del prompt
        del model
        return output

    return ReferenceSemanticParser(
        backend=CallableBackend(
            generate,
            provider_name="fake-teacher",
        ),
        config=SemanticParserConfig(
            model_name="teacher-model",
            prompt_version="2.0.0",
        ),
    )


def test_factuality_defaults_to_asserted() -> None:
    """Old structured outputs remain compatible."""
    data = {
        "assertions": [
            {
                "predicate": "Evaluation",
                "relation": "announce",
                "arguments": [
                    {
                        "value": "company",
                        "role": "agent",
                        "type": None,
                    },
                    {
                        "value": "product",
                        "role": "patient",
                        "type": None,
                    },
                ],
                "fallback": True,
                "polarity": "positive",
                "confidence": 0.99,
                "source_span": "The company announced a product.",
                "alternatives": [],
            }
        ]
    }

    result = SemanticParseResult.model_validate(data)

    assert result.assertions[0].factuality == "asserted"


@pytest.mark.parametrize(
    "factuality",
    [
        "asserted",
        "opinion",
        "speculative",
        "hypothetical",
    ],
)
def test_accepts_supported_factuality_values(
    factuality: str,
) -> None:
    result = SemanticParseResult.model_validate(
        {
            "assertions": [
                assertion(
                    factuality=factuality,
                )
            ]
        }
    )

    assert result.assertions[0].factuality == factuality


@pytest.mark.parametrize(
    "factuality",
    [
        "fact",
        "uncertain",
        "possible",
        "conditional",
        "subjective",
        "",
    ],
)
def test_rejects_unsupported_factuality_values(
    factuality: str,
) -> None:
    with pytest.raises(ValidationError):
        SemanticParseResult.model_validate(
            {
                "assertions": [
                    assertion(
                        factuality=factuality,
                    )
                ]
            }
        )


@pytest.mark.parametrize(
    "factuality",
    [
        "asserted",
        "opinion",
        "speculative",
        "hypothetical",
    ],
)
def test_reference_parser_preserves_factuality(
    factuality: str,
) -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                factuality=factuality,
            )
        )
    )

    result = parser.generate_structured("Supporting sentence.")

    assert result.assertions[0].factuality == factuality


def test_opinion_remains_opinion() -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                predicate="Evaluation",
                relation="think",
                values=("I", "Paris_is_beautiful"),
                roles=("agent", "patient"),
                fallback=True,
                factuality="opinion",
                source_span="I think Paris is beautiful.",
            )
        )
    )

    result = parser.generate_structured("I think Paris is beautiful.")

    assertion_result = result.assertions[0]

    assert assertion_result.relation == "think"
    assert assertion_result.factuality == "opinion"


def test_speculative_statement_remains_speculative() -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                predicate="Evaluation",
                relation="work",
                values=("drug",),
                roles=("agent",),
                fallback=True,
                factuality="speculative",
                source_span="The drug probably works.",
            )
        )
    )

    result = parser.generate_structured("The drug probably works.")

    assert result.assertions[0].factuality == "speculative"


def test_speculative_temporal_modifier_inherits_factuality() -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                predicate="Evaluation",
                relation="become",
                values=("Paris", "larger"),
                roles=("agent", "patient"),
                fallback=True,
                factuality="speculative",
                source_span="Paris might become larger",
            ),
            assertion(
                predicate="OccursAt",
                relation=None,
                values=("become", "next_year"),
                roles=("event_or_state", "time"),
                fallback=False,
                factuality="speculative",
                source_span="next year",
            ),
        )
    )

    result = parser.generate_structured("Paris might become larger next year.")

    assert [item.factuality for item in result.assertions] == [
        "speculative",
        "speculative",
    ]


def test_hypothetical_assertions_remain_hypothetical() -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                predicate="Evaluation",
                relation="rain",
                values=("it",),
                roles=("agent",),
                fallback=True,
                factuality="hypothetical",
                source_span="If it rains",
            ),
            assertion(
                predicate="Evaluation",
                relation="become",
                values=("road", "slippery"),
                roles=("patient", "state"),
                fallback=True,
                factuality="hypothetical",
                source_span="the road may become slippery",
            ),
        )
    )

    result = parser.generate_structured("If it rains, the road may become slippery.")

    assert len(result.assertions) == 2

    assert all(item.factuality == "hypothetical" for item in result.assertions)


def test_factuality_is_independent_of_confidence() -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                predicate="Evaluation",
                relation="work",
                values=("drug",),
                roles=("agent",),
                fallback=True,
                factuality="speculative",
                confidence=0.99,
                source_span="The drug probably works.",
            )
        )
    )

    result = parser.generate_structured("The drug probably works.")

    assertion_result = result.assertions[0]

    assert assertion_result.factuality == "speculative"
    assert assertion_result.confidence == 0.99


def test_factuality_is_independent_of_polarity() -> None:
    parser = make_teacher(
        structured_output(
            assertion(
                predicate="Evaluation",
                relation="work",
                values=("drug",),
                roles=("agent",),
                fallback=True,
                polarity="negative",
                factuality="speculative",
                source_span="The drug might not work.",
            )
        )
    )

    result = parser.generate_structured("The drug might not work.")

    assertion_result = result.assertions[0]

    assert assertion_result.polarity == "negative"
    assert assertion_result.factuality == "speculative"
