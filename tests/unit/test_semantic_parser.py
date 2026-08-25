"""Unit tests for structured semantic-parser orchestration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from parser.grammar.atomese import LinkAtom
from parser.semantic import (
    ALLOWED_PREDICATES,
    DistilledSemanticParser,
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)


def assertion(
    *,
    predicate: str = "Has",
    relation: str | None = None,
    values: tuple[str, ...] = ("dog", "fur"),
    roles: tuple[str, ...] = ("owner", "possessed"),
    fallback: bool = False,
    polarity: str = "positive",
    confidence: float = 0.99,
) -> dict[str, Any]:
    """Return one structured assertion dictionary."""
    return {
        "predicate": predicate,
        "relation": relation,
        "arguments": [
            {"value": value, "role": role, "type": None}
            for value, role in zip(values, roles, strict=True)
        ],
        "fallback": fallback,
        "polarity": polarity,
        "confidence": confidence,
        "source_span": "supporting text",
        "alternatives": [],
    }


def structured_output(*assertions: dict[str, Any]) -> str:
    """Serialize assertions as the model's JSON response."""
    return json.dumps({"assertions": list(assertions)})


class FakeBackend:
    """Return configured output and record model generation calls."""

    provider_name = "fake"

    def __init__(
        self,
        output: str | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = structured_output(assertion()) if output is None else output
        self.error = error
        self.calls: list[dict[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        if self.error is not None:
            raise self.error
        return self.output


def make_teacher(
    output: str | None = None,
) -> tuple[ReferenceSemanticParser, FakeBackend]:
    backend = FakeBackend(output)
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
    )
    return parser, backend


class TestSemanticParserConfig:
    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            SemanticParserConfig(model_name="   ")

    def test_rejects_empty_prompt_version(self) -> None:
        with pytest.raises(ValueError, match="prompt_version cannot be empty"):
            SemanticParserConfig(model_name="model", prompt_version=" ")


class TestPrompt:
    def test_contains_sentence_context_and_predicates(self) -> None:
        prompt = ReferenceSemanticParser.build_prompt(
            "It has stores.",
            context="Apple is a company.",
        )

        assert "<sentence>\nIt has stores.\n</sentence>" in prompt
        assert "<context>\nApple is a company.\n</context>" in prompt
        assert ALLOWED_PREDICATES in prompt
        assert "On(entity, surface)" in prompt
        assert "Has(owner, possessed)" in prompt

    def test_requires_json_instead_of_metta(self) -> None:
        prompt = ReferenceSemanticParser.build_prompt("A dog has fur.")

        assert "Return valid JSON only" in prompt
        assert "Do not output MeTTa" in prompt


class TestReferenceSemanticParser:
    def test_generates_normalized_structured_semantics(self) -> None:
        output = structured_output(
            assertion(
                values=(" Apple Computer ", "retail stores"),
                roles=(" OWNER ", " possessed "),
            )
        )
        parser, backend = make_teacher(output)

        result = parser.generate_structured(
            "Apple Computer has retail stores.",
            aliases={"Apple Computer": "Apple"},
        )

        assert result.assertions[0].arguments[0].value == "Apple"
        assert result.assertions[0].arguments[1].value == "retail_stores"
        assert backend.calls[0]["model"] == "teacher-model"

    def test_returns_validated_link_atom(self) -> None:
        parser, _ = make_teacher()

        result = parser.parse("A dog has fur.")

        assert len(result) == 1
        assert isinstance(result[0], LinkAtom)
        assert str(result[0]) == "(Has dog fur)"

    def test_returns_multiple_assertions_separately(self) -> None:
        output = structured_output(
            assertion(
                predicate="StateOf",
                values=("cat", "sleeping"),
                roles=("entity", "state"),
            ),
            assertion(
                predicate="On",
                values=("cat", "chair"),
                roles=("entity", "surface"),
            ),
        )
        parser, _ = make_teacher(output)

        atoms = parser.parse("The cat is sleeping on the chair.")

        assert [str(atom) for atom in atoms] == [
            "(StateOf cat sleeping)",
            "(On cat chair)",
        ]

    def test_renders_evaluation_and_negation(self) -> None:
        output = structured_output(
            assertion(
                predicate="Evaluation",
                relation="acquire",
                values=("Apple", "Tesla"),
                roles=("agent", "patient"),
                fallback=True,
                polarity="negative",
            )
        )
        parser, _ = make_teacher(output)

        atom = parser.parse("Apple did not acquire Tesla.")[0]

        assert str(atom) == "(Not (Evaluation acquire (List Apple Tesla)))"

    def test_accepts_one_optional_json_code_fence(self) -> None:
        output = f"```json\n{structured_output(assertion())}\n```"
        parser, _ = make_teacher(output)

        assert str(parser.parse("A dog has fur.")[0]) == "(Has dog fur)"

    @pytest.mark.parametrize(
        "output",
        [
            "not JSON",
            "(Has dog fur)",
            json.dumps({"assertions": []}),
            structured_output(assertion(confidence=1.5)),
        ],
    )
    def test_rejects_invalid_structured_output(self, output: str) -> None:
        parser, _ = make_teacher(output)

        with pytest.raises(SemanticParseError, match="invalid structured"):
            parser.parse("A dog has fur.")

    def test_rejects_unknown_predicate(self) -> None:
        parser, _ = make_teacher(
            structured_output(assertion(predicate="InventedPredicate"))
        )

        with pytest.raises(SemanticParseError, match="Unknown semantic predicate"):
            parser.parse("A dog has fur.")

    def test_rejects_wrong_argument_roles(self) -> None:
        parser, _ = make_teacher(
            structured_output(assertion(roles=("possessed", "owner")))
        )

        with pytest.raises(SemanticParseError, match="requires argument roles"):
            parser.parse("A dog has fur.")

    def test_rejects_empty_sentence_without_calling_backend(self) -> None:
        parser, backend = make_teacher()

        with pytest.raises(ValueError, match="sentence cannot be empty"):
            parser.parse("   ")

        assert backend.calls == []

    def test_rejects_empty_model_response(self) -> None:
        parser, _ = make_teacher("   ")

        with pytest.raises(ModelGenerationError, match="empty response"):
            parser.parse("A dog has fur.")

    def test_wraps_backend_failure(self) -> None:
        backend = FakeBackend(error=RuntimeError("network failed"))
        parser = ReferenceSemanticParser(
            backend=backend,
            config=SemanticParserConfig(model_name="teacher-model"),
        )

        with pytest.raises(ModelGenerationError, match="reference parser"):
            parser.parse("A dog has fur.")


class TestDistilledSemanticParser:
    def test_uses_the_same_structured_pipeline(self) -> None:
        backend = FakeBackend(
            structured_output(
                assertion(
                    predicate="Cause",
                    values=("Rain", "Flood"),
                    roles=("cause", "effect"),
                )
            )
        )
        parser = DistilledSemanticParser(
            backend=backend,
            config=SemanticParserConfig(model_name="student-model"),
        )

        assert str(parser.parse("Rain causes flooding.")[0]) == "(Cause Rain Flood)"

    def test_reports_distilled_role_on_failure(self) -> None:
        backend = FakeBackend(error=RuntimeError("local inference failed"))
        parser = DistilledSemanticParser(
            backend=backend,
            config=SemanticParserConfig(model_name="student-model"),
        )

        with pytest.raises(ModelGenerationError, match="distilled parser"):
            parser.parse("A dog has fur.")
