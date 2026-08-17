"""tests/unit/test_semantic_parser.py

Run with: python -m pytest tests/unit/test_semantic_parser.py -v

These tests define the behavior shared by reference and distilled parsers.
They never call an external API.
"""

from __future__ import annotations

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

# ── Test backend ───────────────────────────────────────────────────────────────


class FakeBackend:
    """Return configured output and record generation calls."""

    provider_name = "fake"

    def __init__(
        self,
        output: str = "(Has dog fur)",
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        if self.error is not None:
            raise self.error
        return self.output


def make_teacher(output: str = "(Has dog fur)"):
    backend = FakeBackend(output)
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
    )
    return parser, backend


# ── Configuration ──────────────────────────────────────────────────────────────


class TestSemanticParserConfig:
    def test_rejects_empty_model_name(self):
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            SemanticParserConfig(model_name="   ")


# ── Prompt construction ────────────────────────────────────────────────────────


class TestPrompt:
    def test_contains_sentence_and_predicates(self):
        prompt = ReferenceSemanticParser.build_prompt("A dog has fur.")
        assert "<sentence>\nA dog has fur.\n</sentence>" in prompt
        assert ALLOWED_PREDICATES in prompt

    def test_contains_output_contract(self):
        prompt = ReferenceSemanticParser.build_prompt("A dog has fur.")
        assert "output exactly:\n\nUNSUPPORTED" in prompt
        assert "Do not output explanations, labels, comments, Markdown" in prompt


# ── Shared parsing behavior ────────────────────────────────────────────────────


class TestReferenceSemanticParser:
    def test_returns_canonical_link_atom(self):
        parser, _ = make_teacher("(Has Dog Fur)")
        result = parser.parse("A dog has fur.")
        assert len(result) == 1
        assert isinstance(result[0], LinkAtom)
        assert str(result[0]) == "(Has dog fur)"

    def test_preserves_passive_semantic_roles(self):
        parser, _ = make_teacher("(Evaluation Buy (List Ben Car))")
        result = parser.parse("A car was bought by Ben.")
        assert str(result[0]) == "(Evaluation buy (List ben car))"

    def test_removes_code_fence(self):
        parser, _ = make_teacher("```metta\n(Has dog fur)\n```")
        assert str(parser.parse("A dog has fur.")[0]) == "(Has dog fur)"

    def test_returns_adjacent_top_level_atoms_separately(self):
        parser, _ = make_teacher("(StateOf cat sleeping) (On cat chair)")

        atoms = parser.parse("The cat is sleeping on the chair.")

        assert [str(atom) for atom in atoms] == [
            "(StateOf cat sleeping)",
            "(On cat chair)",
        ]

    def test_rejects_unsupported(self):
        parser, _ = make_teacher("UNSUPPORTED")
        with pytest.raises(SemanticParseError, match="Unsupported sentence"):
            parser.parse("Something ambiguous.")

    def test_rejects_invalid_metta(self):
        parser, _ = make_teacher("(Unknown dog fur)")
        with pytest.raises(SemanticParseError, match="invalid MeTTa"):
            parser.parse("A dog has fur.")

    def test_wraps_backend_failure(self):
        backend = FakeBackend(error=RuntimeError("network failed"))
        parser = ReferenceSemanticParser(
            backend=backend,
            config=SemanticParserConfig(model_name="teacher-model"),
        )
        with pytest.raises(ModelGenerationError, match="reference parser"):
            parser.parse("A dog has fur.")


class TestDistilledSemanticParser:
    def test_uses_same_validation_pipeline(self):
        backend = FakeBackend("(Cause Rain Flood)")
        parser = DistilledSemanticParser(
            backend=backend,
            config=SemanticParserConfig(model_name="student-model"),
        )
        assert str(parser.parse("Rain causes flooding.")[0]) == ("(Cause rain flood)")

    def test_reports_distilled_role_on_failure(self):
        backend = FakeBackend(error=RuntimeError("local inference failed"))
        parser = DistilledSemanticParser(
            backend=backend,
            config=SemanticParserConfig(model_name="student-model"),
        )
        with pytest.raises(ModelGenerationError, match="distilled parser"):
            parser.parse("A dog has fur.")
