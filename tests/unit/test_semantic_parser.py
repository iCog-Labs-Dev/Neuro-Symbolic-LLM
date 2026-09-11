"""Unit tests for structured semantic-parser orchestration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

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


@pytest.fixture
def student(monkeypatch, tmp_path):
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    encoding = {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.ones(1, 2)}
    tokenizer.return_value.to.return_value = encoding
    tokenizer.decode.return_value = structured_output(assertion())
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = torch.tensor([[2, 3, 4]])
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    parser = DistilledSemanticParser.from_pretrained(
        str(tmp_path), max_new_tokens=42, num_beams=2
    )
    return parser, tokenizer, model


class TestDistilledSemanticParser:
    def test_uses_shared_pipeline_and_student_prompt(self, student):
        from parser.semantic.student_prompt import build_student_prompt

        parser, tokenizer, model = student
        tokenizer.decode.return_value = (
            "```json\n" + structured_output(assertion()) + "\n```"
        )
        assert str(parser.parse(" A dog has fur. ")[0]) == "(Has dog fur)"
        tokenizer.assert_called_once_with(
            build_student_prompt("A dog has fur."),
            return_tensors="pt",
            add_special_tokens=False,
        )
        assert model.generate.call_args.kwargs["max_new_tokens"] == 42
        assert model.generate.call_args.kwargs["num_beams"] == 2

    def test_reports_distilled_role_on_failure(self, student):
        parser, _, model = student
        model.generate.side_effect = RuntimeError("local inference failed")
        with pytest.raises(ModelGenerationError, match="distilled parser"):
            parser.parse("A dog has fur.")

    def test_rejects_empty_input_before_generation(self, student):
        parser, _, model = student
        with pytest.raises(ValueError, match="empty"):
            parser.parse("  ")
        model.generate.assert_not_called()

    def test_applies_aliases(self, student):
        parser, _, _ = student
        atoms = parser.parse("A dog has fur.", aliases={"dog": "Rex"})
        assert str(atoms[0]) == "(Has Rex fur)"


@pytest.mark.parametrize(
    "device, capabilities, expected",
    [
        ("auto", [], torch.float32),
        ("auto", [True], "auto"),
        ("auto", [True, False], torch.float32),
        ("cpu", [True], torch.float32),
        ("mps", [True], torch.float32),
        ("cuda", [True], "auto"),
        ("cuda:1", [True, False], torch.float32),
        ("cuda:1", [False, True], "auto"),
    ],
)
def test_distilled_precision_respects_requested_device(
    monkeypatch, tmp_path, device, capabilities, expected
):
    from contextlib import contextmanager

    current = [0]

    @contextmanager
    def cuda_device(index):
        previous = current[0]
        current[0] = previous if index is None else index
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(capabilities))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(torch.cuda, "device", cuda_device)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda: capabilities[current[0]]
    )
    tokenizer = MagicMock()
    loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoTokenizer.from_pretrained",
        lambda _: tokenizer,
    )
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoModelForCausalLM.from_pretrained", loader
    )
    DistilledSemanticParser.from_pretrained(str(tmp_path), device=device)
    assert loader.call_args.kwargs["torch_dtype"] == expected
    assert loader.call_args.kwargs["device_map"] == device
