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
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import DistilledSemanticParser


def assertion(
    *,
    predicate: str = "Has",
    relation: str | None = None,
    values: tuple[str, ...] = ("dog", "fur"),
    roles: tuple[str, ...] = ("owner", "possessed"),
    fallback: bool = False,
    polarity: str = "positive",
    factuality: str = "asserted",
    confidence: float = 0.99,
    source_span: str = "A dog has fur.",
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
        "factuality": factuality,
        "confidence": confidence,
        "source_span": source_span,
        "alternatives": [],
    }


def structured_output(*assertions: dict[str, Any]) -> str:
    """Serialize assertions as the model's JSON response."""
    return json.dumps({"assertions": list(assertions)})


def structured_rule_output(
    *,
    antecedents: list[dict[str, Any]],
    consequents: list[dict[str, Any]],
    source_span: str,
) -> str:
    """Serialize fake teacher output containing one conditional rule."""
    return json.dumps(
        {
            "assertions": [],
            "rules": [
                {
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "source_span": source_span,
                }
            ],
        }
    )


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

    def test_contains_temporal_and_spatial_modifier_contract(self) -> None:
        prompt = ReferenceSemanticParser.build_prompt(
            "The engineer designed the bridge in 2025."
        )

        assert "OccursAt(event_or_state, time)" in prompt
        assert "OccursIn(event_or_state, location)" in prompt
        assert "Temporal and other secondary relations must not replace" in prompt
        assert "Spatial modifiers must not replace the core event" in prompt


class TestReferenceSemanticParser:
    def test_generates_normalized_structured_semantics(self) -> None:
        output = structured_output(
            assertion(
                values=(" Apple Computer ", "retail stores"),
                roles=(" OWNER ", " possessed "),
                source_span="Apple Computer has retail stores.",
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
                source_span="The cat is sleeping on the chair.",
            ),
            assertion(
                predicate="On",
                values=("cat", "chair"),
                roles=("entity", "surface"),
                source_span="The cat is sleeping on the chair.",
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
                source_span="Apple did not acquire Tesla.",
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

    def test_preserves_core_event_with_during_modifier(self) -> None:
        output = structured_output(
            assertion(
                predicate="Evaluation",
                relation="collect",
                values=("researcher", "data"),
                roles=("agent", "patient"),
                fallback=True,
                source_span="The researcher collected the data during the experiment.",
            ),
            assertion(
                predicate="During",
                values=("collect", "experiment"),
                roles=("event_or_state", "time_or_event"),
                source_span="The researcher collected the data during the experiment.",
            ),
        )
        parser, _ = make_teacher(output)

        atoms = parser.parse("The researcher collected the data during the experiment.")

        assert [str(atom) for atom in atoms] == [
            "(Evaluation collect (List researcher data))",
            "(During collect experiment)",
        ]

    def test_preserves_core_event_with_occurs_at_modifier(self) -> None:
        output = structured_output(
            assertion(
                predicate="Evaluation",
                relation="design",
                values=("engineer", "bridge"),
                roles=("agent", "patient"),
                fallback=True,
                source_span="The engineer designed the bridge in 2025.",
            ),
            assertion(
                predicate="OccursAt",
                values=("design", "2025"),
                roles=("event_or_state", "time"),
                source_span="The engineer designed the bridge in 2025.",
            ),
        )
        parser, _ = make_teacher(output)

        atoms = parser.parse("The engineer designed the bridge in 2025.")

        assert [str(atom) for atom in atoms] == [
            "(Evaluation design (List engineer bridge))",
            "(OccursAt design time_2025)",
        ]

    def test_preserves_core_event_with_occurs_in_modifier(self) -> None:
        output = structured_output(
            assertion(
                predicate="Evaluation",
                relation="examine",
                values=("doctor", "patient"),
                roles=("agent", "patient"),
                fallback=True,
                source_span="The doctor examined the patient at the hospital.",
            ),
            assertion(
                predicate="OccursIn",
                values=("examine", "hospital"),
                roles=("event_or_state", "location"),
                source_span="The doctor examined the patient at the hospital.",
            ),
        )
        parser, _ = make_teacher(output)

        atoms = parser.parse("The doctor examined the patient at the hospital.")

        assert [str(atom) for atom in atoms] == [
            "(Evaluation examine (List doctor patient))",
            "(OccursIn examine hospital)",
        ]

    def test_distinguishes_entity_location_from_event_location(self) -> None:
        entity_output = structured_output(
            assertion(
                predicate="LocatedIn",
                values=("doctor", "hospital"),
                roles=("entity", "location"),
                source_span="The doctor is in the hospital.",
            )
        )
        event_output = structured_output(
            assertion(
                predicate="Evaluation",
                relation="examine",
                values=("doctor", "patient"),
                roles=("agent", "patient"),
                fallback=True,
                source_span="The doctor examined the patient at the hospital.",
            ),
            assertion(
                predicate="OccursIn",
                values=("examine", "hospital"),
                roles=("event_or_state", "location"),
                source_span="The doctor examined the patient at the hospital.",
            ),
        )

        entity_parser, _ = make_teacher(entity_output)
        event_parser, _ = make_teacher(event_output)

        entity_atoms = entity_parser.parse("The doctor is in the hospital.")
        event_atoms = event_parser.parse(
            "The doctor examined the patient at the hospital."
        )

        assert [str(atom) for atom in entity_atoms] == ["(LocatedIn doctor hospital)"]
        assert [str(atom) for atom in event_atoms] == [
            "(Evaluation examine (List doctor patient))",
            "(OccursIn examine hospital)",
        ]

    @pytest.mark.parametrize(
        "predicate,values,roles",
        [
            (
                "OccursAt",
                ("design", "2025"),
                ("time", "event_or_state"),
            ),
            (
                "OccursIn",
                ("examine", "hospital"),
                ("location", "event_or_state"),
            ),
        ],
    )
    def test_rejects_wrong_modifier_role_order(
        self,
        predicate: str,
        values: tuple[str, ...],
        roles: tuple[str, ...],
    ) -> None:
        parser, _ = make_teacher(
            structured_output(
                assertion(
                    predicate=predicate,
                    values=values,
                    roles=roles,
                )
            )
        )

        with pytest.raises(
            SemanticParseError,
            match="requires argument roles",
        ):
            parser.parse("Supporting sentence.")

    def test_accepts_exact_source_span(self) -> None:
        result = SemanticParseResult.model_validate(
            {
                "assertions": [
                    {
                        "predicate": "Evaluation",
                        "relation": "open",
                        "arguments": [
                            {"value": "Alice", "role": "agent"},
                            {"value": "door", "role": "patient"},
                        ],
                        "fallback": True,
                        "polarity": "positive",
                        "factuality": "asserted",
                        "confidence": 0.99,
                        "source_span": "Alice opened the door.",
                        "alternatives": [],
                    }
                ]
            }
        )

        validated = ReferenceSemanticParser.validate_source_spans(
            result,
            sentence="Alice opened the door.",
        )

        assert validated == result

    def test_accepts_source_span_with_case_and_whitespace_differences(self) -> None:
        result = SemanticParseResult.model_validate(
            {
                "assertions": [
                    {
                        "predicate": "Evaluation",
                        "relation": "open",
                        "arguments": [
                            {"value": "Alice", "role": "agent"},
                            {"value": "door", "role": "patient"},
                        ],
                        "fallback": True,
                        "polarity": "positive",
                        "factuality": "asserted",
                        "confidence": 0.99,
                        "source_span": "alice   opened   the   door.",
                        "alternatives": [],
                    }
                ]
            }
        )

        validated = ReferenceSemanticParser.validate_source_spans(
            result,
            sentence="Alice opened the door.",
        )

        assert validated == result

    def test_accepts_source_span_from_context(self) -> None:
        result = SemanticParseResult.model_validate(
            {
                "assertions": [
                    {
                        "predicate": "Evaluation",
                        "relation": "arrive",
                        "arguments": [
                            {"value": "Bob", "role": "agent"},
                        ],
                        "fallback": True,
                        "polarity": "positive",
                        "factuality": "asserted",
                        "confidence": 0.99,
                        "source_span": "Bob arrived yesterday.",
                        "alternatives": [],
                    }
                ]
            }
        )

        validated = ReferenceSemanticParser.validate_source_spans(
            result,
            sentence="He entered the building.",
            context="Bob arrived yesterday.",
        )

        assert validated == result

    def test_rejects_unsupported_source_span(self) -> None:
        result = SemanticParseResult.model_validate(
            {
                "assertions": [
                    {
                        "predicate": "Evaluation",
                        "relation": "open",
                        "arguments": [
                            {"value": "Alice", "role": "agent"},
                            {"value": "door", "role": "patient"},
                        ],
                        "fallback": True,
                        "polarity": "positive",
                        "factuality": "asserted",
                        "confidence": 0.99,
                        "source_span": "Bob closed the window.",
                        "alternatives": [],
                    }
                ]
            }
        )

        with pytest.raises(
            SemanticParseError,
            match="source_span is not supported by the input text",
        ):
            ReferenceSemanticParser.validate_source_spans(
                result,
                sentence="Alice opened the door.",
            )


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
        from parser.student.student_prompt import build_student_prompt

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


def test_preserves_single_conditional_rule() -> None:
    output = structured_rule_output(
        antecedents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("someone", "red"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="someone is red",
            )
        ],
        consequents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("someone", "nice"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="they are nice",
            )
        ],
        source_span="If someone is red then they are nice.",
    )

    parser, _ = make_teacher(output)

    result = parser.generate_structured("If someone is red then they are nice.")

    assert result.assertions == []
    assert len(result.rules) == 1
    assert len(result.rules[0].antecedents) == 1
    assert len(result.rules[0].consequents) == 1

    atoms = parser.parse("If someone is red then they are nice.")

    assert [str(atom) for atom in atoms] == [
        ("(Implies " "(PropertyOf someone red) " "(PropertyOf someone nice))")
    ]


def test_preserves_multiple_rule_antecedents() -> None:
    output = structured_rule_output(
        antecedents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "big"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry is big",
            ),
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "rough"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry is rough",
            ),
        ],
        consequents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "nice"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry is nice",
            )
        ],
        source_span=("If Harry is big and Harry is rough then Harry is nice."),
    )

    parser, _ = make_teacher(output)

    result = parser.generate_structured(
        "If Harry is big and Harry is rough then Harry is nice."
    )

    rule = result.rules[0]

    assert len(rule.antecedents) == 2
    assert len(rule.consequents) == 1


def test_repairs_conditional_rule_duplicated_as_assertions() -> None:
    antecedent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("Harry", "big"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="Harry is big",
    )

    consequent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("Harry", "nice"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="Harry is nice",
    )

    output = json.dumps(
        {
            "assertions": [
                antecedent,
                consequent,
            ],
            "rules": [
                {
                    "antecedents": [
                        antecedent,
                    ],
                    "consequents": [
                        consequent,
                    ],
                    "source_span": ("If Harry is big then Harry is nice."),
                }
            ],
        }
    )

    parser, _ = make_teacher(output)

    result = parser.generate_structured("If Harry is big then Harry is nice.")

    assert result.assertions == []

    assert len(result.rules) == 1
    assert len(result.rules[0].antecedents) == 1
    assert len(result.rules[0].consequents) == 1

    atoms = parser.parse("If Harry is big then Harry is nice.")

    assert [str(atom) for atom in atoms] == [
        ("(Implies " "(PropertyOf Harry big) " "(PropertyOf Harry nice))")
    ]


def test_rejects_unsupported_source_span_inside_rule() -> None:
    output = structured_rule_output(
        antecedents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "big"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry isbig",
            )
        ],
        consequents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "nice"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry is nice",
            )
        ],
        source_span=("If Harry is big then Harry is nice."),
    )

    parser, _ = make_teacher(output)

    with pytest.raises(
        SemanticParseError,
        match="source_span is not supported by the input text",
    ):
        parser.generate_structured("If Harry is big then Harry is nice.")


def test_rejects_unsupported_rule_source_span() -> None:
    output = structured_rule_output(
        antecedents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "big"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry is big",
            )
        ],
        consequents=[
            assertion(
                predicate="PropertyOf",
                relation=None,
                values=("Harry", "nice"),
                roles=("entity", "property"),
                fallback=False,
                factuality="hypothetical",
                source_span="Harry is nice",
            )
        ],
        source_span=("If Harryis big then Harry is nice."),
    )

    parser, _ = make_teacher(output)

    with pytest.raises(
        SemanticParseError,
        match="rule source_span is not supported by the input text",
    ):
        parser.generate_structured("If Harry is big then Harry is nice.")


def test_preserves_universal_property_rule() -> None:
    antecedent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "big"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="big people",
    )

    consequent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "red"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="red",
    )

    output = json.dumps(
        {
            "assertions": [],
            "rules": [
                {
                    "antecedents": [antecedent],
                    "consequents": [consequent],
                    "source_span": "All big people are red.",
                }
            ],
        }
    )

    parser, _ = make_teacher(output)

    result = parser.generate_structured("All big people are red.")

    assert result.assertions == []
    assert len(result.rules) == 1

    atoms = parser.parse("All big people are red.")

    assert [str(atom) for atom in atoms] == [
        "(Implies (PropertyOf $x big) (PropertyOf $x red))"
    ]


def test_preserves_multi_property_universal_rule() -> None:
    first = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "round"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="round",
    )

    second = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "smart"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="smart things",
    )

    consequent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "furry"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="furry",
    )

    output = json.dumps(
        {
            "assertions": [],
            "rules": [
                {
                    "antecedents": [first, second],
                    "consequents": [consequent],
                    "source_span": ("All round, smart things are furry."),
                }
            ],
        }
    )

    parser, _ = make_teacher(output)

    atoms = parser.parse("All round, smart things are furry.")

    assert [str(atom) for atom in atoms] == [
        (
            "(Implies "
            "(And "
            "(PropertyOf $x round) "
            "(PropertyOf $x smart)) "
            "(PropertyOf $x furry))"
        )
    ]


def repair_rule_duplication(
    result: SemanticParseResult,
) -> SemanticParseResult:
    """Remove standalone assertions redundantly emitted with rules.

    Exact duplicates are always removed.

    Also remove a hypothetical standalone assertion when:
    - it contains a MeTTa variable;
    - that variable also occurs inside a rule; and
    - its source span is contained within that rule's source span.

    This handles redundant universal-rule output such as emitting both
    PropertyOf($x, rough) and the rule
    PropertyOf($x, smart) -> PropertyOf($x, rough),
    without attempting broader semantic rewriting.
    """
    if not result.rules or not result.assertions:
        return result

    repaired = result.model_copy(deep=True)

    rule_assertions = {
        assertion.model_dump_json()
        for rule in repaired.rules
        for assertion in [
            *rule.antecedents,
            *rule.consequents,
        ]
    }

    def assertion_variables(assertion) -> set[str]:
        return {
            argument.value
            for argument in assertion.arguments
            if argument.value.startswith("$")
        }

    rule_metadata = []

    for rule in repaired.rules:
        variables = {
            variable
            for assertion in [
                *rule.antecedents,
                *rule.consequents,
            ]
            for variable in assertion_variables(assertion)
        }

        rule_span = " ".join(rule.source_span.split()).casefold()

        rule_metadata.append((variables, rule_span))

    kept_assertions = []

    for assertion in repaired.assertions:
        if assertion.model_dump_json() in rule_assertions:
            continue

        assertion_vars = assertion_variables(assertion)

        assertion_span = " ".join(assertion.source_span.split()).casefold()

        redundant_rule_assertion = False

        if assertion.factuality == "hypothetical" and assertion_vars:
            for rule_vars, rule_span in rule_metadata:
                if (
                    assertion_vars & rule_vars
                    and assertion_span
                    and assertion_span in rule_span
                ):
                    redundant_rule_assertion = True
                    break

        if not redundant_rule_assertion:
            kept_assertions.append(assertion)

    repaired.assertions = kept_assertions

    return repaired


def test_rule_repair_preserves_independent_asserted_fact() -> None:
    standalone = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("Harry", "blue"),
        roles=("entity", "property"),
        fallback=False,
        factuality="asserted",
        source_span="Harry is blue",
    )

    antecedent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "smart"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="Smart people",
    )

    consequent = assertion(
        predicate="PropertyOf",
        relation=None,
        values=("$x", "rough"),
        roles=("entity", "property"),
        fallback=False,
        factuality="hypothetical",
        source_span="rough",
    )

    output = json.dumps(
        {
            "assertions": [standalone],
            "rules": [
                {
                    "antecedents": [antecedent],
                    "consequents": [consequent],
                    "source_span": "Smart people are rough.",
                }
            ],
        }
    )

    parser, _ = make_teacher(output)

    result = parser.generate_structured("Harry is blue. Smart people are rough.")

    assert len(result.assertions) == 1
    assert result.assertions[0].arguments[0].value == "Harry"


def test_resolves_rule_pronoun_to_unique_generic_participant() -> None:
    first = assertion(
        predicate="Evaluation",
        relation="see",
        values=("someone", "rabbit"),
        roles=("agent", "patient"),
        fallback=True,
        factuality="hypothetical",
        source_span="someone sees the rabbit",
    )

    second = assertion(
        predicate="Evaluation",
        relation="need",
        values=("rabbit", "bear"),
        roles=("agent", "patient"),
        fallback=True,
        factuality="hypothetical",
        source_span="the rabbit needs the bear",
    )

    consequent = assertion(
        predicate="Evaluation",
        relation="need",
        values=("they", "bear"),
        roles=("agent", "patient"),
        fallback=True,
        factuality="hypothetical",
        polarity="negative",
        source_span="they do not need the bear",
    )

    output = json.dumps(
        {
            "assertions": [],
            "rules": [
                {
                    "antecedents": [first, second],
                    "consequents": [consequent],
                    "source_span": (
                        "If someone sees the rabbit and the rabbit needs "
                        "the bear then they do not need the bear."
                    ),
                }
            ],
        }
    )

    parser, _ = make_teacher(output)

    atoms = parser.parse(
        "If someone sees the rabbit and the rabbit needs "
        "the bear then they do not need the bear."
    )

    assert [str(atom) for atom in atoms] == [
        (
            "(Implies "
            "(And "
            "(Evaluation see (List someone rabbit)) "
            "(Evaluation need (List rabbit bear))) "
            "(Not (Evaluation need (List someone bear))))"
        )
    ]


def test_rejects_ambiguous_pronoun_inside_rule() -> None:
    first = assertion(
        predicate="Evaluation",
        relation="see",
        values=("someone", "something"),
        roles=("agent", "patient"),
        fallback=True,
        factuality="hypothetical",
        source_span="someone sees something",
    )

    consequent = assertion(
        predicate="Evaluation",
        relation="leave",
        values=("they",),
        roles=("agent",),
        fallback=True,
        factuality="hypothetical",
        source_span="they leave",
    )

    output = json.dumps(
        {
            "assertions": [],
            "rules": [
                {
                    "antecedents": [first],
                    "consequents": [consequent],
                    "source_span": ("If someone sees something then they leave."),
                }
            ],
        }
    )

    parser, _ = make_teacher(output)

    with pytest.raises(
        SemanticParseError,
        match="Unresolved pronoun inside conditional rule",
    ):
        parser.generate_structured("If someone sees something then they leave.")
