"""Unit tests for structured semantic-parser orchestration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)
from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticParseResult,
    SourceDocument,
)
from parser.semantic.semantic_parser import DistilledSemanticParser
from tests.semantic_fixtures import target, write_checkpoint_metadata


class FakeBackend:
    """Return fixed model output and record generation requests."""

    provider_name = "fake"

    def __init__(self, output: str, *, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        if self.error is not None:
            raise self.error
        return self.output


def _teacher(
    output: str, *, catalog=None, error: Exception | None = None
) -> tuple[ReferenceSemanticParser, FakeBackend]:
    backend = FakeBackend(output, error=error)
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        catalog=catalog,
    )
    return parser, backend


def _has_payload(source: str) -> dict[str, Any]:
    return {
        "entities": [
            {
                "id": "Mira",
                "entity_type": "Entity",
                "mention": {"text": "Mira"},
            },
            {
                "id": "car",
                "entity_type": "Entity",
                "mention": {"text": "car"},
            },
        ],
        "assertions": [
            {
                "predicate": "Has",
                "relation": " own ",
                "arguments": [
                    {"entity_id": "Mira", "role": " OWNER "},
                    {"entity_id": "car", "role": " possessed "},
                ],
                "fallback": True,
                "confidence": 0.95,
                "evidence": [{"text": source}],
            }
        ],
        "rules": [],
    }


def _specific_rule_payload(source: str, *, duplicate_antecedent: bool = False):
    entities = [
        {
            "id": "mira",
            "entity_type": "Entity",
            "mention": {"text": "Mira"},
        },
        {
            "id": "red",
            "entity_type": "Entity",
            "mention": {"text": "red"},
        },
        {
            "id": "nice",
            "entity_type": "Entity",
            "mention": {"text": "nice"},
        },
    ]
    antecedent = {
        "predicate": "PropertyOf",
        "arguments": [
            {"entity_id": "mira", "role": "entity"},
            {"entity_id": "red", "role": "property"},
        ],
        "factuality": "hypothetical",
        "confidence": 0.95,
        "evidence": [{"text": "Mira is red"}],
    }
    consequent = {
        "predicate": "PropertyOf",
        "arguments": [
            {"entity_id": "mira", "role": "entity"},
            {"entity_id": "nice", "role": "property"},
        ],
        "factuality": "hypothetical",
        "confidence": 0.95,
        "evidence": [{"text": "Mira is nice."}],
    }
    assertions = [antecedent] if duplicate_antecedent else []
    return {
        "entities": entities,
        "assertions": assertions,
        "rules": [
            {
                "scope": "specific",
                "antecedents": [antecedent],
                "consequents": [consequent],
                "evidence": [{"text": source}],
            }
        ],
    }


def test_parser_config_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        SemanticParserConfig(model_name="   ")


def test_prompt_contains_source_context_and_selected_catalog() -> None:
    catalog = load_ontology_catalog(modules=("upper",))
    parser, _ = _teacher(json.dumps(target("Mira left.")), catalog=catalog)

    prompt = parser.build_document_prompt("Mira left.", context="Earlier event.")

    assert "Ontology versions: upper-v1" in prompt.text
    assert "<reference_context>\nEarlier event.\n</reference_context>" in prompt.text
    assert "<source_text>\nMira left.\n</source_text>" in prompt.text
    assert "<worked_example>" in prompt.text
    assert prompt.version == parser.prompt_version


def test_parse_returns_versioned_result_and_uses_one_backend_call() -> None:
    source = "Mira left."
    parser, backend = _teacher(json.dumps(target(source)))

    result = parser.parse(source)

    assert isinstance(result, SemanticParseResult)
    assert result.contract_version == SEMANTIC_CONTRACT_VERSION
    assert result.document.source_text == source
    assert result.provenance.provider == "fake"
    assert result.provenance.model == "teacher-model"
    assert result.provenance.prompt_version == parser.prompt_version
    assert len(backend.calls) == 1
    assert backend.calls[0][1] == "teacher-model"


def test_parse_normalizes_names_and_applies_only_explicit_aliases() -> None:
    source = "Mira owns a car."
    payload = _has_payload(source)
    payload["assertions"][0]["predicate"] = "Evaluation"
    parser, _ = _teacher(json.dumps(payload))

    result = parser.parse_document(
        SourceDocument(
            document_id="alias-test",
            corpus_id="unit-tests",
            source_text=source,
        ),
        aliases={"own": "possess"},
    )

    assertion = result.extraction.assertions[0]
    assert [entity.id for entity in result.extraction.entities] == ["mira", "car"]
    assert assertion.predicate == "Evaluation"
    assert assertion.relation == "possess"
    assert assertion.fallback is True
    assert assertion.arguments[0].role == "owner"
    assert assertion.arguments[1].role == "possessed"


def test_repairs_catalog_predicate_contract_and_records_repair() -> None:
    source = "Mira owns a car."
    parser, _ = _teacher(json.dumps(_has_payload(source)))

    result = parser.parse(source)
    assertion = result.extraction.assertions[0]

    assert assertion.predicate == "Has"
    assert assertion.relation is None
    assert assertion.fallback is False
    assert result.provenance.repairs[0].kind == "predicate_contract"


def test_rejects_unknown_predicate_and_catalog_role_order() -> None:
    unknown = target("Mira left.")
    unknown["assertions"][0]["predicate"] = "InventedPredicate"
    with pytest.raises(SemanticParseError, match="Unknown semantic predicate"):
        _teacher(json.dumps(unknown))[0].parse("Mira left.")

    wrong_roles = _has_payload("Mira owns a car.")
    wrong_roles["assertions"][0]["relation"] = None
    wrong_roles["assertions"][0]["fallback"] = False
    wrong_roles["assertions"][0]["arguments"].reverse()
    with pytest.raises(SemanticParseError, match="requires argument roles"):
        _teacher(json.dumps(wrong_roles))[0].parse("Mira owns a car.")


def test_context_is_not_a_source_for_entity_mentions_or_evidence() -> None:
    source = "She left."
    parser, backend = _teacher(json.dumps(target("Nora arrived yesterday.")))

    with pytest.raises(SemanticParseError, match="not in the source text"):
        parser.parse(source, context="Nora arrived yesterday.")
    assert "<reference_context>\nNora arrived yesterday." in backend.calls[0][0]


def test_preserves_specific_rule_and_renders_hypothetical_literals() -> None:
    source = "If Mira is red, Mira is nice."
    parser, _ = _teacher(json.dumps(_specific_rule_payload(source)))

    result = parser.parse(source)
    rule = result.extraction.rules[0]

    assert result.extraction.assertions == []
    assert rule.scope == "specific"
    assert rule.antecedents[0].factuality == "hypothetical"
    assert rule.consequents[0].factuality == "hypothetical"
    assert parser.render_metta(result) == [
        "(Implies (PropertyOf mira red) (PropertyOf mira nice))"
    ]


def test_removes_a_standalone_copy_of_a_rule_literal_and_records_repair() -> None:
    source = "If Mira is red, Mira is nice."
    parser, _ = _teacher(
        json.dumps(_specific_rule_payload(source, duplicate_antecedent=True))
    )

    result = parser.parse(source)

    assert result.extraction.assertions == []
    assert [repair.kind for repair in result.provenance.repairs] == ["rule_duplicate"]


@pytest.mark.parametrize("output", ["not JSON", "[]", "{}"])
def test_rejects_invalid_or_empty_extraction_output(output: str) -> None:
    with pytest.raises(SemanticParseError):
        _teacher(output)[0].parse("Mira left.")


def test_wraps_backend_failures_and_rejects_empty_source() -> None:
    parser, backend = _teacher("", error=RuntimeError("private provider detail"))
    with pytest.raises(ModelGenerationError, match="backend raised RuntimeError"):
        parser.parse("Mira left.")
    assert len(backend.calls) == 1

    backend.calls.clear()
    with pytest.raises(ValueError, match="empty"):
        parser.parse("  ")
    assert backend.calls == []


@pytest.fixture
def distilled_student(monkeypatch, tmp_path):
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    encoding = {
        "input_ids": torch.tensor([[2, 3]]),
        "attention_mask": torch.ones(1, 2),
    }
    tokenizer.return_value.to.return_value = encoding
    tokenizer.decode.return_value = json.dumps(target("Mira left."))
    write_checkpoint_metadata(tmp_path)
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


def test_distilled_parser_uses_shared_prompt_and_returns_current_result(
    distilled_student,
) -> None:
    from parser.student.student_prompt import build_student_prompt

    parser, tokenizer, model = distilled_student
    result = parser.parse("Mira left.")

    assert result.contract_version == SEMANTIC_CONTRACT_VERSION
    tokenizer.assert_called_once_with(
        build_student_prompt("Mira left."),
        return_tensors="pt",
        add_special_tokens=False,
    )
    assert model.generate.call_args.kwargs["max_new_tokens"] == 42
    assert model.generate.call_args.kwargs["num_beams"] == 2


def test_distilled_parser_rejects_old_checkpoint_before_loading_weights(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "training_metadata.json").write_text(
        json.dumps({"target_contract_version": "semantic-document-v1"}),
        encoding="utf-8",
    )
    loader = MagicMock()
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoTokenizer.from_pretrained", loader
    )

    with pytest.raises(ValueError, match="wrong target contract"):
        DistilledSemanticParser.from_pretrained(str(tmp_path))
    loader.assert_not_called()


def test_distilled_precision_respects_requested_device(monkeypatch, tmp_path) -> None:
    write_checkpoint_metadata(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    tokenizer = MagicMock()
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "parser.semantic.semantic_parser.AutoModelForCausalLM.from_pretrained",
        loader,
    )

    DistilledSemanticParser.from_pretrained(str(tmp_path), device="cpu")

    assert loader.call_args.kwargs["torch_dtype"] == torch.float32
    assert loader.call_args.kwargs["device_map"] == "cpu"
