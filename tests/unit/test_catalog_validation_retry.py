"""Bounded regeneration retains source, catalog and label validation."""

import json
from copy import deepcopy

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.backends import OutputTruncatedError
from parser.semantic.catalog_student import (
    make_student_pair,
    validate_catalog_student_pair,
)
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)
from tests.semantic_fixtures import target


class SequenceBackend:
    provider_name = "sequence"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []
        self.schemas = []

    def generate(self, *, prompt, model):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    def generate_json(self, *, prompt, model, schema):
        self.schemas.append(deepcopy(schema))
        return self.generate(prompt=prompt, model=model)


def parser(backend, retries=2):
    return ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="sequence", validation_retries=retries),
        require_structured_output=True,
    )


@pytest.mark.parametrize("defect", ["reference", "quotation", "type", "predicate"])
def test_replacement_is_validated_and_remains_a_student_target(defect):
    source = "Mira left."
    valid = target(source)
    invalid = deepcopy(valid)
    if defect == "reference":
        invalid["entities"] = []
    elif defect == "quotation":
        invalid["assertions"][0]["evidence"][0]["text"] = "Mira departed."
    elif defect == "type":
        invalid["entities"][0]["entity_type"] = "Entity"
    else:
        invalid["assertions"][0]["predicate"] = "NotInTheCatalog"
    backend = SequenceBackend(invalid, valid)
    instance = parser(backend)

    result = instance.parse(source)

    assert len(backend.prompts) == 2
    assert "<validation_feedback>" not in backend.prompts[0]
    assert "<validation_feedback>" in backend.prompts[1]
    assert backend.schemas[0] == backend.schemas[1]
    assert result.document.source_text == source
    assert result.extraction.assertions[0].arguments[0].entity_id == "e1"
    assert result.provenance.prompt_version == instance.prompt_version
    assert result.provenance.repairs[-1].kind == "validation_regeneration"
    assert result.provenance.repairs[-1].detail.endswith("validation retries: 1")
    pair = make_student_pair(result)
    _, label, rebuilt = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog()
    )
    assert rebuilt == result
    assert "validation_feedback" not in label
    assert "Mira departed." not in label


@pytest.mark.parametrize("invalid", ["[", "[]", ""])
def test_invalid_json_or_empty_output_can_be_regenerated(invalid):
    backend = SequenceBackend(invalid, target("Mira left."))
    assert parser(backend).parse("Mira left.").extraction.assertions
    assert len(backend.prompts) == 2


def test_each_replacement_is_checked_and_only_latest_feedback_is_sent():
    valid = target("Mira left.")
    reference_error = deepcopy(valid)
    reference_error["entities"] = []
    quote_error = deepcopy(valid)
    quote_error["assertions"][0]["evidence"][0]["text"] = "invented quotation"
    backend = SequenceBackend(reference_error, quote_error, valid)

    result = parser(backend).parse("Mira left.")

    assert len(backend.prompts) == 3
    assert backend.prompts[-1].count("<validation_feedback>") == 1
    assert "invented quotation" in backend.prompts[-1]
    assert result.provenance.repairs[-1].detail.endswith("validation retries: 2")


def test_permanent_failure_stops_after_the_configured_budget():
    invalid = target("Mira left.")
    invalid["entities"] = []
    backend = SequenceBackend(invalid, invalid, invalid)
    with pytest.raises(SemanticParseError, match="3 generation attempt"):
        parser(backend).parse("Mira left.")
    assert len(backend.prompts) == 3


def test_retries_can_be_disabled():
    invalid = target("Mira left.")
    invalid["entities"] = []
    backend = SequenceBackend(invalid)
    with pytest.raises(SemanticParseError, match="1 generation attempt"):
        parser(backend, retries=0).parse("Mira left.")
    assert len(backend.prompts) == 1


def test_valid_first_draft_has_no_regeneration_record():
    backend = SequenceBackend(target("Mira left."))
    result = parser(backend).parse("Mira left.")
    assert len(backend.prompts) == 1
    assert result.provenance.repairs == []


def test_validation_feedback_is_bounded(monkeypatch):
    valid = target("Mira left.")
    backend = SequenceBackend(valid, valid)
    instance = parser(backend)
    validate = instance._validate_extraction_payload
    attempts = 0

    def reject_first(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SemanticParseError("x" * 5000)
        return validate(*args, **kwargs)

    monkeypatch.setattr(instance, "_validate_extraction_payload", reject_first)
    instance.parse("Mira left.")
    feedback = backend.prompts[-1].split("<validation_feedback>\n", 1)[1]
    feedback = feedback.split("\n</validation_feedback>", 1)[0]
    assert json.loads(feedback)["error"] == "x" * 2000


@pytest.mark.parametrize(
    "error",
    [RuntimeError("connection closed"), OutputTruncatedError("output token limit")],
)
def test_provider_failures_are_not_retried_as_validation_errors(error):
    backend = SequenceBackend(error)
    with pytest.raises(ModelGenerationError):
        parser(backend).parse("Mira left.")
    assert len(backend.prompts) == 1


def test_regeneration_does_not_bypass_existing_recorded_repairs():
    invalid = target("Mira left.")
    invalid["entities"] = []
    valid = target("Mira left.")
    valid["assertions"][0].update(relation="leave", fallback=True)
    backend = SequenceBackend(invalid, valid)

    result = parser(backend).parse("Mira left.")

    assert result.extraction.assertions[0].relation is None
    assert result.extraction.assertions[0].fallback is False
    assert [repair.kind for repair in result.provenance.repairs] == [
        "predicate_contract",
        "validation_regeneration",
    ]


@pytest.mark.parametrize("count", [-1, 3, True, 1.5, None, "2"])
def test_invalid_retry_limits_are_rejected(count):
    with pytest.raises(ValueError, match="validation_retries"):
        SemanticParserConfig(model_name="sequence", validation_retries=count)
