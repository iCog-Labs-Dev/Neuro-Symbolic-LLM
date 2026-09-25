"""Tests for the canonical Hybrid Miner section 6.1 pattern record."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from Residual.symbolic_head.contracts import (
    PatternRecord,
    PatternRecordError,
    validate_q1_pattern_record,
)


def valid_record_data() -> dict[str, object]:
    return {
        "id": "pattern-concession-001",
        "canonical_term": "(Concession $clause_0 $clause_1)",
        "interface_variables": ["$clause_0", "$clause_1"],
        "existential_variables": [],
        "type_constraints": {
            "$clause_0": "Clause",
            "$clause_1": "Clause",
        },
        "semantic_level": "extracted",
        "ontology_groundings": ["discourse-v1:Concession"],
        "semantic_consequences": None,
        "source_spans": [{"document_id": "doc-17", "start": 438, "end": 521}],
        "parser_confidences": [0.91],
        "derivation_or_definition": None,
        "provenance": {
            "corpus_id": "q1-concession-fixtures-v1",
            "parser_version": "semantic-v1",
            "ontology_versions": ["upper-v1", "discourse-v1"],
            "source_document_ids": ["doc-17"],
            "miner_version": "hyperon-miner-unit-fixture-v1",
            "experiment_version": "q1-s1-v1",
        },
        "generator_history": ["greedy:one-clause-seed"],
        "cheap_features": {"support": 4, "clause_count": 1},
        "cheap_endpoint": {"support_estimate": 4},
        "expensive_endpoint": {"exact_support": 4},
        "expensive_trace": {"bindings": 4},
        "controller_prediction": None,
        "controller_uncertainty": None,
        "causal_statistics": None,
        "runtime_cost": 0.004,
        "status": "candidate",
    }


def test_accepts_q1_extracted_pattern() -> None:
    record = PatternRecord.model_validate(valid_record_data())

    assert record.semantic_level == "extracted"
    assert record.source_spans[0].document_id == "doc-17"
    assert record.type_constraints["$clause_0"] == "Clause"
    assert validate_q1_pattern_record(record) is record


def test_canonical_json_round_trip_is_deterministic() -> None:
    first_data = valid_record_data()
    second_data = deepcopy(first_data)
    second_data["type_constraints"] = {
        "$clause_1": "Clause",
        "$clause_0": "Clause",
    }
    second_data["cheap_features"] = {"clause_count": 1, "support": 4}

    first = PatternRecord.model_validate(first_data)
    second = PatternRecord.model_validate(second_data)
    encoded = first.to_canonical_json()

    assert encoded == second.to_canonical_json()
    assert PatternRecord.from_canonical_json(encoded) == first


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_rejects_invalid_parser_confidence(confidence: float) -> None:
    data = valid_record_data()
    data["parser_confidences"] = [confidence]

    with pytest.raises(ValidationError, match="parser confidences"):
        PatternRecord.model_validate(data)


def test_rejects_overlapping_variable_classes() -> None:
    data = valid_record_data()
    data["existential_variables"] = ["$clause_0"]

    with pytest.raises(ValidationError, match="variables overlap"):
        PatternRecord.model_validate(data)


def test_rejects_constraint_for_undeclared_variable() -> None:
    data = valid_record_data()
    raw_constraints = data["type_constraints"]
    assert isinstance(raw_constraints, dict)
    constraints = dict(raw_constraints)
    constraints["$missing"] = "Clause"
    data["type_constraints"] = constraints

    with pytest.raises(ValidationError, match="undeclared variables"):
        PatternRecord.model_validate(data)


def test_rejects_span_document_absent_from_provenance() -> None:
    data = valid_record_data()
    data["source_spans"] = [{"document_id": "doc-untracked", "start": 0, "end": 5}]

    with pytest.raises(ValidationError, match="absent from provenance"):
        PatternRecord.model_validate(data)


def test_extracted_pattern_requires_auditable_evidence() -> None:
    data = valid_record_data()
    data["source_spans"] = []
    data["parser_confidences"] = []
    data["ontology_groundings"] = []

    with pytest.raises(ValidationError, match="require source spans"):
        PatternRecord.model_validate(data)


def test_rejects_non_json_endpoint_payload() -> None:
    data = valid_record_data()
    data["expensive_trace"] = {"invalid": object()}

    with pytest.raises(ValidationError, match="finite JSON data"):
        PatternRecord.model_validate(data)


def test_q1_policy_rejects_later_stage_fields() -> None:
    data = valid_record_data()
    data["semantic_level"] = "derived"
    data["derivation_or_definition"] = "(rule concession-defeat)"
    data["controller_prediction"] = 0.8

    record = PatternRecord.model_validate(data)

    with pytest.raises(PatternRecordError, match="Q1 PatternRecord violations"):
        validate_q1_pattern_record(record)


def test_q1_policy_requires_mining_provenance_and_features() -> None:
    data = valid_record_data()
    raw_provenance = data["provenance"]
    assert isinstance(raw_provenance, dict)
    provenance = dict(raw_provenance)
    provenance["miner_version"] = None
    data["provenance"] = provenance
    data["generator_history"] = []
    data["cheap_features"] = {}

    record = PatternRecord.model_validate(data)

    with pytest.raises(PatternRecordError) as error:
        validate_q1_pattern_record(record)

    message = str(error.value)
    assert "generator_history" in message
    assert "miner_version" in message
    assert "cheap_features" in message


def test_forbids_unknown_fields_and_top_level_mutation() -> None:
    data = valid_record_data()
    data["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PatternRecord.model_validate(data)

    record = PatternRecord.model_validate(valid_record_data())
    with pytest.raises(ValidationError, match="Instance is frozen"):
        record.status = "promoted"
