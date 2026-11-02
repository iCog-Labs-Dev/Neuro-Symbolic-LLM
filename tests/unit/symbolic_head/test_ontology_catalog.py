"""Modular ontology signatures, version identities and schema validation."""

from pathlib import Path

import pytest

from residual.symbolic_head.contracts import (
    OntologyCatalogError,
    load_ontology_catalog,
)

_CORE_RELATIONS = {
    "Entity",
    "Event",
    "Proposition",
    "Clause",
    "Agent",
    "Patient",
    "Time",
    "Negated",
    "Causes",
    "Concession",
    "Contrast",
    "EvokesExpectation",
    "Defeats",
    "DiscourseFocus",
    "CounterexampleTo",
    "ArgMin",
    "Transformation",
    "MeasureDecrease",
    "OpenObligation",
    "CreatesObligation",
    "Owes",
    "Content",
    "Discharges",
    "Kick",
    "Bucket",
    "IdiomaticSense",
    "DeathEvent",
    "LiteralMotionContext",
    "Application",
    "Lambda",
    "Map",
    "Fold",
}


def _write_catalog(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_loads_complete_modular_catalog() -> None:
    catalog = load_ontology_catalog()

    assert catalog.relations.keys() >= _CORE_RELATIONS
    assert catalog.versions == (
        "discourse-v2",
        "general-v1",
        "lexical-v1",
        "narrative-v2",
        "program-v1",
        "proof-v1",
        "upper-v1",
    )
    assert catalog.relation("Concession").arguments[0].allowed_types == ("Clause",)
    assert catalog.relation("Concession").arguments[1].allowed_types == ("Clause",)
    assert catalog.relation("Defeats").arguments[0].allowed_types == ("Clause",)
    assert catalog.relation("Defeats").arguments[1].allowed_types == ("Proposition",)
    assert "AliasOf" not in catalog.relations
    assert "PipelineRole" not in catalog.relations
    assert catalog.relation("Kick").arity == 3
    assert catalog.relation("Fold").arity == 3


def test_rejects_arity_argument_mismatch(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path / "bad.yaml",
        """
version: bad-v1
relations:
  Entity:
    arity: 2
    arguments:
      - {name: entity, type: String}
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
""",
    )

    with pytest.raises(OntologyCatalogError, match="declares arity 2"):
        load_ontology_catalog(tmp_path)


def test_rejects_duplicate_relations_across_modules(tmp_path: Path) -> None:
    definition = """
version: {version}
relations:
  Entity:
    arity: 1
    arguments:
      - {{name: entity, type: String}}
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
"""
    _write_catalog(tmp_path / "one.yaml", definition.format(version="one-v1"))
    _write_catalog(tmp_path / "two.yaml", definition.format(version="two-v1"))

    with pytest.raises(OntologyCatalogError, match="Duplicate relation 'Entity'"):
        load_ontology_catalog(tmp_path)


def test_rejects_unknown_argument_type(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path / "bad.yaml",
        """
version: bad-v1
relations:
  Entity:
    arity: 1
    arguments:
      - {name: entity, type: MissingType}
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
""",
    )

    with pytest.raises(OntologyCatalogError, match=r"unknown types \['MissingType'\]"):
        load_ontology_catalog(tmp_path)


def test_accepts_declared_types_matching_unary_relations(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path / "upper.yaml",
        """
version: upper-v1
types: [Entity]
relations:
  Entity:
    arity: 1
    arguments:
      - {name: entity, type: Entity}
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
""",
    )

    catalog = load_ontology_catalog(tmp_path)

    assert catalog.relation("Entity").arguments[0].allowed_types == ("Entity",)


@pytest.mark.parametrize(
    "declared_types, argument_type, message",
    [
        ("[Missing]", "Entity", "must have a unary relation"),
        ("[Entity]", "String", "must accept its own type"),
        ("[Entity, Entity]", "Entity", "duplicate type names"),
        ("Entity", "Entity", "list of non-empty names"),
    ],
)
def test_rejects_invalid_type_declarations(
    tmp_path: Path, declared_types: str, argument_type: str, message: str
) -> None:
    _write_catalog(
        tmp_path / "upper.yaml",
        f"""
version: upper-v1
types: {declared_types}
relations:
  Entity:
    arity: 1
    arguments:
      - {{name: entity, type: {argument_type}}}
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
""",
    )

    with pytest.raises(OntologyCatalogError, match=message):
        load_ontology_catalog(tmp_path)


@pytest.mark.parametrize(
    "confidence_range",
    ["[-0.1, 1.0]", "[0.8, 1.1]", "[0.9, 0.8]", "[true, 1.0]"],
)
def test_rejects_invalid_confidence_range(
    tmp_path: Path, confidence_range: str
) -> None:
    _write_catalog(
        tmp_path / "bad.yaml",
        f"""
version: bad-v1
relations:
  Entity:
    arity: 1
    arguments:
      - {{name: entity, type: String}}
    confidence_range: {confidence_range}
    extraction_instruction: Extract an entity.
""",
    )

    with pytest.raises(OntologyCatalogError, match="confidence_range"):
        load_ontology_catalog(tmp_path)
