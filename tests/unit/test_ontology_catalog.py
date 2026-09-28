"""Contract tests for the shared, versioned ontology catalog."""

from pathlib import Path

import pytest

from parser.ontology.catalog import OntologyCatalogError, load_ontology_catalog


def _write_module(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_loads_versioned_catalog_with_explicit_types() -> None:
    catalog = load_ontology_catalog()

    assert catalog.types == frozenset({"Entity", "Event", "Proposition", "Clause"})
    assert catalog.versions == (
        "discourse-v1",
        "lexical-v1",
        "narrative-v1",
        "program-v1",
        "proof-v1",
        "upper-v1",
    )
    concession = catalog.relation("Concession")
    assert concession.ontology_version == "discourse-v1"
    assert concession.arity == 2
    assert [argument.allowed_types for argument in concession.arguments] == [
        ("Clause",),
        ("Clause",),
    ]
    assert "OpenObligation" not in catalog.types


def test_rejects_unary_relation_used_as_undeclared_type(tmp_path: Path) -> None:
    _write_module(
        tmp_path / "upper.yaml",
        """
version: upper-v1
types: [Entity]
relations:
  Entity:
    arity: 1
    arguments: [{name: id, type: String}]
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
  OpenObligation:
    arity: 1
    arguments: [{name: condition, type: Entity}]
    confidence_range: [0.5, 1.0]
    extraction_instruction: Extract an open obligation.
  RefersTo:
    arity: 1
    arguments: [{name: target, type: OpenObligation}]
    confidence_range: [0.5, 1.0]
    extraction_instruction: Extract a reference.
""",
    )

    with pytest.raises(OntologyCatalogError, match="unknown types"):
        load_ontology_catalog(tmp_path)


def test_declared_type_requires_unary_relation(tmp_path: Path) -> None:
    _write_module(
        tmp_path / "upper.yaml",
        """
version: upper-v1
types: [Entity]
relations:
  Entity:
    arity: 2
    arguments:
      - {name: left, type: String}
      - {name: right, type: String}
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
""",
    )

    with pytest.raises(OntologyCatalogError, match="must have a unary relation"):
        load_ontology_catalog(tmp_path)


def test_rejects_duplicate_relation_across_modules(tmp_path: Path) -> None:
    definition = """
version: {version}
{types}
relations:
  Entity:
    arity: 1
    arguments: [{{name: id, type: String}}]
    confidence_range: [0.8, 1.0]
    extraction_instruction: Extract an entity.
"""
    _write_module(
        tmp_path / "one.yaml",
        definition.format(version="one-v1", types="types: [Entity]"),
    )
    _write_module(
        tmp_path / "two.yaml",
        definition.format(version="two-v1", types=""),
    )

    with pytest.raises(OntologyCatalogError, match="Duplicate relation"):
        load_ontology_catalog(tmp_path)


@pytest.mark.parametrize(
    "confidence_range",
    ["[-0.1, 1.0]", "[0.8, 1.1]", "[0.9, 0.8]", "[true, 1.0]"],
)
def test_rejects_invalid_confidence_bounds(
    tmp_path: Path, confidence_range: str
) -> None:
    _write_module(
        tmp_path / "upper.yaml",
        f"""
version: upper-v1
types: [Entity]
relations:
  Entity:
    arity: 1
    arguments: [{{name: id, type: String}}]
    confidence_range: {confidence_range}
    extraction_instruction: Extract an entity.
""",
    )

    with pytest.raises(OntologyCatalogError, match="confidence_range"):
        load_ontology_catalog(tmp_path)
