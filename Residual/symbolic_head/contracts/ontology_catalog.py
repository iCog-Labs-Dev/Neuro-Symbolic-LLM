"""Strict loader for the modular ontology catalog from Hybrid Miner Appendix A.3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_PRIMITIVE_TYPES = frozenset({"String"})
_DOCUMENT_FIELDS = frozenset({"version", "relations"})
_RELATION_FIELDS = frozenset(
    {"arity", "arguments", "confidence_range", "extraction_instruction"}
)
_ARGUMENT_FIELDS = frozenset({"name", "type"})


class OntologyCatalogError(ValueError):
    """Raised when an ontology document violates the shared catalog contract."""


@dataclass(frozen=True)
class ArgumentSpec:
    """One named, typed argument in a relation signature."""

    name: str
    allowed_types: tuple[str, ...]


@dataclass(frozen=True)
class RelationSpec:
    """Validated relation definition from one ontology module."""

    name: str
    ontology: str
    ontology_version: str
    arguments: tuple[ArgumentSpec, ...]
    confidence_range: tuple[float, float]
    extraction_instruction: str

    @property
    def arity(self) -> int:
        return len(self.arguments)


@dataclass(frozen=True)
class OntologyCatalog:
    """Immutable merged view of independently versioned ontology modules."""

    versions: tuple[str, ...]
    relations: Mapping[str, RelationSpec]

    def relation(self, name: str) -> RelationSpec:
        try:
            return self.relations[name]
        except KeyError as error:
            raise OntologyCatalogError(f"Unknown ontology relation {name!r}") from error


def default_catalog_directory() -> Path:
    """Return the repository's paper-aligned ontology directory."""

    return Path(__file__).resolve().parents[3] / "parser" / "ontology"


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OntologyCatalogError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise OntologyCatalogError(f"{context} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], required: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != required:
        missing = sorted(required - actual)
        unknown = sorted(actual - required)
        raise OntologyCatalogError(
            f"{context} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _parse_argument(value: Any, context: str) -> ArgumentSpec:
    raw = _require_mapping(value, context)
    _require_exact_fields(raw, _ARGUMENT_FIELDS, context)

    name = raw["name"]
    type_expression = raw["type"]
    if not isinstance(name, str) or not name.strip():
        raise OntologyCatalogError(f"{context}.name must be a non-empty string")
    if not isinstance(type_expression, str) or not type_expression.strip():
        raise OntologyCatalogError(f"{context}.type must be a non-empty string")

    allowed_types = tuple(part.strip() for part in type_expression.split("|"))
    if any(not part for part in allowed_types) or len(set(allowed_types)) != len(
        allowed_types
    ):
        raise OntologyCatalogError(f"{context}.type contains an invalid type union")
    return ArgumentSpec(name=name.strip(), allowed_types=allowed_types)


def _parse_confidence_range(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise OntologyCatalogError(f"{context} must contain exactly two numbers")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise OntologyCatalogError(f"{context} must contain only numbers")
    lower, upper = (float(value[0]), float(value[1]))
    if not 0.0 <= lower <= upper <= 1.0:
        raise OntologyCatalogError(f"{context} must satisfy 0 <= lower <= upper <= 1")
    return lower, upper


def _parse_relation(
    name: str,
    value: Any,
    *,
    ontology: str,
    version: str,
) -> RelationSpec:
    context = f"{ontology}.{name}"
    if not name.strip():
        raise OntologyCatalogError(f"{ontology} contains an empty relation name")
    raw = _require_mapping(value, context)
    _require_exact_fields(raw, _RELATION_FIELDS, context)

    arity = raw["arity"]
    if isinstance(arity, bool) or not isinstance(arity, int) or arity < 1:
        raise OntologyCatalogError(f"{context}.arity must be a positive integer")
    raw_arguments = raw["arguments"]
    if not isinstance(raw_arguments, list):
        raise OntologyCatalogError(f"{context}.arguments must be a list")
    arguments = tuple(
        _parse_argument(argument, f"{context}.arguments[{index}]")
        for index, argument in enumerate(raw_arguments)
    )
    if len(arguments) != arity:
        raise OntologyCatalogError(
            f"{context} declares arity {arity} but defines {len(arguments)} arguments"
        )
    argument_names = [argument.name for argument in arguments]
    if len(set(argument_names)) != len(argument_names):
        raise OntologyCatalogError(f"{context} contains duplicate argument names")

    instruction = raw["extraction_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise OntologyCatalogError(
            f"{context}.extraction_instruction must be a non-empty string"
        )
    return RelationSpec(
        name=name,
        ontology=ontology,
        ontology_version=version,
        arguments=arguments,
        confidence_range=_parse_confidence_range(
            raw["confidence_range"], f"{context}.confidence_range"
        ),
        extraction_instruction=instruction.strip(),
    )


def load_ontology_catalog(directory: Path | None = None) -> OntologyCatalog:
    """Load, merge, and validate every ontology YAML module in a directory."""

    catalog_directory = directory or default_catalog_directory()
    paths = sorted(catalog_directory.glob("*.yaml"))
    if not paths:
        raise OntologyCatalogError(
            f"No ontology YAML files found in {catalog_directory}"
        )

    versions: list[str] = []
    relations: dict[str, RelationSpec] = {}
    for path in paths:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise OntologyCatalogError(f"Cannot load ontology file {path}: {error}") from error
        raw = _require_mapping(document, path.name)
        _require_exact_fields(raw, _DOCUMENT_FIELDS, path.name)

        version = raw["version"]
        if not isinstance(version, str) or not version.strip():
            raise OntologyCatalogError(f"{path.name}.version must be a non-empty string")
        if version in versions:
            raise OntologyCatalogError(f"Duplicate ontology version {version!r}")
        versions.append(version)

        raw_relations = _require_mapping(raw["relations"], f"{path.name}.relations")
        if not raw_relations:
            raise OntologyCatalogError(f"{path.name}.relations cannot be empty")
        for name, definition in raw_relations.items():
            if name in relations:
                first = relations[name].ontology
                raise OntologyCatalogError(
                    f"Duplicate relation {name!r} in {first!r} and {path.stem!r}"
                )
            relations[name] = _parse_relation(
                name,
                definition,
                ontology=path.stem,
                version=version,
            )

    declared_types = {
        name for name, relation in relations.items() if relation.arity == 1
    }
    allowed_types = declared_types | _PRIMITIVE_TYPES
    for relation in relations.values():
        for argument in relation.arguments:
            unknown = set(argument.allowed_types) - allowed_types
            if unknown:
                raise OntologyCatalogError(
                    f"{relation.ontology}.{relation.name}.{argument.name} references "
                    f"unknown types {sorted(unknown)}"
                )

    return OntologyCatalog(
        versions=tuple(versions),
        relations=MappingProxyType(relations),
    )
