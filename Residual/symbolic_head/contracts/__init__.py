# ruff: noqa: N999 - The lead-mandated component package is named ``Residual``.
"""Stable data contracts shared across Tier 2 boundaries."""

from Residual.symbolic_head.contracts.ontology_catalog import (
    ArgumentSpec,
    OntologyCatalog,
    OntologyCatalogError,
    RelationSpec,
    default_catalog_directory,
    load_ontology_catalog,
)

__all__ = [
    "ArgumentSpec",
    "OntologyCatalog",
    "OntologyCatalogError",
    "RelationSpec",
    "default_catalog_directory",
    "load_ontology_catalog",
]
