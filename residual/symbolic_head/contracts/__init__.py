"""Stable data contracts shared across Tier 2 boundaries."""

from residual.symbolic_head.contracts.ontology_catalog import (
    ArgumentSpec,
    OntologyCatalog,
    OntologyCatalogError,
    RelationSpec,
    default_catalog_directory,
    load_ontology_catalog,
)
from residual.symbolic_head.contracts.pattern_record import (
    PatternProvenance,
    PatternRecord,
    PatternRecordError,
    SemanticLevel,
    SourceSpan,
    validate_q1_pattern_record,
)

__all__ = [
    "ArgumentSpec",
    "OntologyCatalog",
    "OntologyCatalogError",
    "RelationSpec",
    "default_catalog_directory",
    "load_ontology_catalog",
    "PatternProvenance",
    "PatternRecord",
    "PatternRecordError",
    "SemanticLevel",
    "SourceSpan",
    "validate_q1_pattern_record",
]
