"""Validate catalog-guided model output against the miner's document contract."""

from __future__ import annotations

from hybrid_miner.contracts.semantic_document import SemanticDocument
from pydantic import ValidationError

from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.backends import ModelBackend
from parser.semantic.catalog_generation import _generate_catalog_payload
from parser.semantic.semantic_parser import SemanticParseError

CATALOG_PARSER_VERSION = "catalog-semantic-parser-v1"


class CatalogSemanticDocumentParser:
    """Produce validated extraction hypotheses without rendering Atomese."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        model_name: str,
        catalog: OntologyCatalog | None = None,
        model_revision: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision cannot be empty")
        self._backend = backend
        self._model_name = model_name
        self._catalog = catalog if catalog is not None else load_ontology_catalog()
        self._model_revision = model_revision

    def parse_document(
        self, *, document_id: str, corpus_id: str, source_text: str
    ) -> SemanticDocument:
        """Return a contract- and catalog-validated semantic document.

        Document identity, exact source text, and parser provenance come from
        the caller and runtime, never from the model response.
        """

        generated = _generate_catalog_payload(
            source_text,
            catalog=self._catalog,
            backend=self._backend,
            model_name=self._model_name,
        )
        try:
            document = SemanticDocument.model_validate(
                {
                    "document_id": document_id,
                    "corpus_id": corpus_id,
                    "source_text": source_text,
                    "parser": {
                        "model": self._model_name,
                        "parser_version": CATALOG_PARSER_VERSION,
                        "prompt_version": generated.prompt_version,
                        "ontology_versions": self._catalog.versions,
                        "provider": self._backend.provider_name,
                        "model_revision": self._model_revision,
                    },
                    "entities": generated.payload["entities"],
                    "assertions": generated.payload["assertions"],
                }
            )
            document.validate_against_signatures(
                ontology_versions=self._catalog.versions,
                signatures=self._catalog.signatures(),
            )
            self._validate_catalog_constraints(document)
        except (ValidationError, ValueError) as error:
            raise SemanticParseError(f"Invalid semantic document: {error}") from error
        return document

    def _validate_catalog_constraints(self, document: SemanticDocument) -> None:
        """Enforce catalog limits not covered by the interchange contract."""

        allowed_types = self._catalog.types | {"String"}
        for entity in document.entities:
            if entity.type not in allowed_types:
                raise ValueError(f"Undeclared ontology entity type {entity.type!r}")
        for assertion in document.assertions:
            if assertion.predicate == "ProvisionalRelation":
                continue
            lower, upper = self._catalog.relation(assertion.predicate).confidence_range
            if not lower <= assertion.confidence <= upper:
                raise ValueError(
                    f"{assertion.predicate} confidence {assertion.confidence} "
                    f"is outside catalog range [{lower}, {upper}]"
                )
