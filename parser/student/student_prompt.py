"""One prompt builder for document-target student training and inference."""

from __future__ import annotations

from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt

STUDENT_PROMPT = "catalog-document"


def build_student_prompt(
    sentence: str, context: str = "", *, catalog: OntologyCatalog | None = None
) -> str:
    """Use the same extraction guidance, examples and schema as the teacher."""
    return build_catalog_document_prompt(
        sentence,
        catalog=catalog if catalog is not None else load_ontology_catalog(),
        context=context,
    ).text
