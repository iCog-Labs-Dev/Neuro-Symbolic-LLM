"""Private model-generation step for catalog-guided document extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.backends import ModelBackend
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.output_cleanup import clean_model_output
from parser.semantic.semantic_parser import ModelGenerationError, SemanticParseError


@dataclass(frozen=True, slots=True)
class _CatalogModelOutput:
    """Untrusted model fields; not a validated semantic document."""

    prompt_version: str
    payload: dict[str, object]


def _generate_catalog_payload(
    source_text: str,
    *,
    catalog: OntologyCatalog,
    backend: ModelBackend,
    model_name: str,
) -> _CatalogModelOutput:
    """Generate JSON fields only; the consumer must validate the full contract."""

    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    prompt = build_catalog_document_prompt(source_text, catalog=catalog)
    try:
        output = backend.generate(prompt=prompt.text, model=model_name)
    except Exception as error:
        raise ModelGenerationError(
            f"Catalog parser failed with model {model_name!r}"
        ) from error

    cleaned = clean_model_output(output)
    if not cleaned:
        raise ModelGenerationError("The model returned an empty response")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise SemanticParseError("The model returned invalid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"entities", "assertions"}:
        raise SemanticParseError(
            "Catalog output must contain only entities and assertions"
        )
    if not isinstance(payload["entities"], list) or not isinstance(
        payload["assertions"], list
    ):
        raise SemanticParseError("Catalog entities and assertions must be lists")
    return _CatalogModelOutput(prompt_version=prompt.version, payload=payload)
