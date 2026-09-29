"""Catalog-derived prompt for the semantic-document extraction route."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from parser.ontology.catalog import OntologyCatalog

_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "catalog_document_prompt.yaml"
)


@dataclass(frozen=True, slots=True)
class CatalogDocumentPrompt:
    """Rendered prompt and the version to record in parser provenance."""

    version: str
    text: str


def _catalog_guide(catalog: OntologyCatalog) -> str:
    lines = []
    for name, relation in sorted(catalog.relations.items()):
        arguments = ", ".join(
            f"{argument.name}: {'|'.join(argument.allowed_types)}"
            for argument in relation.arguments
        )
        lower, upper = relation.confidence_range
        lines.append(
            f"- {name}({arguments}) [confidence {lower:g}..{upper:g}]: "
            f"{relation.extraction_instruction}"
        )
    return "\n".join(lines)


def build_catalog_document_prompt(
    source_text: str, *, catalog: OntologyCatalog
) -> CatalogDocumentPrompt:
    """Render catalog guidance without modifying the source or its offsets."""

    if not source_text.strip():
        raise ValueError("source_text must contain non-whitespace text")

    raw = yaml.safe_load(_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Catalog prompt configuration must be a mapping")
    version = raw.get("prompt_version")
    system_prompt = raw.get("system_prompt")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Catalog prompt requires a nonempty prompt_version")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Catalog prompt requires a nonempty system_prompt")
    if (
        "{ontology_versions}" not in system_prompt
        or "{catalog_guide}" not in system_prompt
    ):
        raise ValueError("Catalog prompt is missing its ontology placeholders")

    guidance = system_prompt.replace(
        "{ontology_versions}", ", ".join(catalog.versions)
    ).replace("{catalog_guide}", _catalog_guide(catalog))
    return CatalogDocumentPrompt(
        version=version,
        text=f"{guidance.rstrip()}\n\n<source_text>\n{source_text}\n</source_text>",
    )
