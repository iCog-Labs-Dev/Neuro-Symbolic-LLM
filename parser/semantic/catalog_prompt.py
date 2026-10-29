"""Compose the shared semantic-document prompt from reviewed extraction guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.schema import SemanticExtraction

_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "catalog_document_prompt.yaml"
)
_PROMPT_EXAMPLES_PATH = _PROMPT_CONFIG_PATH.with_name("catalog_document_examples.jsonl")
_EXTRACTION_CONFIG_PATH = _PROMPT_CONFIG_PATH.with_name("parser_prompt.yaml")


def load_shared_extraction_guidance() -> str:
    """Read target-neutral interpretation rules shared with the existing parser."""
    raw = yaml.safe_load(_EXTRACTION_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("parser"), dict):
        raise ValueError("Parser prompt configuration requires a parser mapping")
    guidance = raw["parser"].get("shared_extraction_guidance")
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("Parser prompt requires shared_extraction_guidance")
    return guidance


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


def _worked_examples(catalog: OntologyCatalog) -> str:
    selected_versions = set(catalog.versions)
    rendered = []
    for line_number, line in enumerate(
        _PROMPT_EXAMPLES_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            example = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid catalog example on line {line_number}"
            ) from error
        if not isinstance(example, dict) or set(example) not in (
            {"required_ontology_versions", "source_text", "entities", "assertions"},
            {
                "required_ontology_versions",
                "source_text",
                "entities",
                "assertions",
                "rules",
            },
        ):
            raise ValueError(f"Invalid catalog example fields on line {line_number}")
        required = example["required_ontology_versions"]
        if (
            not isinstance(required, list)
            or not required
            or any(not isinstance(version, str) or not version for version in required)
            or len(set(required)) != len(required)
            or not isinstance(example["source_text"], str)
            or not example["source_text"].strip()
            or not isinstance(example["entities"], list)
            or not isinstance(example["assertions"], list)
        ):
            raise ValueError(f"Invalid catalog example on line {line_number}")
        if not set(required) <= selected_versions:
            continue
        output = SemanticExtraction.model_validate(
            {
                "entities": example["entities"],
                "assertions": example["assertions"],
                "rules": example.get("rules", []),
            }
        ).model_dump(mode="json")
        rendered.append(
            "<worked_example>\n"
            f"<example_source_text>\n{example['source_text']}\n</example_source_text>\n"
            f"<example_output_json>\n{json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n"
            "</example_output_json>\n</worked_example>"
        )
    if not rendered:
        raise ValueError("No catalog examples match the selected ontology versions")
    return "\n".join(rendered)


def build_catalog_document_prompt(
    source_text: str, *, catalog: OntologyCatalog, context: str = ""
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
        or "{worked_examples}" not in system_prompt
        or "{extraction_guidance}" not in system_prompt
    ):
        raise ValueError("Catalog prompt is missing its ontology placeholders")

    guidance = (
        system_prompt.replace("{ontology_versions}", ", ".join(catalog.versions))
        .replace("{catalog_guide}", _catalog_guide(catalog))
        .replace("{extraction_guidance}", load_shared_extraction_guidance())
        .replace("{worked_examples}", _worked_examples(catalog))
    )
    guidance = guidance.rstrip()
    # Source text varies per record; the extraction instructions must not vary
    # silently between annotation, training, and checkpoint inference.
    fingerprint = sha256(guidance.encode("utf-8")).hexdigest()
    return CatalogDocumentPrompt(
        version=f"{version}+sha256.{fingerprint}",
        text=(
            f"{guidance}\n\n"
            + (
                f"<reference_context>\n{context}\n</reference_context>\n"
                if context
                else ""
            )
            + f"<source_text>\n{source_text}\n</source_text>"
        ),
    )
