"""Build general and discourse extraction prompts from ontology catalogs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from parser.ontology.catalog import (
    OntologyCatalog,
    load_ontology_catalog,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "parser_config"

_GENERAL_PROMPT_PATH = _CONFIG_DIR / "general_prompt.yaml"
_DISCOURSE_PROMPT_PATH = _CONFIG_DIR / "discourse_prompt.yaml"
_ROUTER_PROMPT_PATH = _CONFIG_DIR / "parser_prompt.yaml"


@dataclass(frozen=True, slots=True)
class CatalogDocumentPrompt:
    """Rendered general prompt and its provenance identity."""

    version: str
    text: str


@dataclass(frozen=True, slots=True)
class DiscoursePrompt:
    """Rendered discourse prompt and its provenance identity."""

    version: str
    text: str


# ── Ontology rendering ───────────────────────────────────────────────────────


_INPUT_HEADING = re.compile(r"^[ \t]*## \d+\. INPUT\b", re.MULTILINE)


def _general_catalog_versions(
    catalog: OntologyCatalog,
) -> tuple[str, ...]:
    """Return versions containing general, non-discourse predicates."""

    return tuple(
        version
        for version in catalog.versions
        if any(
            relation.ontology != "discourse" and relation.ontology_version == version
            for relation in catalog.relations.values()
        )
    )


def _catalog_guide(catalog: OntologyCatalog) -> str:
    """Render general ontology predicates, excluding discourse relations."""

    lines = []

    for name, relation in sorted(catalog.relations.items()):
        if relation.ontology == "discourse":
            continue

        arguments = ", ".join(
            f"{argument.name}: {'|'.join(argument.allowed_types)}"
            for argument in relation.arguments
        )

        lower, upper = relation.confidence_range

        lines.append(
            f"- {name}({arguments}) "
            f"[confidence {lower:g}..{upper:g}]: "
            f"{relation.extraction_instruction}"
        )

    if not lines:
        raise ValueError("The selected ontology catalog contains no general predicates")

    return "\n".join(lines)


def _discourse_catalog_guide(catalog: OntologyCatalog) -> str:
    """Render discourse predicates and their exact argument signatures."""

    relations = {
        name: relation
        for name, relation in catalog.relations.items()
        if relation.ontology == "discourse"
    }

    if not relations:
        raise ValueError(
            "The selected ontology catalog contains no discourse predicates"
        )

    versions = sorted({relation.ontology_version for relation in relations.values()})

    lines = [
        f"Discourse ontology versions: {', '.join(versions)}",
        "",
    ]

    for name, relation in sorted(relations.items()):
        arguments = ", ".join(
            f"{argument.name}: {'|'.join(argument.allowed_types)}"
            for argument in relation.arguments
        )

        lower, upper = relation.confidence_range

        lines.append(
            f"- {name}({arguments}) "
            f"[confidence {lower:g}..{upper:g}]: "
            f"{relation.extraction_instruction}"
        )

    return "\n".join(lines)


# ── Configuration loading ────────────────────────────────────────────────────


def _load_prompt_config(path: Path, label: str) -> tuple[str, str]:
    """Load and validate one prompt YAML configuration."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"Cannot load {label} prompt configuration: {error}"
        ) from error

    if not isinstance(raw, dict):
        raise ValueError(f"{label.capitalize()} prompt configuration must be a mapping")

    version = raw.get("prompt_version")
    system_prompt = raw.get("system_prompt")

    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"{label.capitalize()} prompt requires a nonempty prompt_version"
        )

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(
            f"{label.capitalize()} prompt requires a nonempty system_prompt"
        )

    return version, system_prompt


def _require_placeholders(
    system_prompt: str,
    placeholders: tuple[str, ...],
    label: str,
) -> None:
    """Ensure that all required prompt placeholders are present."""

    missing = [
        placeholder for placeholder in placeholders if placeholder not in system_prompt
    ]

    if missing:
        raise ValueError(
            f"{label.capitalize()} prompt is missing placeholders: "
            + ", ".join(missing)
        )


def _prompt_fingerprint(guidance: str) -> str:
    """Calculate a stable fingerprint for prompt instructions."""

    return sha256(guidance.encode("utf-8")).hexdigest()


# ── General semantic prompt ──────────────────────────────────────────────────


def build_catalog_document_prompt(
    source_text: str,
    *,
    catalog: OntologyCatalog,
    context: str = "",
) -> CatalogDocumentPrompt:
    """Build the general semantic extraction prompt."""

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must contain non-whitespace text")

    if not isinstance(context, str):
        raise ValueError("context must be a string")

    version, system_prompt = _load_prompt_config(
        _GENERAL_PROMPT_PATH,
        "general",
    )

    _require_placeholders(
        system_prompt,
        (
            "{ontology_versions}",
            "{catalog_guide}",
            "{context}",
            "{source_text}",
        ),
        "general",
    )

    general_versions = _general_catalog_versions(catalog)

    if not general_versions:
        raise ValueError(
            "No general ontology modules are available in the selected catalog"
        )

    guidance = (
        system_prompt.replace(
            "{ontology_versions}",
            ", ".join(general_versions),
        ).replace(
            "{catalog_guide}",
            _catalog_guide(catalog),
        )
    ).rstrip()

    fingerprint = _prompt_fingerprint(guidance)

    rendered = guidance.replace("{context}", context).replace(
        "{source_text}", source_text
    )

    return CatalogDocumentPrompt(
        version=f"{version}+sha256.{fingerprint}",
        text=rendered,
    )


# ── Discourse prompt ─────────────────────────────────────────────────────────


_DISCOURSE_GUIDANCE = """
Preserve the identity of existing clauses, events, entities, and propositions.

Reuse existing document-local IDs whenever they are supplied.

Do not duplicate general semantic assertions.

Extract only discourse relationships supported by the original source.

Create implicit propositions only when a contextually supported expectation
is required by a discourse relationship.

Implicit propositions must have canonical semantic content and source-grounded
supporting evidence, but must not have fabricated source mentions.

Preserve genuine ambiguity and distinguish clauses, events, and propositions.

The original source text is authoritative for evidence.
General semantic extraction is context for interpretation, not a source
for inventing additional facts.
""".strip()


def build_discourse_prompt(
    source_text: str,
    clause_context: str,
    general_semantics: str,
    *,
    catalog: OntologyCatalog | None = None,
) -> DiscoursePrompt:
    """Build the specialized discourse extraction prompt.

    Uses the discourse ontology and the general extraction as context.
    Does not repeat general semantic assertions.
    """

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text cannot be empty")

    if not isinstance(clause_context, str) or not clause_context.strip():
        raise ValueError("clause_context cannot be empty")

    if not isinstance(general_semantics, str):
        raise ValueError("general_semantics must be a string")

    active_catalog = catalog or load_ontology_catalog(modules=("upper", "discourse"))

    version, system_prompt = _load_prompt_config(
        _DISCOURSE_PROMPT_PATH,
        "discourse",
    )

    _require_placeholders(
        system_prompt,
        (
            "{discourse_catalog_guide}",
            "{extraction_guidance}",
            "{source_text}",
            "{clause_context}",
            "{general_semantics}",
        ),
        "discourse",
    )

    catalog_guide = _discourse_catalog_guide(active_catalog)

    guidance = (
        system_prompt.replace(
            "{discourse_catalog_guide}",
            catalog_guide,
        ).replace(
            "{extraction_guidance}",
            _DISCOURSE_GUIDANCE,
        )
    ).rstrip()

    fingerprint = _prompt_fingerprint(guidance)

    rendered = (
        guidance.replace("{source_text}", source_text)
        .replace("{clause_context}", clause_context)
        .replace("{general_semantics}", general_semantics)
    )

    return DiscoursePrompt(
        version=f"{version}+sha256.{fingerprint}",
        text=rendered,
    )


# ── Routing prompt ───────────────────────────────────────────────────────────


def build_router_prompt(source_text: str) -> CatalogDocumentPrompt:
    """Render the lightweight routing decision prompt."""
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text cannot be empty")

    version, template = _load_prompt_config(_ROUTER_PROMPT_PATH, "router")

    _require_placeholders(template, ("{source_text}",), "router")

    return CatalogDocumentPrompt(
        version=f"{version}+sha256.{_prompt_fingerprint(template)}",
        text=template.replace("{source_text}", source_text),
    )


def build_routed_extraction_prompt(
    source_text: str,
    *,
    catalog: OntologyCatalog,
    run_discourse: bool,
    context: str = "",
) -> CatalogDocumentPrompt:
    """Build one extraction prompt for the selected route."""

    general = build_catalog_document_prompt(
        source_text,
        catalog=catalog,
        context=context,
    )

    if not run_discourse:
        return general

    discourse_version, discourse_template = _load_prompt_config(
        _DISCOURSE_PROMPT_PATH,
        "discourse",
    )

    _require_placeholders(
        discourse_template,
        ("{discourse_catalog_guide}",),
        "discourse",
    )

    discourse_guidance = discourse_template.replace(
        "{discourse_catalog_guide}",
        _discourse_catalog_guide(catalog),
    ).strip()

    # Keep the shared general instructions in one place.
    # Insert the discourse extension before the source input
    # so the source is not sent twice.

    input_pattern = re.compile(r"(?m)^##\s+\d+\.\s+INPUT\s*$")

    input_sections = list(input_pattern.finditer(general.text))

    if len(input_sections) != 1:
        raise ValueError(
            "General prompt must contain exactly one numbered INPUT section"
        )

    input_section_match = input_sections[0]

    input_heading = input_section_match.group(0)

    general_instructions = general.text[: input_section_match.start()]

    input_section = general.text[input_section_match.end() :]
    combined = (
        discourse_guidance
        + "\n\n"
        + general_instructions.rstrip()
        + "\n\n"
        + """
    ## FINAL COMBINED EXTRACTION REQUIREMENTS

    You must perform BOTH stages in this single generation.

    STAGE 1: DISCOURSE ANALYSIS
    - Identify all relevant clauses.
    - Extract all supported discourse relationships.
    - Identify supported implicit propositions.
    - Preserve shared entity and proposition IDs.

    STAGE 2: GENERAL SEMANTIC EXTRACTION
    - Extract the semantic content of EVERY relevant clause.
    - Include actions, participants, semantic roles, and polarity.
    - Preserve negation independently of discourse relations.
    - Do not omit general assertions because discourse
    assertions already describe the relationship between clauses.
    - Reuse the entities and IDs identified in Stage 1.

    FINAL OUTPUT:
    Return exactly ONE SemanticExtraction JSON object containing:
    - entities (BOTH general and discourse entities)
    - assertions (BOTH discourse and general assertions)
    - rules

    ENTITY COMPLETENESS IS MANDATORY:

    The entities array MUST include every entity referenced
    by any assertion.

    When discourse analysis identifies clauses:
    - Include each clause as an explicit Clause entity.
    - Include exact source-grounded clause mentions.
    - Reuse those clause IDs in discourse assertions.

    When discourse analysis identifies an implicit expectation:
    - Include a Proposition entity with origin="implicit".
    - Include natural-language content and source evidence.
    - Include hypothetical symbolic_content.
    - Reference that Proposition entity from EvokesExpectation
      and Defeats.
    - Never generate predicate="Proposition" as an assertion.

    Before returning the JSON:
    1. Collect every entity_id used in assertions.
    2. Check that each ID exists in entities.
    3. Check that every implicit Proposition has symbolic_content.
    4. Check that Evaluation assertions use fallback=true.
    5. Check that catalog assertions use fallback=false.
    6. Check that all referenced entity IDs are consistent.

    Never return an extraction with undeclared entity IDs.

    Before returning, verify that every clause containing an
    explicit action or semantic relationship has its corresponding
    general semantic assertion or rule representation.

    Do not return only discourse assertions.
    Do not return a separate DiscourseExtraction object.
    """
        + "\n\n"
        + input_heading
        + "\n"
        + input_section
    )

    return CatalogDocumentPrompt(
        version=(
            f"{general.version}|"
            f"{discourse_version}+sha256."
            f"{_prompt_fingerprint(discourse_guidance)}"
        ),
        text=combined,
    )
