"""Compact, versioned combined extraction prompt for the student.

The student generates one SemanticExtraction containing general
and discourse semantics. Teacher routing and correction are not
part of the student inference workflow.

The same prompt builder is used during training and inference.
"""

from __future__ import annotations

from hashlib import sha256

from parser.ontology.catalog import (
    OntologyCatalog,
    load_ontology_catalog,
)
from parser.semantic.catalog_prompt import CatalogDocumentPrompt

STUDENT_PROMPT = "catalog-combined-compact-v1"


def _predicate_guide(catalog: OntologyCatalog) -> str:
    """Render compact predicate signatures from the active catalog."""
    lines = []

    for name, relation in sorted(catalog.relations.items()):
        arguments = ", ".join(
            f"{arg.name}:{'|'.join(arg.allowed_types)}" for arg in relation.arguments
        )

        lower, upper = relation.confidence_range

        lines.append(
            f"- {name}({arguments}) "
            f"[confidence {lower:g}..{upper:g}]: "
            f"{relation.extraction_instruction}"
        )

    if not lines:
        raise ValueError("Ontology catalog has no predicates")

    return "\n".join(lines)


def _instructions(catalog: OntologyCatalog) -> str:
    """Build the shared student instructions and ontology guide."""
    return (
        "You are a semantic extraction model.\n"
        "Return exactly one valid SemanticExtraction JSON object.\n"
        "Extract BOTH general and discourse semantics in one response.\n\n"
        "OUTPUT CONTRACT\n"
        'Top-level JSON: {"entities":[], "assertions":[], "rules":[]}\n'
        "Use the exact fields and allowed values of the v2 "
        "SemanticExtraction schema. Do not output metadata or provenance.\n\n"
        "ENTITIES\n"
        "Each entity has id, kind, entity_type, origin, mention, "
        "content, evidence, symbolic_content.\n"
        "Explicit entities require a source mention and must not have "
        "implicit content, evidence, or symbolic_content.\n"
        "Implicit entities must have entity_type=Proposition and "
        "origin=implicit, with no source mention.\n"
        "Every implicit Proposition requires nonempty canonical content, "
        "source-grounded evidence, and structured symbolic_content.\n"
        "symbolic_content has predicate, relation, arguments, fallback, "
        "polarity, and factuality. Its factuality must be hypothetical.\n"
        "Use a catalog predicate with exact argument roles and types, "
        "or Evaluation with a base-form relation and fallback=true.\n"
        "For Evaluation symbolic_content, relation is required. "
        "Catalog predicates must have their required arguments.\n"
        "Each symbolic_content argument references a declared entity "
        "using entity_id and role.\n"
        "Proposition is an entity_type, never an assertion predicate.\n"
        "An inferred expectation is hypothetical, not an asserted fact. "
        "Create one only when supported by source evidence.\n"
        "Use unique IDs and reuse them consistently.\n\n"
        "ASSERTIONS\n"
        "Each assertion has predicate, relation, arguments, fallback, "
        "polarity, factuality, modality, confidence, evidence, "
        "alternatives.\n"
        "Arguments have entity_id and role; reference declared "
        "entities or rule variables.\n"
        "Use catalog predicates and exact argument signatures.\n"
        "Use Evaluation with a base-form relation when no specific "
        "predicate fits; set fallback=true.\n"
        "Allowed polarity: positive, negative.\n"
        "Allowed factuality: asserted, opinion, speculative, hypothetical.\n"
        "Allowed modality: none, obligation, permission.\n"
        "Confidence must be between 0 and 1.\n"
        "Every assertion requires source evidence.\n\n"
        "RULES\n"
        "Each rule has scope, variables, antecedents, consequents, "
        "evidence.\n"
        "Scope is universal or specific.\n"
        "Universal rules require typed variables with IDs such as "
        "$party; specific rules must not declare variables.\n"
        "Antecedents and consequents contain assertions with "
        "factuality=hypothetical.\n"
        "Do not assert rule conditions as independent facts.\n\n"
        "DISCOURSE\n"
        "Extract supported discourse relationships using discourse "
        "predicates from the catalog.\n"
        "Reuse existing entity/proposition IDs where appropriate.\n"
        "Create implicit propositions only when supported by source "
        "evidence. Do not invent source mentions.\n"
        "Avoid duplicate assertions.\n\n"
        "GROUNDING\n"
        "Evidence and mentions use SourceQuote objects: "
        '{"text":"exact source substring","occurrence":0}.\n'
        "All quotations must occur in the source text, not only "
        "in reference context.\n"
        "Do not invent entities, relations, rules, or evidence.\n"
        "Preserve negation, modality, uncertainty, and ambiguity.\n"
        "Return JSON only; no Markdown or explanation.\n\n"
        "ONTOLOGY PREDICATES\n" + _predicate_guide(catalog)
    )


def build_student_prompt_spec(
    sentence: str,
    context: str = "",
    *,
    catalog: OntologyCatalog | None = None,
) -> CatalogDocumentPrompt:
    """Build the compact prompt and its stable compatibility version."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Source text must be nonempty")

    if not isinstance(context, str):
        raise ValueError("Context must be a string")

    active_catalog = catalog if catalog is not None else load_ontology_catalog()

    guidance = _instructions(active_catalog)

    fingerprint = sha256((STUDENT_PROMPT + "\n" + guidance).encode("utf-8")).hexdigest()

    prompt_text = (
        guidance
        + "\n\nREFERENCE CONTEXT (interpretation only)\n"
        + (context if context.strip() else "(none)")
        + "\n\nSOURCE TEXT\n"
        + sentence
        + "\n\nSemanticExtraction JSON:\n"
    )

    return CatalogDocumentPrompt(
        version=f"{STUDENT_PROMPT}+sha256.{fingerprint}",
        text=prompt_text,
    )


def build_student_prompt(
    sentence: str,
    context: str = "",
    *,
    catalog: OntologyCatalog | None = None,
) -> str:
    """Return prompt text for existing callers."""
    return build_student_prompt_spec(
        sentence,
        context=context,
        catalog=catalog,
    ).text
