"""parser/semantic/semantic_parser.py

Structured natural-language semantic parsers.

ReferenceSemanticParser uses a large model (teacher) for offline annotation.
DistilledSemanticParser uses a saved fine-tuned model (student) for inference.

Both produce a SemanticParseResult; MeTTa rendering is a derived view.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from parser.grammar.atomese import LinkAtom
from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.backends import ModelBackend, OutputTruncatedError
from parser.semantic.catalog_prompt import (
    build_catalog_document_prompt,
    build_routed_extraction_prompt,
    build_router_prompt,
)
from parser.semantic.metta_renderer import MettaRenderError
from parser.semantic.metta_renderer import (
    render_metta as render_semantic_result,
)
from parser.semantic.metta_renderer import (
    validate_rendered_metta as parse_rendered_metta,
)
from parser.semantic.normalization import (
    SemanticNormalizationError,
    normalize_extraction_payload,
    restore_quote_case,
)
from parser.semantic.output_cleanup import clean_model_output
from parser.semantic.schema import (
    ParserRoute,
    SemanticAssertion,
    SemanticExtraction,
    SemanticParseProvenance,
    SemanticParseResult,
    SemanticRepair,
    SourceDocument,
    SourceQuote,
)


class SemanticParseError(ValueError):
    """Raised when model output cannot be converted into valid semantics."""


class ModelGenerationError(RuntimeError):
    """Raised when a model backend cannot generate usable output."""


# ── Configuration ──────────────────────────────────────────────────────────────

PARSER_VERSION: Final = "1.0.0"

_PRONOUNS: Final = frozenset({"he", "she", "it", "they", "him", "her", "them"})
_EXPLETIVE_IT_RELATIONS: Final = frozenset(
    {"rain", "snow", "hail", "drizzle", "thunder"}
)


def _iter_all_assertions(result: SemanticParseResult) -> Iterator[SemanticAssertion]:
    """Yield standalone assertions and the parts of every rule."""
    yield from result.extraction.assertions

    for rule in result.extraction.rules:
        yield from rule.antecedents
        yield from rule.consequents


@dataclass(frozen=True, slots=True)
class SemanticParserConfig:
    """Configuration shared by both semantic parser roles."""

    model_name: str
    enable_routing: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")


# ── Base Parser ────────────────────────────────────────────────────────────────


class _BaseSemanticParser:
    """Shared structured semantic extraction pipeline.

    Supports both:
    - Reference (teacher): uses backend (external API) to generate
    - Distilled (student): uses saved model from disk
    """

    parser_role = "semantic"
    max_new_tokens = 256
    num_beams = 4

    def __init__(
        self,
        *,
        backend: ModelBackend | None = None,
        config: SemanticParserConfig,
        model: Any = None,
        tokenizer: Any = None,
        catalog: OntologyCatalog | None = None,
        model_revision: str | None = None,
        require_structured_output: bool = False,
    ) -> None:
        """Create a parser with either backend or loaded model."""
        self._backend = backend
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._catalog = catalog if catalog is not None else load_ontology_catalog()
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision cannot be empty")
        self._model_revision = model_revision
        self._require_structured_output = require_structured_output

    @property
    def provider_name(self) -> str:
        if self._backend is not None:
            return self._backend.provider_name
        return "distilled-local"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def prompt_version(self) -> str:
        """Return the active prompt identity without invoking the model."""
        catalog = self._catalog
        return build_catalog_document_prompt(
            "prompt identity",
            catalog=catalog,
        ).version

    @staticmethod
    def build_prompt(
        sentence: str,
        context: str = "",
        *,
        catalog: OntologyCatalog | None = None,
    ) -> str:
        """Build the same catalog-constrained prompt used for document labels."""
        if not sentence.strip():
            raise ValueError("The sentence cannot be empty")
        return build_catalog_document_prompt(
            sentence,
            catalog=catalog if catalog is not None else load_ontology_catalog(),
            context=context,
        ).text

    @staticmethod
    def clean_model_output(output: str) -> str:
        """Unwrap JSON presentation without altering decoded semantic values."""
        return clean_model_output(output)

    def build_document_prompt(self, source_text: str, context: str = ""):
        """Inspect the prompt for this instance's configured catalog."""
        return build_catalog_document_prompt(
            source_text,
            catalog=self._catalog,
            context=context,
        )

    def repair_known_predicate_contract(
        self, result: SemanticParseResult
    ) -> tuple[SemanticParseResult, list[SemanticRepair]]:
        """Clear relation and fallback on catalog predicates.

        Only Evaluation carries a relation and fallback=true. Clearing these
        fields on a catalog predicate keeps its facts instead of rejecting the
        whole document for a format slip.
        """
        repaired = result.model_copy(deep=True)
        repairs: list[SemanticRepair] = []
        for assertion in _iter_all_assertions(repaired):
            if assertion.predicate == "Evaluation":
                if not assertion.fallback:
                    repairs.append(
                        SemanticRepair(
                            kind="predicate_contract",
                            detail="set fallback=true on Evaluation",
                        )
                    )
                    assertion.fallback = True
                continue

            if assertion.predicate not in self._catalog.relations:
                continue

            if assertion.relation is None and not assertion.fallback:
                continue

            repairs.append(
                SemanticRepair(
                    kind="predicate_contract",
                    detail=f"cleared relation and fallback on {assertion.predicate}",
                )
            )
            assertion.relation = None
            assertion.fallback = False
        return repaired, repairs

    @staticmethod
    def repair_ambiguous_pronouns(
        result: SemanticParseResult,
    ) -> tuple[SemanticParseResult, list[SemanticRepair]]:
        """Remove pronoun kinds and bind unresolved pronouns in universal rules.

        The model resolves a pronoun by giving its entity the kind of what it
        refers to, so a pronoun entity without a kind is unresolved. Entity IDs
        belong to one document, so an unresolved pronoun is still a valid,
        linked participant. Two cases stay unsafe: a pronoun used as a kind,
        which would join unrelated documents, and an unresolved pronoun inside
        a universal rule, which must stand for the rule's variable. A universal
        rule without exactly one variable to bind is rejected rather than
        guessed. Every unresolved pronoun that stays in the document is
        recorded, so the rate of unresolved pronouns can be measured.
        Expletive ``it`` in weather expressions ("it rains") refers to nothing
        and is kept.
        """
        repaired = result.model_copy(deep=True)
        repairs: list[SemanticRepair] = []
        extraction = repaired.extraction

        for entity in extraction.entities:
            if entity.kind in _PRONOUNS:
                repairs.append(
                    SemanticRepair(
                        kind="pronoun_kind",
                        detail=f"cleared pronoun kind {entity.kind!r} on {entity.id!r}",
                    )
                )
                entity.kind = None

        pronouns = {
            entity.id: entity.mention.text.casefold()
            for entity in extraction.entities
            if (
                entity.kind is None
                and entity.mention is not None
                and entity.mention.text.casefold() in _PRONOUNS
            )
        }

        def binds(part: SemanticAssertion, entity_id: str) -> bool:
            if entity_id not in pronouns:
                return False
            return not (
                pronouns[entity_id] == "it" and part.relation in _EXPLETIVE_IT_RELATIONS
            )

        for rule in extraction.rules:
            if rule.scope != "universal":
                continue
            parts = [*rule.antecedents, *rule.consequents]
            used = sorted(
                {
                    argument.entity_id
                    for part in parts
                    for argument in part.arguments
                    if binds(part, argument.entity_id)
                }
            )
            if not used:
                continue
            if len(rule.variables) != 1:
                raise SemanticParseError(
                    f"Pronoun entities {used} inside a universal rule with "
                    f"{len(rule.variables)} variables cannot be resolved"
                )
            variable = rule.variables[0].id
            for part in parts:
                for argument in part.arguments:
                    if binds(part, argument.entity_id):
                        repairs.append(
                            SemanticRepair(
                                kind="pronoun_in_rule",
                                detail=f"bound {argument.entity_id!r} to {variable}",
                            )
                        )
                        argument.entity_id = variable

        kept = sorted(
            {
                argument.entity_id
                for part in _iter_all_assertions(repaired)
                for argument in part.arguments
                if binds(part, argument.entity_id)
            }
        )
        for entity_id in kept:
            repairs.append(
                SemanticRepair(
                    kind="unresolved_pronoun",
                    detail=f"kept unresolved pronoun {entity_id!r} as a participant",
                )
            )

        return repaired, repairs

    def validate_predicates(self, result: SemanticParseResult) -> SemanticParseResult:
        """Require catalog predicates, or Evaluation as the fallback.

        Evaluation must set fallback to true. A catalog predicate must set
        fallback to false, leave relation empty, and keep its confidence
        inside the catalog's confidence range.
        """
        for assertion in _iter_all_assertions(result):
            predicate = assertion.predicate
            if predicate == "Evaluation":
                if not assertion.fallback:
                    raise SemanticParseError("Evaluation must have fallback=true")
                continue
            if predicate not in self._catalog.relations:
                raise SemanticParseError(f"Unknown semantic predicate: {predicate!r}")
            if assertion.fallback:
                raise SemanticParseError(
                    f"{predicate} is a catalog predicate, so fallback must be false"
                )
            if assertion.relation is not None:
                raise SemanticParseError(f"{predicate} should have relation=null")
            lower, upper = self._catalog.relations[predicate].confidence_range
            if not lower <= assertion.confidence <= upper:
                raise SemanticParseError(
                    f"{predicate} confidence {assertion.confidence} is outside "
                    f"the catalog range [{lower}, {upper}]"
                )
        return result

    def validate_arguments(self, result: SemanticParseResult) -> SemanticParseResult:
        """Check entity types and catalog argument signatures.

        Every entity and rule variable must use a catalog type. A catalog
        predicate must have its catalog arity, its argument names as roles in
        catalog order, and an allowed type in every position. Evaluation
        arguments are free-form and are not checked here.
        """
        extraction = result.extraction
        entity_types = {entity.id: entity.entity_type for entity in extraction.entities}
        for entity_id, entity_type in entity_types.items():
            self._require_catalog_type(entity_type, f"Entity {entity_id!r}")

        for assertion in extraction.assertions:
            self._validate_signature(assertion, entity_types)
        for entity in extraction.entities:
            symbolic = entity.symbolic_content

            if symbolic is None:
                continue

            if (
                symbolic.predicate != "Evaluation"
                and symbolic.predicate not in self._catalog.relations
            ):
                raise SemanticParseError(
                    f"Implicit proposition {entity.id!r} uses "
                    f"unknown predicate {symbolic.predicate!r}"
                )

            symbolic_assertion = SemanticAssertion(
                predicate=symbolic.predicate,
                relation=symbolic.relation,
                arguments=symbolic.arguments,
                fallback=symbolic.fallback,
                polarity=symbolic.polarity,
                factuality="hypothetical",
                modality="none",
                confidence=1.0,
                evidence=entity.evidence,
                alternatives=[],
            )

            self._validate_signature(
                symbolic_assertion,
                entity_types,
            )
        for rule in extraction.rules:
            rule_types = dict(entity_types)
            for variable in rule.variables:
                self._require_catalog_type(
                    variable.entity_type, f"Variable {variable.id!r}"
                )
                rule_types[variable.id] = variable.entity_type
            for part in [*rule.antecedents, *rule.consequents]:
                self._validate_signature(part, rule_types)

        return result

    def _require_catalog_type(self, entity_type: str, owner: str) -> None:
        """Reject a type the active catalog does not declare."""
        if entity_type not in self._catalog.types:
            raise SemanticParseError(
                f"{owner} uses unknown type {entity_type!r}; "
                f"allowed types are {sorted(self._catalog.types)}"
            )

    def _validate_signature(
        self, assertion: SemanticAssertion, types: Mapping[str, str]
    ) -> None:
        """Check one catalog assertion against its relation signature."""
        if assertion.predicate == "Evaluation":
            return

        relation = self._catalog.relations[assertion.predicate]
        if len(assertion.arguments) != relation.arity:
            raise SemanticParseError(
                f"{assertion.predicate} requires {relation.arity} arguments, "
                f"got {len(assertion.arguments)}"
            )

        expected_roles = [spec.name for spec in relation.arguments]
        actual_roles = [argument.role for argument in assertion.arguments]
        if actual_roles != expected_roles:
            raise SemanticParseError(
                f"{assertion.predicate} requires argument roles "
                f"{expected_roles!r}, got {actual_roles!r}"
            )

        for argument, spec in zip(assertion.arguments, relation.arguments, strict=True):
            actual = types[argument.entity_id]
            if actual not in spec.allowed_types:
                raise SemanticParseError(
                    f"{assertion.predicate} argument {spec.name!r} must be "
                    f"{'|'.join(spec.allowed_types)}, got {actual!r} "
                    f"for {argument.entity_id!r}"
                )

    def _build_extraction_output_schema(self) -> dict[str, Any]:
        """Build a generation schema with predicate- and rule-specific constraints."""

        schema = SemanticExtraction.model_json_schema()
        assertion_schema = schema["$defs"]["SemanticAssertion"]

        # Every assertion must explicitly provide its predicate contract fields.
        assertion_required = set(assertion_schema["required"])
        assertion_required.update(
            {
                "relation",
                "fallback",
                "arguments",
                "polarity",
                "factuality",
                "modality",
            }
        )
        assertion_schema["required"] = sorted(assertion_required)

        catalog_predicates = sorted(
            predicate
            for predicate in self._catalog.relations
            if predicate not in {"Evaluation", "Proposition"}
        )

        evaluation_branch = {
            "type": "object",
            "properties": {
                "predicate": {"const": "Evaluation"},
                "relation": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "^[a-z][a-z0-9_]*$",
                },
                "fallback": {"const": True},
            },
            "required": ["predicate", "relation", "fallback"],
        }

        assertion_branches = [evaluation_branch]

        if catalog_predicates:
            assertion_branches.append(
                {
                    "type": "object",
                    "properties": {
                        "predicate": {"enum": catalog_predicates},
                        "relation": {"type": "null"},
                        "fallback": {"const": False},
                    },
                    "required": ["predicate", "relation", "fallback"],
                }
            )

        assertion_schema["allOf"] = [{"anyOf": assertion_branches}]

        # Inline rule-literal schemas instead of relying on a nested $ref
        # combined with allOf. This makes the hypothetical constraint explicit
        # at the antecedent/consequent item level for Ollama's JSON decoder.
        rule_schema = schema["$defs"]["SemanticRule"]
        # Require variables to be explicitly provided for every rule.
        # Universal rules must provide at least one variable.
        # Specific rules must explicitly provide an empty list.
        rule_required = set(rule_schema["required"])
        rule_required.add("variables")
        rule_schema["required"] = sorted(rule_required)
        for field in ("antecedents", "consequents"):
            item_schema = {
                **assertion_schema,
                "properties": {
                    **assertion_schema["properties"],
                    "factuality": {
                        "type": "string",
                        "const": "hypothetical",
                    },
                },
                "required": sorted(
                    set(assertion_schema["required"]) | {"arguments", "factuality"}
                ),
            }

            rule_schema["properties"][field]["items"] = item_schema

        # Universal rules require at least one variable; specific rules
        # must not declare universal variables.
        rule_schema["allOf"] = [
            {
                "anyOf": [
                    {
                        "properties": {
                            "scope": {"const": "universal"},
                            "variables": {"minItems": 1},
                        },
                        "required": ["scope", "variables"],
                    },
                    {
                        "properties": {
                            "scope": {"const": "specific"},
                            "variables": {"maxItems": 0},
                        },
                        "required": ["scope", "variables"],
                    },
                ]
            }
        ]

        return schema

    @staticmethod
    def repair_rule_duplication(
        result: SemanticParseResult,
    ) -> tuple[SemanticParseResult, list[SemanticRepair]]:
        """Remove standalone assertions that repeat a part of a rule.

        A standalone assertion is a copy when it has the same predicate,
        relation, polarity, modality, and arguments as a rule part, and its
        evidence lies inside that rule's evidence. Keeping the copy would turn
        a conditional statement into an unconditional fact.
        """
        extraction = result.extraction
        if not extraction.rules or not extraction.assertions:
            return result, []

        def signature(assertion: SemanticAssertion) -> tuple[object, ...]:
            return (
                assertion.predicate,
                assertion.relation,
                assertion.polarity,
                assertion.modality,
                tuple((arg.entity_id, arg.role) for arg in assertion.arguments),
            )

        def extent(quotes: list[SourceQuote]) -> tuple[int, int]:
            offsets = [result.locate(quote) for quote in quotes]
            return min(start for start, _ in offsets), max(end for _, end in offsets)

        rule_parts = [
            (signature(part), extent(rule.evidence))
            for rule in extraction.rules
            for part in [*rule.antecedents, *rule.consequents]
        ]
        kept: list[SemanticAssertion] = []
        repairs: list[SemanticRepair] = []
        for assertion in extraction.assertions:
            start, end = extent(assertion.evidence)
            if any(
                part == signature(assertion) and rule_start <= start and end <= rule_end
                for part, (rule_start, rule_end) in rule_parts
            ):
                name = assertion.predicate + (
                    f"({assertion.relation})" if assertion.relation else ""
                )
                repairs.append(
                    SemanticRepair(
                        kind="rule_duplicate",
                        detail=f"removed standalone {name} repeated in a rule",
                    )
                )
            else:
                kept.append(assertion)

        if not repairs:
            return result, []
        repaired = result.model_copy(
            update={"extraction": extraction.model_copy(update={"assertions": kept})}
        )
        return repaired, repairs

    def _apply_repairs(self, result: SemanticParseResult) -> SemanticParseResult:
        """Run every repair in order and record each change in the provenance."""
        repairs: list[SemanticRepair] = []
        for repair in (
            self.repair_known_predicate_contract,
            self.repair_ambiguous_pronouns,
            self.repair_rule_duplication,
        ):
            result, applied = repair(result)
            repairs.extend(applied)
        if not repairs:
            return result
        provenance = result.provenance.model_copy(
            update={"repairs": [*result.provenance.repairs, *repairs]}
        )
        return result.model_copy(update={"provenance": provenance})

    @staticmethod
    def render_metta(result: SemanticParseResult) -> list[str]:
        """Convert structured semantics into deterministic MeTTa expressions."""
        try:
            return render_semantic_result(result)
        except MettaRenderError as error:
            raise SemanticParseError(str(error)) from error

    @staticmethod
    def validate_rendered_metta(expressions: list[str]) -> list[LinkAtom]:
        """Validate rendered MeTTa and convert it into LinkAtom objects."""
        try:
            return parse_rendered_metta(expressions)
        except MettaRenderError as error:
            raise SemanticParseError(str(error)) from error

    @staticmethod
    def split_top_level_atoms(output: str) -> list[str]:
        """Split whitespace-separated top-level links without splitting nesting."""
        atoms: list[str] = []
        depth = 0
        start: int | None = None

        for idx, ch in enumerate(output):
            if ch.isspace() and depth == 0:
                continue
            if ch == "(":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise SemanticParseError("Unbalanced parentheses")
                if depth == 0 and start is not None:
                    atoms.append(output[start : idx + 1])
                    start = None
            elif depth == 0:
                raise SemanticParseError("Top-level LinkAtom expected")

        if depth != 0:
            raise SemanticParseError("Unbalanced parentheses")
        if not atoms:
            raise SemanticParseError("No MeTTa expressions")
        return atoms

    def _generate_text(self, prompt: str) -> str:
        """Generate text using either backend or saved model."""
        if self._model is not None and self._tokenizer is not None:
            # Distilled: use saved model
            enc = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(self._model.device)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            new_ids = out_ids[0][prompt_len:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        elif self._backend is not None:
            return self._backend.generate(prompt=prompt, model=self.model_name)

        else:
            raise ValueError("No backend or model available")

    def _generate_json_text(
        self, prompt: str, *, output_schema: dict[str, Any] | None = None
    ) -> str:
        """Ask the model for one response and strip its JSON envelope.

        Uses the backend's JSON-schema decoding when an output schema is given and
        the backend supports it. Otherwise it falls back to plain generation, or
        raises ValueError if require_structured_output is set. Backend failures
        become ModelGenerationError; only a truncation keeps its message, since
        other provider errors can contain request details. The text inside the
        envelope is not changed.
        """
        generate_json = getattr(self._backend, "generate_json", None)
        if (
            output_schema is not None
            and self._require_structured_output
            and not callable(generate_json)
        ):
            raise ValueError(
                "The selected backend does not support JSON-schema decoding"
            )
        try:
            if output_schema is not None and callable(generate_json):
                output = generate_json(
                    prompt=prompt, model=self.model_name, schema=output_schema
                )
            else:
                output = self._generate_text(prompt)
        except OutputTruncatedError as error:
            raise ModelGenerationError(
                f"{self.parser_role} parser failed with model {self.model_name!r}: "
                f"{error}"
            ) from error
        except Exception as error:
            raise ModelGenerationError(
                f"{self.parser_role} parser failed with model {self.model_name!r}: "
                f"backend raised {type(error).__name__}"
            ) from error
        return self.clean_model_output(output)

    def generate_extraction_payload(
        self,
        document: SourceDocument,
        *,
        catalog: OntologyCatalog,
        run_discourse: bool = False,
    ) -> tuple[dict[str, Any], str]:
        """Generate one routed semantic extraction."""

        prompt = build_routed_extraction_prompt(
            document.source_text,
            catalog=catalog,
            context=document.context,
            run_discourse=run_discourse,
        )

        cleaned = self._generate_json_text(
            prompt.text,
            output_schema=self._build_extraction_output_schema(),
        )

        if not cleaned:
            raise SemanticParseError("The model returned an empty response")

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise SemanticParseError("The model returned invalid JSON") from error

        if not isinstance(payload, dict):
            raise SemanticParseError("Model output must be a JSON object")

        return payload, prompt.version

    def repair_extraction_payload(
        self,
        document: SourceDocument,
        payload: dict[str, Any],
        validation_error: Exception,
        *,
        catalog: OntologyCatalog,
        run_discourse: bool,
    ) -> dict[str, Any]:
        """Ask the teacher to correct an invalid extraction once."""

        original_prompt = build_routed_extraction_prompt(
            document.source_text,
            catalog=catalog,
            context=document.context,
            run_discourse=run_discourse,
        )

        repair_prompt = (
            original_prompt.text
            + "\n\n## VALIDATION CORRECTION\n"
            + "Your previous extraction failed validation.\n"
            + "Correct the JSON while preserving all supported "
            "semantic information.\n"
            + "Do not invent unsupported entities or assertions.\n"
            + "Every assertion argument must reference a declared "
            "entity or valid rule variable.\n"
            + "Declare all referenced Clause entities with exact "
            "source mentions.\n"
            + "Declare implicit Proposition entities with content, "
            "evidence, and hypothetical symbolic_content.\n"
            + "Never use Proposition as an assertion predicate.\n"
            + "Evaluation assertions must use fallback=true.\n"
            + "Catalog assertions must use fallback=false.\n"
            + "Return the complete corrected SemanticExtraction "
            "JSON only.\n\n"
            + "Validation errors:\n"
            + str(validation_error)
            + "\n\nPrevious JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        cleaned = self._generate_json_text(
            repair_prompt,
            output_schema=self._build_extraction_output_schema(),
        )

        if not cleaned:
            raise SemanticParseError("Teacher returned an empty correction")

        try:
            corrected = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise SemanticParseError(
                "Teacher returned invalid correction JSON"
            ) from error

        if not isinstance(corrected, dict):
            raise SemanticParseError("Corrected extraction must be a JSON object")

        return corrected

    def route_document(self, document: SourceDocument) -> tuple[ParserRoute, str]:
        """Decide whether a discourse pass is needed."""
        prompt = build_router_prompt(document.source_text)

        raw = self._generate_json_text(
            prompt.text,
            output_schema=ParserRoute.model_json_schema(),
        )

        try:
            route = ParserRoute.model_validate_json(raw)
        except ValidationError as error:
            raise SemanticParseError("Invalid router response") from error

        return route, prompt.version

    def parse_document(
        self,
        document: SourceDocument,
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> SemanticParseResult:
        """Parse with one bounded correction attempt for invalid teacher output."""

        catalog = self._catalog

        try:
            prompt_versions: list[str] = []
            run_discourse = False

            # Call 1: route the document when routing is enabled.
            if self._config.enable_routing:
                route, router_version = self.route_document(document)
                run_discourse = route.run_discourse
                prompt_versions.append(router_version)

            # Call 2: extract general semantics, optionally with discourse.
            payload, extraction_version = self.generate_extraction_payload(
                document,
                catalog=catalog,
                run_discourse=run_discourse,
            )
            prompt_versions.append(extraction_version)

            def validate_payload(
                candidate: dict[str, Any],
            ) -> SemanticParseResult:
                """Normalize, validate, and apply existing repairs."""

                normalized = normalize_extraction_payload(
                    candidate,
                    aliases=aliases,
                )
                normalized, quote_changes = restore_quote_case(
                    normalized, document.source_text
                )

                extraction = SemanticExtraction.model_validate(normalized)

                result = SemanticParseResult(
                    document=document,
                    provenance=SemanticParseProvenance(
                        provider=self.provider_name,
                        model=self.model_name,
                        model_revision=self._model_revision,
                        parser_version=PARSER_VERSION,
                        prompt_version="|".join(prompt_versions),
                        ontology_versions=list(catalog.versions),
                        repairs=[
                            SemanticRepair(kind="quote_case", detail=change)
                            for change in quote_changes
                        ],
                    ),
                    extraction=extraction,
                )

                # Existing post-validation repairs.
                result = self._apply_repairs(result)

                # Existing semantic and ontology validation.
                self.validate_predicates(result)
                self.validate_arguments(result)
                validate_against_catalog(result, catalog)

                return result

            try:
                return validate_payload(payload)

            except (
                SemanticNormalizationError,
                SemanticParseError,
                ValidationError,
            ) as first_error:
                # The distilled student does not perform teacher retries.
                if self.parser_role != "reference":
                    raise

                # Call 3 (only on failure): request corrected extraction.
                corrected_payload = self.repair_extraction_payload(
                    document,
                    payload,
                    first_error,
                    catalog=catalog,
                    run_discourse=run_discourse,
                )

                try:
                    return validate_payload(corrected_payload)

                except (
                    SemanticNormalizationError,
                    SemanticParseError,
                    ValidationError,
                ) as second_error:
                    raise SemanticParseError(
                        "Extraction failed after one correction attempt. "
                        f"Original error: {first_error}; "
                        f"Correction error: {second_error}"
                    ) from second_error

        except (
            SemanticNormalizationError,
            SemanticParseError,
            ValidationError,
        ) as error:
            raise SemanticParseError(
                f"Document {document.document_id!r}: {error}"
            ) from error

    def parse(self, sentence: str, context: str = "") -> SemanticParseResult:
        """Parse one sentence, using optional reference-only background."""
        from hashlib import sha256

        if not sentence.strip():
            raise ValueError("The sentence cannot be empty")
        document = SourceDocument(
            document_id="text-"
            + sha256(
                ((context + "\0") if context else "").encode("utf-8")
                + sentence.encode("utf-8")
            ).hexdigest(),
            corpus_id="interactive",
            source_text=sentence,
            context=context,
        )
        return self.parse_document(document)


# ── Reference Parser (Teacher) ───────────────────────────────────────────────


class ReferenceSemanticParser(_BaseSemanticParser):
    """Reference model parser for offline annotation and dataset generation."""

    parser_role = "reference"

    def __init__(
        self,
        *,
        backend: ModelBackend,
        config: SemanticParserConfig,
        catalog: OntologyCatalog | None = None,
        model_revision: str | None = None,
        require_structured_output: bool = False,
    ):
        super().__init__(
            backend=backend,
            config=config,
            catalog=catalog,
            model_revision=model_revision,
            require_structured_output=require_structured_output,
        )


# ── Distilled Parser (Student) ──────────────────────────────────────────────


class DistilledSemanticParser(_BaseSemanticParser):
    """Local parser that loads a saved fine-tuned student model."""

    parser_role = "distilled"

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_new_tokens: int = 3072,
        num_beams: int = 4,
        *,
        catalog: OntologyCatalog | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        metadata_path = self.model_dir / "training_metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {"base_model": "unknown"}
        if not isinstance(self.metadata, dict):
            raise ValueError("Student training metadata must be a JSON object")
        from parser.semantic.catalog_student import validate_catalog_student_checkpoint

        catalog = catalog if catalog is not None else load_ontology_catalog()
        validate_catalog_student_checkpoint(self.model_dir, catalog=catalog)

        # Load tokenizer from disk
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Preserve checkpoint precision only on BF16-capable CUDA devices.
        # An explicit CPU/MPS request must not depend on a GPU also being present.
        supports_bf16 = False
        if torch.cuda.is_available() and (
            device == "auto" or device.startswith("cuda")
        ):
            devices = (
                list(range(torch.cuda.device_count()))
                if device == "auto"
                else [torch.device(device).index]
            )
            supports_bf16 = bool(devices)
            for index in devices:
                with torch.cuda.device(index):
                    supports_bf16 = supports_bf16 and torch.cuda.is_bf16_supported()
        if not supports_bf16:
            print(
                "Using float32 for distilled inference on the requested device",
                file=sys.stderr,
            )

        # Load model from disk
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype="auto" if supports_bf16 else torch.float32,
            device_map=device,
        )
        self.model.eval()

        # Create config
        config = SemanticParserConfig(
            model_name=self.metadata.get("base_model", "unknown"),
        )

        # Initialize base class with model and tokenizer
        super().__init__(
            backend=None,
            config=config,
            model=self.model,
            tokenizer=self.tokenizer,
            catalog=catalog,
        )

        print(f"DistilledSemanticParser loaded from {model_dir}", file=sys.stderr)
        print(
            f"  Base model: {self.metadata.get('base_model', 'unknown')}",
            file=sys.stderr,
        )

    def parse_document(
        self,
        document: SourceDocument,
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> SemanticParseResult:
        """Run one combined student extraction for a source document.

        Unlike the teacher, the student does not call the router or
        perform a separate discourse generation.

        It predicts one SemanticExtraction containing general and
        discourse semantics, followed by deterministic validation.
        """
        from parser.student.student_prompt import build_student_prompt_spec

        catalog = self._catalog

        prompt = build_student_prompt_spec(
            document.source_text,
            context=document.context,
            catalog=catalog,
        )

        # Exactly one model generation.
        cleaned = self._generate_json_text(
            prompt.text,
            output_schema=self._build_extraction_output_schema(),
        )

        if not cleaned:
            raise SemanticParseError("The student returned an empty extraction")

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise SemanticParseError("The student returned invalid JSON") from error

        if not isinstance(payload, dict):
            raise SemanticParseError("Student extraction must be a JSON object")

        try:
            normalized, quote_changes = restore_quote_case(
                normalize_extraction_payload(
                    payload,
                    aliases=aliases,
                ),
                document.source_text,
            )
            extraction = SemanticExtraction.model_validate(normalized)

            result = SemanticParseResult(
                document=document,
                provenance=SemanticParseProvenance(
                    provider=self.provider_name,
                    model=self.model_name,
                    model_revision=self._model_revision,
                    parser_version=PARSER_VERSION,
                    prompt_version=prompt.version,
                    ontology_versions=list(catalog.versions),
                    repairs=[
                        SemanticRepair(kind="quote_case", detail=change)
                        for change in quote_changes
                    ],
                ),
                extraction=extraction,
            )

            # Deterministic repairs, not additional model calls.
            result = self._apply_repairs(result)

            # Validate the complete extraction.
            validate_against_catalog(result, catalog)

        except (
            SemanticNormalizationError,
            SemanticParseError,
            ValidationError,
        ) as error:
            raise SemanticParseError(
                f"Student document {document.document_id!r}: {error}"
            ) from error

        return result

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "auto",
        max_new_tokens: int = 3072,
        num_beams: int = 4,
        *,
        catalog: OntologyCatalog | None = None,
    ) -> DistilledSemanticParser:
        """Load a saved model for inference."""
        return cls(model_dir, device, max_new_tokens, num_beams, catalog=catalog)


# ── Builder functions ─────────────────────────────────────────────────────────


def validate_against_catalog(
    result: SemanticParseResult, catalog: OntologyCatalog
) -> None:
    """Run the parser's catalog checks on an existing result, such as a label.

    Raises SemanticParseError for the same predicates, argument signatures,
    types, and confidence ranges that parse_document rejects.
    """
    checker = _BaseSemanticParser(
        config=SemanticParserConfig(model_name=result.provenance.model),
        catalog=catalog,
    )
    checker.validate_predicates(result)
    checker.validate_arguments(result)


def build_distilled_parser(
    model_dir: str = "models/semantic_document_parser",
    device: str = "auto",
    max_new_tokens: int = 3072,
    num_beams: int = 4,
) -> DistilledSemanticParser:
    """Build a distilled parser from a saved model directory."""
    return DistilledSemanticParser.from_pretrained(
        model_dir=model_dir,
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
