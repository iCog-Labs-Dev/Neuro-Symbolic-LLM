"""Semantic parser output contract, version 2.

The parser's own types: what the model extracts from one source document and
the record the parser builds around it. This module does not depend on
hybrid-miner; the parser-miner adapter converts a SemanticParseResult into
hybrid-miner's SemanticDocument.

1. Source pointers: SourceQuote.
2. Model output (the generation target): SemanticEntity, SemanticArgument,
   SemanticAssertion, SemanticRuleVariable, SemanticRule, SemanticExtraction.
3. Parse record: SourceDocument (the input), plus SemanticRepair,
   SemanticParseProvenance and SemanticParseResult (added by the parser).
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEMANTIC_CONTRACT_VERSION: Final = "2.0.0"

_SNAKE_CASE = r"^[a-z][a-z0-9_]*$"
_VARIABLE = r"^\$[a-z][a-z0-9_]*$"
_NO_WHITESPACE = r"^\S+$"
_NOT_BLANK = r"\S"


class SemanticSchemaModel(BaseModel):
    """Base model shared by every structured semantic value."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ── Source pointers ──────────────────────────────────────────────────────────


class SourceQuote(SemanticSchemaModel):
    """Exact words from the source text, located by occurrence number.

    The model copies the words; SemanticParseResult.locate computes the
    character offsets. ``occurrence`` selects which copy is meant when the
    same words appear more than once (0 is the first).
    """

    text: str = Field(pattern=_NOT_BLANK)
    occurrence: int = Field(default=0, ge=0)


# ── Model output ─────────────────────────────────────────────────────────────


class SemanticEntity(SemanticSchemaModel):
    """An entity mentioned in the source text, identified once and reused.

    ``id`` links every assertion and rule about the same thing. ``entity_type``
    is the ontology type checked against predicate slots (``Entity``); ``kind``
    is the lemma of the noun that names it (``supplier``).
    """

    id: str = Field(pattern=_SNAKE_CASE)
    kind: str | None = Field(default=None, pattern=_SNAKE_CASE)
    entity_type: str = Field(pattern=_NO_WHITESPACE)
    mention: SourceQuote


class SemanticArgument(SemanticSchemaModel):
    """One participant in a relation and its role, such as agent or patient.

    ``entity_id`` refers to a declared entity or, in a universal rule, to one
    of the rule's variables.
    """

    entity_id: str = Field(min_length=1)
    role: str = Field(min_length=1)


class SemanticAssertion(SemanticSchemaModel):
    """A relation extracted from the text, such as a fact, opinion, or obligation.

    A catalog predicate carries its own meaning. When none fits, the model uses
    ``Evaluation`` with ``fallback`` set to true and ``relation`` set to the
    base form of the verb (``deliver``). ``factuality`` says how the text
    presents the claim; ``modality`` marks obligations and permissions, and
    "shall not" or "may not" use negative polarity. ``alternatives`` lists
    other readings when the text is genuinely ambiguous. Rule parts use the
    same type.
    """

    predicate: str = Field(pattern=_NO_WHITESPACE)
    relation: str | None = Field(default=None, pattern=_SNAKE_CASE)

    arguments: list[SemanticArgument] = Field(default_factory=list)

    fallback: bool = False

    polarity: Literal[
        "positive",
        "negative",
    ] = "positive"

    factuality: Literal[
        "asserted",
        "opinion",
        "speculative",
        "hypothetical",
    ] = "asserted"

    modality: Literal[
        "none",
        "obligation",
        "permission",
    ] = "none"

    confidence: float = Field(ge=0.0, le=1.0)

    evidence: list[SourceQuote] = Field(min_length=1)

    alternatives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_argument_presence(self) -> SemanticAssertion:
        """Require arguments for every predicate except Evaluation.

        Evaluation may describe an event with no participants, but it must
        name its relation. The exact arity of other predicates is checked by
        the parser against the predicate catalog.
        """

        if self.predicate != "Evaluation" and not self.arguments:
            raise ValueError(
                f"{self.predicate} must contain at least one semantic argument"
            )

        if self.predicate == "Evaluation" and not self.relation:
            raise ValueError("Evaluation must specify a relation")

        return self


class SemanticRuleVariable(SemanticSchemaModel):
    """A variable for any entity of one type in a universal rule, such as $party."""

    id: str = Field(pattern=_VARIABLE)
    entity_type: str = Field(pattern=_NO_WHITESPACE)


class SemanticRule(SemanticSchemaModel):
    """An if-then rule: its consequents hold when its antecedents hold.

    A ``universal`` rule speaks about any entity through variables ("Each
    Party shall ..."); a ``specific`` rule speaks about entities in this text
    ("If the Supplier fails ..."). No part of a rule is asserted on its own,
    so every antecedent and consequent must be hypothetical.
    """

    scope: Literal[
        "universal",
        "specific",
    ]
    variables: list[SemanticRuleVariable] = Field(default_factory=list)

    antecedents: list[SemanticAssertion] = Field(min_length=1)
    consequents: list[SemanticAssertion] = Field(min_length=1)

    evidence: list[SourceQuote] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rule(self) -> SemanticRule:
        """Validate scope, variables, and hypothetical literals."""

        if self.scope == "universal" and not self.variables:
            raise ValueError("Universal rules must contain variables")

        if self.scope == "specific" and self.variables:
            raise ValueError("Specific rules must not contain variables")

        variable_ids = [variable.id for variable in self.variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("Rule variable IDs must be unique")

        for literal in self.antecedents + self.consequents:
            if literal.factuality != "hypothetical":
                raise ValueError(
                    "Every assertion in a rule must have hypothetical factuality"
                )

        return self


class SemanticExtraction(SemanticSchemaModel):
    """The entities, assertions, and rules extracted from one source document.

    This is the only part the model generates; document IDs, provenance, and
    character offsets are added by the parser.
    """

    entities: list[SemanticEntity] = Field(default_factory=list)
    assertions: list[SemanticAssertion] = Field(default_factory=list)
    rules: list[SemanticRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_extraction(self) -> SemanticExtraction:
        """Require content, unique entity IDs, and resolvable arguments."""

        if not self.assertions and not self.rules:
            raise ValueError("Extraction must contain at least one assertion or rule")

        entity_ids = [entity.id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Entity IDs must be unique")

        known = set(entity_ids)
        for assertion in self.assertions:
            _require_known_arguments(assertion, known)

        for rule in self.rules:
            allowed = known | {variable.id for variable in rule.variables}
            for literal in rule.antecedents + rule.consequents:
                _require_known_arguments(literal, allowed)

        return self


def _require_known_arguments(assertion: SemanticAssertion, allowed: set[str]) -> None:
    """Reject arguments that do not point to a declared entity or variable."""

    for argument in assertion.arguments:
        if argument.entity_id not in allowed:
            raise ValueError(
                f"{assertion.predicate} argument refers to an unknown entity "
                f"or rule variable {argument.entity_id!r}"
            )


# ── Parse record ─────────────────────────────────────────────────────────────


class SourceDocument(SemanticSchemaModel):
    """The source text and its identifiers, with optional background context.

    ``context`` is shown to the model but never quoted: quoting it would count
    the same facts once for every document that repeats it.
    """

    document_id: str = Field(pattern=_NO_WHITESPACE)
    corpus_id: str = Field(pattern=_NO_WHITESPACE)

    source_text: str = Field(pattern=_NOT_BLANK)
    context: str = ""


class SemanticRepair(SemanticSchemaModel):
    """One change the parser made to the model's output, and why.

    Repairs are never silent: each one is recorded in the provenance, so its
    rate per kind can be measured as a signal of parser quality. An
    ``unresolved_pronoun`` record changes nothing; it marks a pronoun the
    model left unresolved and the parser kept as a document participant.
    """

    kind: Literal[
        "predicate_contract",
        "pronoun_kind",
        "pronoun_in_rule",
        "unresolved_pronoun",
        "rule_duplicate",
    ]
    detail: str = Field(pattern=_NOT_BLANK)


class SemanticParseProvenance(SemanticSchemaModel):
    """The provider, model, parser, prompt, and ontology versions behind a parse.

    Recorded by the parser from its configuration; never generated by the model.
    ``repairs`` lists every change the parser made to the model's output.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_revision: str | None = None

    parser_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    ontology_versions: list[str] = Field(min_length=1)

    repairs: list[SemanticRepair] = Field(default_factory=list)


class SemanticParseResult(SemanticSchemaModel):
    """The complete parser output: source document, provenance, and extraction.

    Validation guarantees that every quote resolves inside the source text.
    """

    contract_version: Literal["2.0.0"] = SEMANTIC_CONTRACT_VERSION
    document: SourceDocument
    provenance: SemanticParseProvenance
    extraction: SemanticExtraction

    def locate(self, quote: SourceQuote) -> tuple[int, int]:
        """Return the half-open character offsets of a quote in the source text."""

        source = self.document.source_text
        start = -1

        for _ in range(quote.occurrence + 1):
            start = source.find(quote.text, start + 1)
            if start == -1:
                raise ValueError(
                    f"Quote {quote.text!r} occurrence {quote.occurrence} "
                    "is not in the source text"
                )

        return start, start + len(quote.text)

    @model_validator(mode="after")
    def validate_quotes(self) -> SemanticParseResult:
        """Ensure every quote resolves inside the source text."""

        quotes: list[SourceQuote] = [
            entity.mention for entity in self.extraction.entities
        ]

        for assertion in self.extraction.assertions:
            quotes.extend(assertion.evidence)

        for rule in self.extraction.rules:
            quotes.extend(rule.evidence)
            for literal in rule.antecedents + rule.consequents:
                quotes.extend(literal.evidence)

        for quote in quotes:
            self.locate(quote)

        return self
