"""Convert parser results into hybrid-miner semantic documents.

This is the only module that converts parser results into hybrid-miner
records. Each SemanticParseResult becomes one SemanticDocument:

- an Evaluation becomes an Event entity, with one fact for its relation and
  one fact for each participant role;
- the consequents of a specific rule become hypothetical facts that are
  conditional on a Clause entity covering its antecedents;
- a universal rule becomes a miner SemanticRule with signed literals over its
  variables. An Evaluation part becomes one provisional relation over its
  participants, in role order: the event form cannot be used, because the
  miner forbids consequent variables that are missing from the antecedents;
- each entity kind becomes a provisional kind fact;
- content the miner contract cannot express raises MinerAdapterError: a
  negative catalog fact, modality on a catalog predicate or inside a
  universal rule, and an Evaluation without participants inside a universal
  rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from hybrid_miner.contracts.semantic_document import (
    EntityRecord,
    ParserProvenance,
    RuleLiteral,
    RuleVariable,
    SemanticDocument,
    SourceSpan,
)
from hybrid_miner.contracts.semantic_document import SemanticAssertion as MinerAssertion
from hybrid_miner.contracts.semantic_document import SemanticRule as MinerRule
from pydantic import ValidationError

from parser.semantic.schema import (
    SemanticAssertion,
    SemanticParseResult,
    SemanticRule,
    SourceQuote,
)

_ROLE_PREDICATES: Final = {"agent": "Agent", "patient": "Patient", "time": "Time"}
_PROVISIONAL: Final = "ProvisionalRelation"
_STATUS: Final = "extracted-hypothesis"


class MinerAdapterError(ValueError):
    """The parse result cannot be written in hybrid-miner's document contract."""


def to_semantic_document(result: SemanticParseResult) -> SemanticDocument:
    """Convert one validated parse into hybrid-miner's SemanticDocument.

    Raises MinerAdapterError when the result uses something the miner
    contract cannot express.
    """

    try:
        builder = _SemanticDocumentBuilder(result)
        builder.add_entities()

        for assertion in result.extraction.assertions:
            builder.add_assertion(assertion)

        for rule in result.extraction.rules:
            if rule.scope == "universal":
                builder.add_universal_rule(rule)
            else:
                builder.add_specific_rule(rule)

        builder.add_kind_facts()
        return builder.build()

    except ValidationError as error:
        raise MinerAdapterError(
            f"Document {result.document.document_id!r} is not a valid "
            f"SemanticDocument: {error}"
        ) from error


@dataclass
class _SemanticDocumentBuilder:
    """Collects the miner records for one SemanticParseResult."""

    result: SemanticParseResult
    entities: list[EntityRecord] = field(default_factory=list)
    assertions: list[MinerAssertion] = field(default_factory=list)
    rules: list[MinerRule] = field(default_factory=list)
    entity_types: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def _span(self, quote: SourceQuote) -> SourceSpan:
        """Convert a quote into a character span of the source text."""

        start, end = self.result.locate(quote)
        return SourceSpan(start=start, end=end)

    def _record_entity(
        self, identifier: str, entity_type: str, span: SourceSpan
    ) -> None:
        """Register one entity record and remember its type."""

        self.entities.append(EntityRecord(id=identifier, type=entity_type, span=span))
        self.entity_types[identifier] = entity_type

    def _mint(self, prefix: str, entity_type: str, span: SourceSpan) -> str:
        """Create an entity the model did not write, such as an event."""

        number = self.counters.get(prefix, 0)
        self.counters[prefix] = number + 1
        identifier = f"{prefix}-{number}"
        self._record_entity(identifier, entity_type, span)
        return identifier

    def add_entities(self) -> None:
        """Copy the entities written by the model."""

        for entity in self.result.extraction.entities:
            self._record_entity(
                entity.id, entity.entity_type, self._span(entity.mention)
            )

    def add_fact(
        self,
        predicate: str,
        arguments: list[str],
        source: SemanticAssertion,
        *,
        relation: str | None = None,
        conditional_on: str | None = None,
    ) -> None:
        """Add one miner fact that inherits the evidence of a parser assertion."""

        self.assertions.append(
            MinerAssertion(
                predicate=predicate,
                arguments=tuple(arguments),
                argument_types=tuple(self.entity_types[item] for item in arguments),
                status=_STATUS,
                factuality=source.factuality,
                conditional_on=conditional_on,
                confidence=source.confidence,
                source_spans=tuple(self._span(quote) for quote in source.evidence),
                alternatives=tuple(dict.fromkeys(source.alternatives)),
                provisional_relation_id=relation,
            )
        )

    def add_assertion(
        self, assertion: SemanticAssertion, *, conditional_on: str | None = None
    ) -> None:
        """Add the miner facts for one parser assertion."""

        if assertion.predicate == "Evaluation":
            self.add_evaluation(assertion, conditional_on=conditional_on)
            return

        if assertion.modality != "none":
            raise MinerAdapterError(
                f"{assertion.modality} on {assertion.predicate} has no encoding "
                "in the miner contract"
            )

        if assertion.polarity == "negative":
            raise MinerAdapterError(
                f"Negated {assertion.predicate} has no encoding in the miner contract"
            )

        self.add_fact(
            assertion.predicate,
            [argument.entity_id for argument in assertion.arguments],
            assertion,
            conditional_on=conditional_on,
        )

    def add_evaluation(
        self, assertion: SemanticAssertion, *, conditional_on: str | None = None
    ) -> None:
        """Add an Event entity and its relation, role, polarity and modality facts."""

        event = self._mint("event", "Event", self._span(assertion.evidence[0]))
        self.add_fact(
            _PROVISIONAL,
            [event],
            assertion,
            relation=assertion.relation,
            conditional_on=conditional_on,
        )

        for argument in assertion.arguments:
            predicate = _ROLE_PREDICATES.get(argument.role)
            if predicate is not None:
                self.add_fact(
                    predicate,
                    [event, argument.entity_id],
                    assertion,
                    conditional_on=conditional_on,
                )
            else:
                self.add_fact(
                    _PROVISIONAL,
                    [event, argument.entity_id],
                    assertion,
                    relation=f"role_{argument.role}",
                    conditional_on=conditional_on,
                )

        if assertion.polarity == "negative":
            self.add_fact("Negated", [event], assertion, conditional_on=conditional_on)

        if assertion.modality != "none":
            self.add_fact(
                _PROVISIONAL,
                [event],
                assertion,
                relation=assertion.modality,
                conditional_on=conditional_on,
            )

    def add_specific_rule(self, rule: SemanticRule) -> None:
        """Add a Clause for the antecedents and make the consequents depend on it."""

        spans = [
            self._span(quote) for part in rule.antecedents for quote in part.evidence
        ]
        clause = self._mint(
            "clause",
            "Clause",
            SourceSpan(
                start=min(span.start for span in spans),
                end=max(span.end for span in spans),
            ),
        )

        for part in rule.antecedents:
            self.add_assertion(part)

        for part in rule.consequents:
            self.add_assertion(part, conditional_on=clause)

    def add_universal_rule(self, rule: SemanticRule) -> None:
        """Add one universal rule as a miner SemanticRule.

        Rule variables keep their IDs and types. Each antecedent and
        consequent becomes one RuleLiteral that keeps its polarity.
        """

        variable_types = {
            variable.id: variable.entity_type for variable in rule.variables
        }
        self.rules.append(
            MinerRule(
                id=self._next_rule_id(),
                status=_STATUS,
                variables=tuple(
                    RuleVariable(id=variable.id, type=variable.entity_type)
                    for variable in rule.variables
                ),
                antecedents=tuple(
                    self._rule_literal(part, variable_types)
                    for part in rule.antecedents
                ),
                consequents=tuple(
                    self._rule_literal(part, variable_types)
                    for part in rule.consequents
                ),
                source_spans=tuple(self._span(quote) for quote in rule.evidence),
            )
        )

    def _next_rule_id(self) -> str:
        """Return the next document-local rule ID."""

        number = self.counters.get("rule", 0)
        self.counters["rule"] = number + 1
        return f"rule-{number}"

    def _rule_literal(
        self, part: SemanticAssertion, variable_types: dict[str, str]
    ) -> RuleLiteral:
        """Convert one universal-rule part into a RuleLiteral.

        A catalog predicate keeps its name and argument order. An Evaluation
        becomes ``ProvisionalRelation`` named by its relation, with its
        participants in role order. Negation is the literal's polarity; the
        miner contract has no modality, so a modal part is rejected.
        """

        if part.modality != "none":
            raise MinerAdapterError(
                f"{part.modality} in a universal rule has no encoding "
                "in the miner contract"
            )

        identifiers = [argument.entity_id for argument in part.arguments]
        argument_types = []
        for identifier in identifiers:
            entity_type = variable_types.get(identifier) or self.entity_types.get(
                identifier
            )
            if entity_type is None:
                raise MinerAdapterError(
                    f"Rule argument {identifier!r} is not a declared entity or variable"
                )
            argument_types.append(entity_type)

        if part.predicate == "Evaluation":
            if not identifiers:
                raise MinerAdapterError(
                    f"Evaluation {part.relation!r} without participants "
                    "cannot be a universal rule literal"
                )
            predicate, relation = _PROVISIONAL, part.relation
        else:
            predicate, relation = part.predicate, None

        return RuleLiteral(
            predicate=predicate,
            arguments=tuple(identifiers),
            argument_types=tuple(argument_types),
            status=_STATUS,
            polarity=part.polarity,
            confidence=part.confidence,
            source_spans=tuple(self._span(quote) for quote in part.evidence),
            alternatives=tuple(dict.fromkeys(part.alternatives)),
            provisional_relation_id=relation,
        )

    def add_kind_facts(self) -> None:
        """Add one kind fact per entity that has a kind and is used by a fact."""

        extraction = self.result.extraction
        parts = [part for rule in extraction.rules for part in rule.antecedents]
        parts += [part for rule in extraction.rules for part in rule.consequents]

        confidences: dict[str, list[float]] = {}
        for assertion in [*extraction.assertions, *parts]:
            for argument in assertion.arguments:
                confidences.setdefault(argument.entity_id, []).append(
                    assertion.confidence
                )

        for entity in extraction.entities:
            if entity.kind is None or entity.id not in confidences:
                continue
            self.assertions.append(
                MinerAssertion(
                    predicate=_PROVISIONAL,
                    arguments=(entity.id,),
                    argument_types=(entity.entity_type,),
                    status=_STATUS,
                    factuality=None,
                    confidence=min(confidences[entity.id]),
                    source_spans=(self._span(entity.mention),),
                    provisional_relation_id=f"kind_{entity.kind}",
                )
            )

    def build(self) -> SemanticDocument:
        """Assemble the validated SemanticDocument."""

        document = self.result.document
        provenance = self.result.provenance
        return SemanticDocument(
            document_id=document.document_id,
            corpus_id=document.corpus_id,
            source_text=document.source_text,
            parser=ParserProvenance(
                model=provenance.model,
                parser_version=provenance.parser_version,
                prompt_version=provenance.prompt_version,
                ontology_versions=tuple(provenance.ontology_versions),
                provider=provenance.provider,
                model_revision=provenance.model_revision,
            ),
            entities=tuple(self.entities),
            assertions=tuple(self.assertions),
            rules=tuple(self.rules),
        )
