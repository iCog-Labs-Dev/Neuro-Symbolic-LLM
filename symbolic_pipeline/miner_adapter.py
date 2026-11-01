"""Convert parser results into hybrid-miner semantic documents.

This is the only module that converts parser results into hybrid-miner
records. Each SemanticParseResult becomes one SemanticDocument:

- an Evaluation becomes an Event entity, with one fact for its relation and
  one fact for each participant role;
- the consequents of a specific rule become hypothetical facts that are
  conditional on a Clause entity covering its antecedents;
- each entity kind becomes a provisional kind fact;
- content the miner contract cannot express raises MinerAdapterError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from hybrid_miner.contracts.semantic_document import (
    EntityRecord,
    ParserProvenance,
    SemanticDocument,
    SourceSpan,
)
from hybrid_miner.contracts.semantic_document import SemanticAssertion as MinerAssertion
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
                raise MinerAdapterError("Universal rules are not supported yet")
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
        )
