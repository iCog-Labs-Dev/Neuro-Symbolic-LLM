"""Validated data contract for structured semantic-parser output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SemanticSchemaModel(BaseModel):
    """Base model shared by every structured semantic value."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SemanticArgument(SemanticSchemaModel):
    """One argument participating in a semantic relation."""

    value: str = Field(min_length=1)
    role: str = Field(min_length=1)
    type: str | None = None


class SemanticAssertion(SemanticSchemaModel):
    """One semantic relation extracted from the input text."""

    predicate: str = Field(min_length=1)
    relation: str | None = None

    # Evaluation may represent a participantless event.
    # Other predicates must still contain at least one argument here,
    # and their exact arity is validated against predicate_schema.yaml.
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

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    source_span: str = Field(
        min_length=1,
    )

    alternatives: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_argument_presence(self) -> SemanticAssertion:
        """Allow zero arguments only for Evaluation."""
        if self.predicate != "Evaluation" and not self.arguments:
            raise ValueError(
                f"{self.predicate} must contain at least one semantic argument"
            )

        return self


class SemanticParseResult(SemanticSchemaModel):
    """Complete structured output for one parser request."""

    assertions: list[SemanticAssertion] = Field(min_length=1)
