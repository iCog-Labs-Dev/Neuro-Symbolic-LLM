"""Validated data contract for structured semantic-parser output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    arguments: list[SemanticArgument] = Field(min_length=1)
    fallback: bool = False
    polarity: Literal["positive", "negative"] = "positive"
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: str = Field(min_length=1)
    alternatives: list[str] = Field(default_factory=list)


class SemanticParseResult(SemanticSchemaModel):
    """Complete structured output for one parser request."""

    assertions: list[SemanticAssertion] = Field(min_length=1)
