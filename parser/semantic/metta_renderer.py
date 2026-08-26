"""Deterministic conversion from structured semantics into MeTTa atoms."""

from __future__ import annotations

import re
from collections.abc import Iterable

from parser.grammar.atomese import LinkAtom, parse_atom, validate_metta_string
from parser.semantic.schema import SemanticParseResult

_BARE_SYMBOL_RE = re.compile(r"^[^\s()]+$")


class MettaRenderError(ValueError):
    """Raised when structured semantics cannot be rendered as safe MeTTa."""


def _require_bare_symbol(value: str, field: str) -> str:
    """Reject values that would change the generated MeTTa structure."""
    if not _BARE_SYMBOL_RE.fullmatch(value):
        raise MettaRenderError(
            f"{field} must be one nonempty MeTTa symbol without whitespace "
            "or parentheses"
        )
    return value


def render_metta(result: SemanticParseResult) -> list[str]:
    """Render validated structured assertions as MeTTa expressions."""
    expressions: list[str] = []

    for assertion in result.assertions:
        argument_values = [
            _require_bare_symbol(argument.value, "argument value")
            for argument in assertion.arguments
        ]
        arguments = " ".join(argument_values)

        if assertion.predicate == "Evaluation":
            if assertion.relation is None:
                raise MettaRenderError("Evaluation requires a relation")
            relation = _require_bare_symbol(assertion.relation, "relation")
            expression = f"(Evaluation {relation} (List {arguments}))"
        else:
            predicate = _require_bare_symbol(assertion.predicate, "predicate")
            expression = f"({predicate} {arguments})"

        if assertion.polarity == "negative":
            expression = f"(Not {expression})"

        expressions.append(expression)

    return expressions


def validate_rendered_metta(expressions: Iterable[str]) -> list[LinkAtom]:
    """Validate rendered expressions and return parsed top-level links."""
    atoms: list[LinkAtom] = []

    for expression in expressions:
        is_valid, error = validate_metta_string(expression)
        if not is_valid:
            raise MettaRenderError(f"Rendered invalid MeTTa: {error}")

        try:
            atom = parse_atom(expression)
        except ValueError as error:
            raise MettaRenderError(
                f"Rendered malformed MeTTa: {expression!r}"
            ) from error
        if not isinstance(atom, LinkAtom):
            raise MettaRenderError("Rendered MeTTa must be a top-level LinkAtom")
        atoms.append(atom)

    return atoms
