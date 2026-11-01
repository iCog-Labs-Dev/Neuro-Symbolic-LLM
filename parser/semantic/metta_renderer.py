"""Deterministic conversion from a SemanticParseResult into MeTTa atoms.

The MeTTa view writes entity IDs as symbols, so two entities never collapse
into one name. It shows negation, obligation, permission, and if-then rules:

- "Ethiopia is located in Africa." -> ``(LocatedIn ethiopia africa)``
- "The Supplier delivered the Goods."
  -> ``(Evaluation deliver (List supplier goods))``
- "The Supplier shall deliver the Goods."
  -> ``(Obligation (Evaluation deliver (List supplier goods)))``
- "shall not" and "may not" both forbid the action:
  ``(Obligation (Not ...))`` and ``(Not (Permission ...))``.

Plain MeTTa cannot show how the source presents a claim, so a standalone
opinion, speculation, or hypothetical is refused instead of being shown as a
fact. Rule parts are always hypothetical; ``Implies`` already shows that. The
JSON result remains the complete record.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from parser.grammar.atomese import LinkAtom, parse_atom, validate_metta_string
from parser.semantic.schema import SemanticAssertion, SemanticParseResult

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


def _render_assertion(assertion: SemanticAssertion) -> str:
    """Render one semantic assertion, with its negation and modality, as MeTTa."""
    argument_values = [
        _require_bare_symbol(
            argument.entity_id,
            "argument entity ID",
        )
        for argument in assertion.arguments
    ]

    arguments = " ".join(argument_values)

    if assertion.predicate == "Evaluation":
        if assertion.relation is None:
            raise MettaRenderError("Evaluation requires a relation")

        relation = _require_bare_symbol(
            assertion.relation,
            "relation",
        )

        argument_list = f"(List {arguments})" if arguments else "(List)"

        expression = f"(Evaluation {relation} {argument_list})"

    else:
        predicate = _require_bare_symbol(
            assertion.predicate,
            "predicate",
        )

        expression = f"({predicate} {arguments})"

    negative = assertion.polarity == "negative"

    if assertion.modality == "obligation":
        inner = f"(Not {expression})" if negative else expression
        return f"(Obligation {inner})"

    if assertion.modality == "permission":
        expression = f"(Permission {expression})"

    if negative:
        expression = f"(Not {expression})"

    return expression


def _render_standalone(assertion: SemanticAssertion) -> str:
    """Render a standalone assertion, refusing claims the source does not assert."""
    if assertion.factuality != "asserted":
        name = assertion.relation or assertion.predicate
        raise MettaRenderError(
            f"{name!r} is {assertion.factuality}; MeTTa would show it as a fact, "
            "so use the JSON output"
        )
    return _render_assertion(assertion)


def render_metta(
    result: SemanticParseResult,
    *,
    asserted_only: bool = False,
) -> list[str]:
    """Render validated structured semantics as MeTTa expressions.

    With ``asserted_only``, claims the source does not assert (opinion,
    speculative, and hypothetical) and all rules are left out instead of being
    refused.
    """
    extraction = result.extraction
    expressions: list[str] = []

    for assertion in extraction.assertions:
        if asserted_only and assertion.factuality != "asserted":
            continue
        expressions.append(_render_standalone(assertion))

    if asserted_only:
        return expressions

    for rule in extraction.rules:
        antecedents = [_render_assertion(assertion) for assertion in rule.antecedents]

        consequents = [_render_assertion(assertion) for assertion in rule.consequents]

        if len(antecedents) == 1:
            antecedent = antecedents[0]
        else:
            antecedent = "(And " + " ".join(antecedents) + ")"

        if len(consequents) == 1:
            consequent = consequents[0]
        else:
            consequent = "(And " + " ".join(consequents) + ")"

        expressions.append(f"(Implies {antecedent} {consequent})")

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
