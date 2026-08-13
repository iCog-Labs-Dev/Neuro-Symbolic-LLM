"""parser/grammar/atomese.py

The complete MeTTa Atomese data model.

This file has zero external dependencies — pure Python.
It is the foundation that every other symbolic component validates against.
Write this first, test it first, before touching MORK or the parser.

What lives here:
  - Atom data classes (SymbolAtom, LinkAtom)
  - parse_atom():          string → Atom tree
  - atom_to_string():      Atom tree → string
  - validate_metta_string(): is this valid for our 8-predicate vocabulary?
  - match_template():      does this ground atom match this template?
  - generalize():          produce a template from a ground atom
  - canonical():           normalize a string for deduplication

The 14 predicates we use (closed vocabulary):
  Inheritance  "X is a type of Y"           (Inheritance dog animal)
  Evaluation   "X does Y to Z"              (Evaluation likes (List john mary))
  CanDo        "X can do Y"                 (CanDo bird fly)
  On           "X is on/in Y"               (On cup table)
  Cause        "X causes Y"                 (Cause rain flood)
  Has          "X has Y"                    (Has dog fur)
  PartOf       "X is part of Y"             (PartOf wheel car)
  StateOf      "X is in state Y"            (StateOf world stage)
  LocatedIn    "X is located in Y"          (LocatedIn book shelf)
  MemberOf     "X is a member of Y"         (MemberOf alice team)
  UsedFor      "X is used for Y"            (UsedFor knife cutting)
  Before       "X happens before Y"         (Before dawn sunrise)
  After        "X happens after Y"          (After sunrise dawn)
  List         argument list for Evaluation (List john mary)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Constants ────────────────────────────────────────────────────────────────

PREDICATES = frozenset(
    {
        "CanDo",
        "Cause",
        "Evaluation",
        "Has",
        "Inheritance",
        "List",
        "On",
        "PartOf",
        "StateOf",
        "LocatedIn",
        "MemberOf",
        "UsedFor",
        "Before",
        "After",
    }
)

# fixed arity (number of required children)
ARITY = {
    "Inheritance": 2,
    "CanDo": 2,
    "On": 2,
    "Cause": 2,
    "Has": 2,
    "PartOf": 2,
    "StateOf": 2,
    "LocatedIn": 2,
    "MemberOf": 2,
    "UsedFor": 2,
    "Before": 2,
    "After": 2,
    "Evaluation": 2,  # (Evaluation VERB ARGS)
    # List: variable arity (1–4)
}

VARIABLE_RE = re.compile(r"^\$[A-Z]+$")
CONCEPT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
VAR_NAMES = ["$X", "$Y", "$Z", "$W", "$V", "$U"]


# ── Atom data classes ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SymbolAtom:
    """Leaf node — a concept symbol like 'dog', 'run', or a variable '$X'."""

    name: str

    def __str__(self) -> str:
        return self.name

    @property
    def is_variable(self) -> bool:
        return self.name.startswith("$")


@dataclass(frozen=True)
class LinkAtom:
    """Internal node — a typed link with children.

    Example:
        LinkAtom("Inheritance", (SymbolAtom("dog"), SymbolAtom("animal")))
        → str: "(Inheritance dog animal)"
    """

    predicate: str
    children: tuple  # tuple[Atom, ...]

    def __str__(self) -> str:
        parts = " ".join(str(c) for c in self.children)
        return f"({self.predicate} {parts})"

    def __hash__(self):
        return hash((self.predicate, self.children))


Atom = SymbolAtom | LinkAtom


# ── Parser: string → Atom ─────────────────────────────────────────────────────


def parse_atom(s: str) -> Atom:
    """Parse a MeTTa expression string into an Atom tree.

    Usage:
        atom = parse_atom("(Inheritance dog animal)")
        # LinkAtom("Inheritance", (SymbolAtom("dog"), SymbolAtom("animal")))

        atom = parse_atom("(Evaluation likes (List john mary))")
        # LinkAtom("Evaluation", (SymbolAtom("likes"), LinkAtom("List", ...)))

        atom = parse_atom("$X")
        # SymbolAtom("$X")

    Raises ValueError on malformed input.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty atom string")

    if s.startswith("("):
        if not s.endswith(")"):
            raise ValueError(f"Unbalanced parentheses in: {s!r}")
        inner = s[1:-1].strip()
        parts = _tokenize(inner)
        if not parts:
            raise ValueError(f"Empty link atom: {s!r}")
        predicate = parts[0]
        children = tuple(parse_atom(p) for p in parts[1:])
        return LinkAtom(predicate=predicate, children=children)

    return SymbolAtom(name=s)


def _tokenize(s: str) -> list[str]:
    """Split a string into top-level MeTTa tokens, respecting nested parens."""
    tokens = []
    depth = 0
    current = []

    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
            if depth == 0:
                t = "".join(current).strip()
                if t:
                    tokens.append(t)
                current = []
        elif ch == " " and depth == 0:
            t = "".join(current).strip()
            if t:
                tokens.append(t)
            current = []
        else:
            current.append(ch)

    if current:
        t = "".join(current).strip()
        if t:
            tokens.append(t)

    return tokens


# ── Serializer: Atom → string ─────────────────────────────────────────────────


def atom_to_string(atom: Atom) -> str:
    """Serialize an Atom back to its MeTTa string representation."""
    return str(atom)


# ── Validator ─────────────────────────────────────────────────────────────────


def validate_metta_string(s: str) -> tuple[bool, str]:
    """Check whether a string is valid MeTTa for our 8-predicate vocabulary.

    Returns (is_valid: bool, error_message: str).
    error_message is empty string when is_valid is True.

    Used by:
      - tier2_mork/store.py        to validate /insert payloads
      - mork/tier3_worker.py       to validate parser output before writing
      - parser/grammar/mask.py     automaton state decisions

    Usage:
        ok, err = validate_metta_string("(Inheritance dog animal)")
        # ok=True, err=""

        ok, err = validate_metta_string("(UnknownPred x y)")
        # ok=False, err="Unknown predicate 'UnknownPred'"

        ok, err = validate_metta_string("dog")
        # ok=False, err="Top-level atom must be a link (start with '(')"
    """
    s = s.strip()
    if not s:
        return False, "Empty string"

    try:
        atom = parse_atom(s)
    except Exception as e:
        return False, str(e)

    if not isinstance(atom, LinkAtom):
        return False, "Top-level atom must be a link (start with '(')"

    if atom.predicate not in PREDICATES:
        return False, (
            f"Unknown predicate '{atom.predicate}'. "
            f"Must be one of: {sorted(PREDICATES)}"
        )

    # arity checks
    if atom.predicate in ARITY:
        expected = ARITY[atom.predicate]
        if len(atom.children) != expected:
            return False, (
                f"'{atom.predicate}' requires exactly {expected} children, "
                f"got {len(atom.children)}"
            )
    elif atom.predicate == "List":
        if len(atom.children) < 1:
            return False, "List requires at least 1 child"

    return True, ""


# ── Template utilities ────────────────────────────────────────────────────────


def is_template(atom: Atom) -> bool:
    """True if the atom contains at least one variable ($X, $Y, ...)."""
    if isinstance(atom, SymbolAtom):
        return atom.is_variable
    return any(is_template(c) for c in atom.children)


def get_variables(atom: Atom) -> set[str]:
    """Return all variable names in an atom tree."""
    if isinstance(atom, SymbolAtom):
        return {atom.name} if atom.is_variable else set()
    result: set[str] = set()
    for c in atom.children:
        result |= get_variables(c)
    return result


def generalize(ground: LinkAtom, positions: list[int]) -> LinkAtom:
    """Replace specific child positions with variables to create a template.

    Usage:
        g = parse_atom("(Inheritance dog animal)")
        t = generalize(g, [0])
        str(t)  →  "(Inheritance $X animal)"

        t2 = generalize(g, [0, 1])
        str(t2) →  "(Inheritance $X $Y)"
    """
    children = list(ground.children)
    for i, pos in enumerate(positions):
        if pos < len(children):
            children[pos] = SymbolAtom(VAR_NAMES[i % len(VAR_NAMES)])
    return LinkAtom(predicate=ground.predicate, children=tuple(children))


def match_template(template: Atom, ground: Atom) -> dict[str, Atom] | None:
    """Try to match a template against a ground atom.

    Returns variable bindings dict if match succeeds, None if it fails.

    Usage:
        t = parse_atom("(Inheritance $X animal)")
        g = parse_atom("(Inheritance dog animal)")
        match_template(t, g)
        # {"$X": SymbolAtom("dog")}

        match_template(t, parse_atom("(Cause rain flood)"))
        # None  (predicate mismatch)

    This is the core of the pattern mining loop in step4_annotate.py:
    for each training example, for each template, if match_template returns
    non-None, the example goes into that template's positive set.
    """
    if isinstance(template, SymbolAtom):
        if template.is_variable:
            return {template.name: ground}
        if isinstance(ground, SymbolAtom) and template.name == ground.name:
            return {}
        return None

    if not isinstance(ground, LinkAtom):
        return None
    if template.predicate != ground.predicate:
        return None
    if len(template.children) != len(ground.children):
        return None

    bindings: dict[str, Atom] = {}
    for tc, gc in zip(template.children, ground.children, strict=False):
        sub = match_template(tc, gc)
        if sub is None:
            return None
        for var, val in sub.items():
            if var in bindings and str(bindings[var]) != str(val):
                return None  # conflicting binding
        bindings.update(sub)
    return bindings


# ── Canonical form for deduplication ─────────────────────────────────────────


def canonical(s: str) -> str:
    """Normalize a MeTTa string for deduplication.
    - collapses extra whitespace
    - lowercase concept symbols (never variables)

    Usage:
        canonical("(Inheritance  Dog  Animal)")
        # "(Inheritance dog animal)"

        canonical("(Inheritance $X animal)")
        # "(Inheritance $X animal)"   ← variables preserved as-is
    """
    try:
        atom = parse_atom(s)
        return _canonical_atom(atom)
    except Exception:
        return s.strip()


def _canonical_atom(atom: Atom) -> str:
    if isinstance(atom, SymbolAtom):
        # variables stay uppercase, concepts go lowercase
        if atom.is_variable:
            return atom.name
        return atom.name.lower()
    children_strs = [_canonical_atom(c) for c in atom.children]
    return f"({atom.predicate} {' '.join(children_strs)})"
