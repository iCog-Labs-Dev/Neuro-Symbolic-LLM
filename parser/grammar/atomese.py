"""parser/grammar/atomese.py

The complete MeTTa Atomese data model.

Predicate and arity definitions are loaded from the shared semantic predicate
schema. This module remains responsible for parsing and structural validation.

What lives here:
  - Atom data classes (SymbolAtom, LinkAtom)
  - parse_atom():          string → Atom tree
  - atom_to_string():      Atom tree → string
  - validate_metta_string(): validate the complete Atom tree recursively
  - match_template():      does this ground atom match this template?
  - generalize():          produce a template from a ground atom
  - canonical():           normalize a string for deduplication

The closed vocabulary and its structural arities live in
``configs/parser_config/predicate_schema.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Constants ────────────────────────────────────────────────────────────────

_PREDICATE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "predicate_schema.yaml"
)


def _load_atomese_predicates() -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    """Load the shared predicate vocabulary and its Atomese-level arities."""
    with _PREDICATE_SCHEMA_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Predicate schema must be a YAML mapping")

    semantic = config.get("predicates")
    structural = config.get("structural_predicates")
    if not isinstance(semantic, dict) or not isinstance(structural, dict):
        raise ValueError(
            "Predicate schema must define 'predicates' and "
            "'structural_predicates' mappings"
        )

    duplicates = semantic.keys() & structural.keys()
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"Predicates cannot be both semantic and structural: {names}")

    fixed: dict[str, int] = {}
    variable: dict[str, tuple[int, int]] = {}
    for name, definition in {**semantic, **structural}.items():
        if not isinstance(name, str) or not isinstance(definition, dict):
            raise ValueError("Every Atomese predicate requires a mapping")

        atomese_arity = definition.get("atomese_arity")
        if atomese_arity is None and definition.get("variable_arity") is not True:
            atomese_arity = definition.get("arity")
        if atomese_arity is not None:
            if (
                not isinstance(atomese_arity, int)
                or isinstance(atomese_arity, bool)
                or atomese_arity < 1
            ):
                raise ValueError(f"Invalid Atomese arity for {name!r}")
            fixed[name] = atomese_arity
            continue

        minimum = definition.get("min_arity")
        maximum = definition.get("max_arity")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or minimum < 1
            or maximum < minimum
        ):
            raise ValueError(f"Invalid variable Atomese arity for {name!r}")
        variable[name] = (minimum, maximum)

    return fixed, variable


ARITY, VARIABLE_ARITY = _load_atomese_predicates()
PREDICATES = frozenset(ARITY | VARIABLE_ARITY)

VARIABLE_RE = re.compile(r"^\$[A-Z][A-Z0-9_]*$")
CONCEPT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
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
        return VARIABLE_RE.fullmatch(self.name) is not None


@dataclass(frozen=True)
class LinkAtom:
    """Internal node — a typed link with children.

    Example:
        LinkAtom("Inheritance", (SymbolAtom("dog"), SymbolAtom("animal")))
        → str: "(Inheritance dog animal)"
    """

    predicate: str
    children: tuple[Atom, ...]

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
            if depth < 0:
                raise ValueError("Unexpected closing parenthesis")
            current.append(ch)
            if depth == 0:
                t = "".join(current).strip()
                if t:
                    tokens.append(t)
                current = []
        elif ch.isspace() and depth == 0:
            t = "".join(current).strip()
            if t:
                tokens.append(t)
            current = []
        else:
            current.append(ch)

    if depth != 0:
        raise ValueError("Unbalanced parentheses")

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
    """Check whether a string is valid for the configured Atomese vocabulary.

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

    error = _validate_atom(atom)
    return (False, error) if error else (True, "")


def _validate_atom(atom: Atom) -> str:
    """Return the first validation error in an Atom tree, or an empty string."""
    if isinstance(atom, SymbolAtom):
        if VARIABLE_RE.fullmatch(atom.name) or CONCEPT_RE.fullmatch(atom.name):
            return ""
        return f"Invalid symbol '{atom.name}'"

    if atom.predicate not in PREDICATES:
        return (
            f"Unknown predicate '{atom.predicate}'. "
            f"Must be one of: {sorted(PREDICATES)}"
        )

    child_count = len(atom.children)
    if atom.predicate in ARITY:
        expected = ARITY[atom.predicate]
        if child_count != expected:
            return (
                f"'{atom.predicate}' requires exactly {expected} children, "
                f"got {child_count}"
            )
    else:
        minimum, maximum = VARIABLE_ARITY[atom.predicate]
        if not minimum <= child_count <= maximum:
            return (
                f"'{atom.predicate}' requires between {minimum} and {maximum} "
                f"children, got {child_count}"
            )

    if atom.predicate == "Evaluation" and (
        not isinstance(atom.children[1], LinkAtom)
        or atom.children[1].predicate != "List"
    ):
        return "'Evaluation' requires a List link as its second child"

    if atom.predicate == "Not" and not isinstance(atom.children[0], LinkAtom):
        return "'Not' requires a link as its child"

    for child in atom.children:
        error = _validate_atom(child)
        if error:
            return error
    return ""


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
