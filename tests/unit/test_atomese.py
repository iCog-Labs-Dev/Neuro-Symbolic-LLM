"""tests/unit/test_atomese.py

Run with: python -m pytest tests/unit/test_atomese.py -v

All tests must pass before writing any other symbolic code.
These tests define exactly what the grammar does and does not accept.
"""

from pathlib import Path

import pytest

from parser.grammar import atomese
from parser.grammar.atomese import (
    LinkAtom,
    SymbolAtom,
    atom_to_string,
    canonical,
    generalize,
    is_template,
    match_template,
    parse_atom,
    validate_metta_string,
)


def test_rejects_predicate_defined_as_semantic_and_structural(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema_path = tmp_path / "predicate_schema.yaml"
    schema_path.write_text(
        """predicates:
  Not:
    arity: 1
structural_predicates:
  Not:
    arity: 1
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(atomese, "_PREDICATE_SCHEMA_PATH", schema_path)

    with pytest.raises(
        ValueError,
        match="Predicates cannot be both semantic and structural: Not",
    ):
        atomese._load_atomese_predicates()


# ── parse_atom ────────────────────────────────────────────────────────────────


class TestParseAtom:
    def test_simple_link(self):
        a = parse_atom("(Inheritance dog animal)")
        assert isinstance(a, LinkAtom)
        assert a.predicate == "Inheritance"
        assert len(a.children) == 2
        assert str(a.children[0]) == "dog"
        assert str(a.children[1]) == "animal"

    def test_nested_link(self):
        a = parse_atom("(Evaluation likes (List john mary))")
        assert isinstance(a, LinkAtom)
        assert a.predicate == "Evaluation"
        assert isinstance(a.children[1], LinkAtom)
        assert a.children[1].predicate == "List"

    def test_symbol_atom(self):
        a = parse_atom("dog")
        assert isinstance(a, SymbolAtom)
        assert a.name == "dog"
        assert not a.is_variable

    def test_variable_atom(self):
        a = parse_atom("$X")
        assert isinstance(a, SymbolAtom)
        assert a.name == "$X"
        assert a.is_variable

    def test_template(self):
        a = parse_atom("(Inheritance $X animal)")
        assert isinstance(a, LinkAtom)
        assert isinstance(a.children[0], SymbolAtom)
        assert a.children[0].is_variable

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_atom("")

    def test_unbalanced_raises(self):
        with pytest.raises(ValueError):
            parse_atom("(Inheritance dog animal")

    @pytest.mark.parametrize(
        "expression",
        ["(Has dog fur))", "((Has dog fur)"],
    )
    def test_rejects_parentheses_that_close_at_the_wrong_depth(self, expression):
        with pytest.raises(ValueError, match="parenthes"):
            parse_atom(expression)

    def test_all_predicates_parse(self):
        cases = [
            "(Inheritance dog animal)",
            "(Evaluation likes (List john mary))",
            "(CanDo bird fly)",
            "(On cup table)",
            "(Cause rain flood)",
            "(Has dog fur)",
            "(PartOf wheel car)",
            "(StateOf world stage)",
            "(LocatedIn book shelf)",
            "(MemberOf alice team)",
            "(UsedFor knife cutting)",
            "(Before dawn sunrise)",
            "(After sunrise dawn)",
        ]
        for expr in cases:
            a = parse_atom(expr)
            assert isinstance(a, LinkAtom), f"Failed to parse: {expr}"

    def test_roundtrip(self):
        cases = [
            "(Inheritance dog animal)",
            "(Evaluation likes (List john mary))",
            "(Cause rain flood)",
            "(Inheritance $X animal)",
            "(Evaluation $V (List $X $Y))",
        ]
        for expr in cases:
            assert atom_to_string(parse_atom(expr)) == expr


# ── validate_metta_string ─────────────────────────────────────────────────────


class TestValidate:
    def test_valid_inheritance(self):
        ok, err = validate_metta_string("(Inheritance dog animal)")
        assert ok, err

    def test_valid_evaluation(self):
        ok, err = validate_metta_string("(Evaluation likes (List john mary))")
        assert ok, err

    def test_valid_template(self):
        ok, err = validate_metta_string("(Inheritance $X animal)")
        assert ok, err

    def test_invalid_predicate(self):
        ok, err = validate_metta_string("(UnknownPred x y)")
        assert not ok
        assert "Unknown predicate" in err

    def test_wrong_arity(self):
        ok, err = validate_metta_string("(Inheritance dog animal extra)")
        assert not ok
        assert "requires exactly 2" in err

    def test_bare_symbol_invalid(self):
        ok, err = validate_metta_string("dog")
        assert not ok
        assert "must be a link" in err

    def test_empty_invalid(self):
        ok, err = validate_metta_string("")
        assert not ok

    def test_all_valid_predicates(self):
        cases = [
            "(Inheritance dog animal)",
            "(CanDo bird fly)",
            "(On cup table)",
            "(Cause rain flood)",
            "(Has dog fur)",
            "(PartOf wheel car)",
            "(StateOf world stage)",
            "(LocatedIn book shelf)",
            "(MemberOf alice team)",
            "(UsedFor knife cutting)",
            "(Before dawn sunrise)",
            "(After sunrise dawn)",
        ]
        for expr in cases:
            ok, err = validate_metta_string(expr)
            assert ok, f"Should be valid: {expr}  error: {err}"

    @pytest.mark.parametrize(
        "predicate",
        ["LocatedIn", "MemberOf", "UsedFor", "Before", "After"],
    )
    def test_added_predicates_require_two_children(self, predicate):
        ok, err = validate_metta_string(f"({predicate} subject object extra)")

        assert not ok
        assert "requires exactly 2" in err

    def test_rejects_unknown_nested_predicate(self):
        ok, err = validate_metta_string("(Has dog (Unknown x))")

        assert not ok
        assert "Unknown predicate 'Unknown'" in err

    def test_evaluation_requires_list_as_second_child(self):
        ok, err = validate_metta_string("(Evaluation likes john)")

        assert not ok
        assert "requires a List link" in err

    def test_list_rejects_more_than_four_children(self):
        ok, err = validate_metta_string("(List one two three four five)")

        assert not ok
        assert "between 1 and 4" in err

    @pytest.mark.parametrize(
        "expression",
        [
            "(Has dog invalid-symbol!)",
            "(Has dog $lowercase)",
            "(Has dog $)",
        ],
    )
    def test_rejects_invalid_symbols(self, expression):
        ok, err = validate_metta_string(expression)

        assert not ok
        assert "Invalid symbol" in err

    def test_accepts_case_preserved_entity_symbols(self):
        ok, err = validate_metta_string("(Has Ben Car)")

        assert ok, err

    def test_not_requires_a_link_child(self):
        ok, err = validate_metta_string("(Not dog)")

        assert not ok
        assert "requires a link" in err


# ── match_template ────────────────────────────────────────────────────────────


class TestMatchTemplate:
    def test_exact_match(self):
        t = parse_atom("(Inheritance dog animal)")
        g = parse_atom("(Inheritance dog animal)")
        result = match_template(t, g)
        assert result == {}

    def test_variable_match(self):
        t = parse_atom("(Inheritance $X animal)")
        g = parse_atom("(Inheritance dog animal)")
        result = match_template(t, g)
        assert result is not None
        assert "$X" in result
        assert str(result["$X"]) == "dog"

    def test_two_variable_match(self):
        t = parse_atom("(Cause $X $Y)")
        g = parse_atom("(Cause rain flood)")
        result = match_template(t, g)
        assert result is not None
        assert str(result["$X"]) == "rain"
        assert str(result["$Y"]) == "flood"

    def test_predicate_mismatch(self):
        t = parse_atom("(Inheritance $X animal)")
        g = parse_atom("(Cause rain flood)")
        assert match_template(t, g) is None

    def test_fixed_child_mismatch(self):
        t = parse_atom("(Inheritance $X animal)")
        g = parse_atom("(Inheritance dog plant)")
        assert match_template(t, g) is None

    def test_nested_template(self):
        t = parse_atom("(Evaluation $V (List $X $Y))")
        g = parse_atom("(Evaluation likes (List john mary))")
        result = match_template(t, g)
        assert result is not None
        assert str(result["$V"]) == "likes"
        assert str(result["$X"]) == "john"

    def test_conflicting_bindings(self):
        # $X cannot bind to both "dog" and "animal"
        t = parse_atom("(Inheritance $X $X)")
        g = parse_atom("(Inheritance dog animal)")
        assert match_template(t, g) is None

    def test_same_variable_consistent(self):
        t = parse_atom("(Inheritance $X $X)")
        g = parse_atom("(Inheritance dog dog)")
        result = match_template(t, g)
        assert result is not None
        assert str(result["$X"]) == "dog"


# ── generalize ────────────────────────────────────────────────────────────────


class TestGeneralize:
    def test_generalize_first(self):
        g = parse_atom("(Inheritance dog animal)")
        t = generalize(g, [0])
        assert str(t) == "(Inheritance $X animal)"

    def test_generalize_second(self):
        g = parse_atom("(Inheritance dog animal)")
        t = generalize(g, [1])
        assert str(t) == "(Inheritance dog $X)"

    def test_generalize_both(self):
        g = parse_atom("(Inheritance dog animal)")
        t = generalize(g, [0, 1])
        assert str(t) == "(Inheritance $X $Y)"

    def test_generalized_is_template(self):
        g = parse_atom("(Cause rain flood)")
        t = generalize(g, [0])
        assert is_template(t)


# ── canonical ─────────────────────────────────────────────────────────────────


class TestCanonical:
    def test_lowercases_concepts(self):
        assert canonical("(Inheritance Dog Animal)") == "(Inheritance dog animal)"

    def test_preserves_variables(self):
        assert canonical("(Inheritance $X Animal)") == "(Inheritance $X animal)"

    def test_collapses_whitespace(self):
        assert canonical("(Inheritance  dog  animal)") == "(Inheritance dog animal)"

    def test_malformed_returns_stripped(self):
        result = canonical("not valid metta")
        assert result.strip() == "not valid metta"
