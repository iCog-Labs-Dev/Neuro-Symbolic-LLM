"""Causal completeness and split isolation for A4 reference probes."""

import pytest

from scripts.prepare_retrieval_probes import (
    causal_probes,
    held_out_documents,
    possible_clauses,
)


def document(name, text):
    return {"corpus_id": "reference", "document_id": name, "source_text": text}


def test_exclusion_uses_identity_text_and_bank_provenance():
    rows = [
        document("a", "seen"),
        document("b", "seen"),
        document("c", "bank"),
        document("d", "new"),
    ]
    assert held_out_documents(rows, [rows[0]], {"c"}) == [rows[3]]


def test_empty_holdout_and_duplicate_text_are_rejected():
    with pytest.raises(ValueError, match="No source-disjoint"):
        held_out_documents([document("a", "seen")], [document("b", "seen")], set())
    with pytest.raises(ValueError, match="duplicate"):
        held_out_documents([document("a", "same"), document("b", "same")], [], set())


def test_prefilter_never_requires_variables_to_be_different():
    assert possible_clauses([("P", "$a", "$b")], [("P", "same", "same")])
    assert not possible_clauses([("P", "missing")], [("P", "present")])
    assert not possible_clauses([("P", "$a")], [("P", "one", "two")])


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def binding(start, end, text):
    return {
        "evidence": [{"source_spans": [{"start": start, "end": end, "text": text}]}]
    }


def test_complete_multilabel_sets_only_use_visible_evidence():
    row = {
        **document("d", "ab.cd.ef."),
        "assertions": [
            {"source_spans": [{"start": i, "end": i + 3}]} for i in (0, 3, 6)
        ],
    }
    labels = [
        ("early", binding(0, 3, "ab.")),
        ("later", binding(3, 6, "cd.")),
        ("same_endpoint", binding(3, 6, "cd.")),
    ]
    probes = causal_probes(row, CharacterTokenizer(), labels, 6)
    assert probes[0]["input_ids"] == [0, 1, 2]
    assert probes[0]["relevant_ids"] == ["early"]
    assert probes[1]["relevant_ids"] == ["early", "later", "same_endpoint"]
    assert probes[2]["input_ids"] == [3, 4, 5, 6, 7, 8]
    assert probes[2]["relevant_ids"] == ["later", "same_endpoint"]
    assert probes[2]["token_index"] == 5


def test_source_mismatches_do_not_become_labels():
    row = {
        **document("d", "ab."),
        "assertions": [{"source_spans": [{"start": 0, "end": 3}]}],
    }
    with pytest.raises(ValueError, match="differs"):
        causal_probes(row, CharacterTokenizer(), [("a", binding(0, 3, "xx."))], 128)
