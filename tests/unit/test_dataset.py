"""Teacher dataset generation, student-pair conversion, and split isolation."""

import json

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import (
    CallableBackend,
    ReferenceSemanticParser,
    SemanticDatasetBuilder,
    SemanticParserConfig,
)
from parser.semantic.catalog_student import validate_catalog_student_pair
from parser.semantic.dataset import (
    split_pairs,
    structured_to_pairs,
    verify_split_overlap,
)
from tests.semantic_fixtures import result as fixture_result
from tests.semantic_fixtures import target


def make_parser(payload=None):
    def generate(*, prompt, model):
        text = prompt.rsplit("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]
        return json.dumps(payload if payload is not None else target(text))

    return ReferenceSemanticParser(
        backend=CallableBackend(generate, provider_name="fixture"),
        config=SemanticParserConfig(model_name="fixture"),
    )


def test_builder_preserves_exact_source_and_writes_parse_results(tmp_path):
    builder = SemanticDatasetBuilder(make_parser())
    accepted, rejected = builder.generate(["  Mira left.  ", "  "])
    assert len(accepted) == len(rejected) == 1
    assert accepted[0].result.document.source_text == "  Mira left.  "
    assert accepted[0].result.contract_version == "2.0.0"
    path = tmp_path / "nested" / "documents.jsonl"
    builder.write_jsonl(accepted, path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["document"]["source_text"] == "  Mira left.  "
    assert record["extraction"]["assertions"]
    assert "target" not in record
    assert record["provenance"]["provider"] == "fixture"
    rejected_path = tmp_path / "rejected.jsonl"
    builder.write_rejected_jsonl(rejected, rejected_path)
    assert json.loads(rejected_path.read_text(encoding="utf-8"))["text"] == "  "


@pytest.mark.parametrize(
    "factuality", ["asserted", "hypothetical", "opinion", "speculative"]
)
def test_builder_retains_each_factuality_without_asserted_only_filter(factuality):
    payload = target("Mira left.")
    payload["assertions"][0]["factuality"] = factuality
    documents, rejected = SemanticDatasetBuilder(make_parser(payload)).generate(
        ["Mira left."]
    )
    assert not rejected
    assert documents[0].result.extraction.assertions[0].factuality == factuality


@pytest.mark.parametrize("option", ["include_metta", "asserted_only"])
def test_dataset_options_only_affect_derived_views(option):
    records, rejected = SemanticDatasetBuilder(
        make_parser(), **{option: True}
    ).generate(["Mira left."])
    assert not rejected
    assert records[0].result.extraction.assertions[0].factuality == "asserted"
    assert records[0].metta


def test_builder_preserves_metadata_through_later_pair_conversion(tmp_path):
    builder = SemanticDatasetBuilder(make_parser())
    records, rejected = builder.generate(
        ["Mira left."], source_dataset="pilot", source_domain="narrative"
    )
    assert not rejected
    assert records[0].source_domain == "narrative"
    source = tmp_path / "documents.jsonl"
    builder.write_jsonl(records, source)
    pairs = structured_to_pairs(
        str(source), str(tmp_path / "pairs.jsonl"), include_confidence=True
    )
    assert pairs[0]["source_dataset"] == "pilot"
    assert pairs[0]["source_domain"] == "narrative"
    assert pairs[0]["confidence"] == (
        records[0].result.extraction.assertions[0].confidence
    )


def test_invalid_model_output_is_quarantined():
    documents, rejected = SemanticDatasetBuilder(
        make_parser({"assertions": []})
    ).generate(["Mira left."])
    assert not documents
    assert rejected[0].error


def test_empty_export_creates_parent(tmp_path):
    path = tmp_path / "nested" / "empty.jsonl"
    SemanticDatasetBuilder.write_jsonl([], path)
    assert path.read_text(encoding="utf-8") == ""
    assert path.with_name("empty.metadata.jsonl").read_text(encoding="utf-8") == ""


def test_parse_result_to_student_pair_roundtrips_exactly(tmp_path):
    catalog = load_ontology_catalog()
    original = fixture_result(catalog=catalog)
    source, output = tmp_path / "docs.jsonl", tmp_path / "pairs.jsonl"
    source.write_text(original.model_dump_json() + "\n", encoding="utf-8")
    pairs = structured_to_pairs(str(source), str(output))
    _, _, restored = validate_catalog_student_pair(pairs[0], catalog=catalog)
    assert restored == original


@pytest.mark.parametrize(
    "record", [None, [], {"text": "Dog", "target": {"assertions": []}}]
)
def test_non_parse_records_do_not_silently_become_labels(tmp_path, record):
    source, output = tmp_path / "source.jsonl", tmp_path / "pairs.jsonl"
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid document"):
        structured_to_pairs(str(source), str(output))
    assert not output.exists()


@pytest.mark.parametrize(
    "ratios",
    [
        (-0.1, 0.5, 0.6),
        (1.1, 0, -0.1),
        (float("nan"), 0, 1),
        (float("inf"), 0, 0),
        (0.5, 0.1, 0.1),
    ],
)
def test_split_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        split_pairs([], *ratios)


def test_split_deduplicates_without_mutating_and_is_reproducible():
    pairs = [{"input": str(i), "label": "{}"} for i in range(20)]
    pairs.extend([{"input": " 0 ", "label": "{}"}, {"input": " "}])
    original = list(pairs)

    splits = split_pairs(pairs)
    assert [len(split) for split in splits] == [16, 2, 2]
    assert splits == split_pairs(pairs)
    assert splits != split_pairs(pairs, seed=17)
    assert pairs == original

    verify_split_overlap(*splits)

    assert {pair["input"] for split in splits for pair in split} == {
        str(i) for i in range(20)
    }


@pytest.mark.parametrize(
    "ratios, expected",
    [
        ((0.8, 0.2, 0), [2, 1, 0]),
        ((0, 0.5, 0.5), [0, 2, 1]),
        ((1, 0, 0), [3, 0, 0]),
    ],
)
def test_split_rounding_keeps_zero_ratio_splits_empty(ratios, expected):
    pairs = [{"input": str(i)} for i in range(3)]
    assert [len(split) for split in split_pairs(pairs, *ratios)] == expected


@pytest.mark.parametrize("pair", [None, [], {"input": None}, {"input": 1}])
def test_split_rejects_malformed_pairs(pair):
    with pytest.raises(ValueError, match="Pair 1"):
        split_pairs([pair])


def test_split_empty_input():
    assert split_pairs([]) == ([], [], [])


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_conversion_preserves_source_when_output_aliases_it(tmp_path, alias):
    from parser.semantic.dataset import structured_to_pairs

    source = tmp_path / "source.jsonl"
    source.write_text('{"text": "Dog"}\n', encoding="utf-8")
    output = source if alias == "same" else tmp_path / "output.jsonl"

    if alias == "symlink":
        output.symlink_to(source)

    elif alias == "hardlink":
        output.hardlink_to(source)

    original = source.read_bytes()
    with pytest.raises(ValueError, match="paths must differ"):
        structured_to_pairs(str(source), str(output))
    assert source.read_bytes() == original
