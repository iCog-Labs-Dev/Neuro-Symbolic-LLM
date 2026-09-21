"""Tests for structured semantic distillation datasets."""

from __future__ import annotations

import json
from typing import Any

import pytest

from parser.semantic import (
    CallableBackend,
    ReferenceSemanticParser,
    SemanticDatasetBuilder,
    SemanticParserConfig,
)
from parser.semantic.dataset import split_pairs, verify_split_overlap


def assertion(
    *,
    predicate: str,
    values: tuple[str, ...],
    roles: tuple[str, ...],
) -> dict[str, Any]:
    """Return one valid structured assertion."""
    return {
        "predicate": predicate,
        "relation": None,
        "arguments": [
            {"value": value, "role": role, "type": None}
            for value, role in zip(values, roles, strict=True)
        ],
        "fallback": False,
        "polarity": "positive",
        "confidence": 0.99,
        "source_span": "supporting text",
        "alternatives": [],
    }


def make_parser() -> ReferenceSemanticParser:
    outputs = {
        "A dog is an animal.": {
            "assertions": [
                assertion(
                    predicate="Inheritance",
                    values=("dog", "animal"),
                    roles=("instance", "class"),
                )
            ]
        },
        "A dog can bark.": {
            "assertions": [
                assertion(
                    predicate="Inheritance",
                    values=("dog", "animal"),
                    roles=("instance", "class"),
                ),
                assertion(
                    predicate="CanDo",
                    values=("dog", "bark"),
                    roles=("agent", "action"),
                ),
            ]
        },
    }

    def generate(*, prompt: str, model: str) -> str:
        del model
        sentence = (
            prompt.rsplit("<sentence>\n", maxsplit=1)[-1]
            .split("\n</sentence>", maxsplit=1)[0]
            .strip()
        )
        return json.dumps(outputs.get(sentence, {"assertions": []}))

    return ReferenceSemanticParser(
        backend=CallableBackend(generate, provider_name="fake-teacher"),
        config=SemanticParserConfig(
            model_name="teacher-model",
            prompt_version="2.0.0",
        ),
    )


class TestSemanticDatasetBuilder:
    def test_separates_accepted_and_rejected_examples(self) -> None:
        builder = SemanticDatasetBuilder(make_parser())

        accepted, rejected = builder.generate(
            ["A dog is an animal.", "Unknown statement.", "   "]
        )

        assert len(accepted) == 1
        assert accepted[0].target["assertions"][0]["predicate"] == "Inheritance"
        assert accepted[0].metta is None
        assert accepted[0].teacher_provider == "fake-teacher"
        assert accepted[0].prompt_version == "2.0.0"
        assert len(rejected) == 2
        assert rejected[0].text == "Unknown statement."
        assert "invalid structured semantic output" in rejected[0].error
        assert rejected[1].text == "   "
        assert rejected[1].error == "The sentence cannot be empty"

    def test_normalizes_text_and_keeps_all_structured_assertions(self) -> None:
        builder = SemanticDatasetBuilder(make_parser())

        accepted, rejected = builder.generate(["  A dog can bark.  "])

        assert rejected == []
        assert accepted[0].text == "A dog can bark."
        assert [item["predicate"] for item in accepted[0].target["assertions"]] == [
            "Inheritance",
            "CanDo",
        ]
        assert accepted[0].teacher_model == "teacher-model"

    def test_optionally_includes_derived_metta(self) -> None:
        builder = SemanticDatasetBuilder(make_parser(), include_metta=True)

        accepted, rejected = builder.generate(["A dog can bark."])

        assert rejected == []
        assert accepted[0].metta == (
            "(Inheritance dog animal)",
            "(CanDo dog bark)",
        )

    def test_writes_nested_json_target_without_metta_by_default(
        self,
        tmp_path: Any,
    ) -> None:
        builder = SemanticDatasetBuilder(make_parser())
        accepted, _ = builder.generate(["A dog is an animal."])
        output_path = tmp_path / "dataset.jsonl"

        builder.write_jsonl(accepted, output_path)

        record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["text"] == "A dog is an animal."
        assert record["target"]["assertions"][0]["predicate"] == "Inheritance"
        assert "metta" not in record
        assert record["teacher_model"] == "teacher-model"

    def test_writes_an_empty_file_and_creates_parent_directories(
        self,
        tmp_path: Any,
    ) -> None:
        output_path = tmp_path / "nested" / "empty.jsonl"

        SemanticDatasetBuilder.write_jsonl([], output_path)

        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == ""

    def test_writes_rejected_records(self, tmp_path: Any) -> None:
        builder = SemanticDatasetBuilder(make_parser())
        _, rejected = builder.generate(["Unknown statement."])
        output_path = tmp_path / "rejected.jsonl"

        builder.write_rejected_jsonl(rejected, output_path)

        record = json.loads(output_path.read_text(encoding="utf-8"))
        assert record["text"] == "Unknown statement."
        assert "invalid structured semantic output" in record["error"]


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
    [((0.8, 0.2, 0), [2, 1, 0]), ((0, 0.5, 0.5), [0, 2, 1]), ((1, 0, 0), [3, 0, 0])],
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


def test_conversion_skips_malformed_shapes_and_keeps_valid_record(tmp_path):
    from parser.semantic.dataset import structured_to_pairs

    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    target = {
        "assertions": [
            assertion(
                predicate="Has", values=("dog", "fur"), roles=("owner", "possessed")
            )
        ]
    }
    records = [
        None,
        [],
        42,
        {"text": None},
        {"text": []},
        {"text": "Dog", "target": target},
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    pairs = structured_to_pairs(str(source), str(output))
    assert len(pairs) == 1
    assert pairs[0]["input"] == "Dog"
