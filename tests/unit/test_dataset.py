"""Tests for structured semantic distillation datasets."""

from __future__ import annotations

import json
from typing import Any

from parser.semantic import (
    CallableBackend,
    ReferenceSemanticParser,
    SemanticDatasetBuilder,
    SemanticParserConfig,
)


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
