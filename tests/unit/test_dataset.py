"""tests/unit/test_dataset.py

Run with: python -m pytest tests/unit/test_dataset.py -v
"""

from __future__ import annotations

import json

from parser.semantic import (
    CallableBackend,
    ReferenceSemanticParser,
    SemanticDatasetBuilder,
    SemanticParserConfig,
)


def make_parser():
    outputs = {
        "A dog is an animal.": "(Inheritance dog animal)",
        "Unknown statement.": "UNSUPPORTED",
        "A dog can bark.": "(Inheritance dog animal) (CanDo dog bark)",
    }

    def generate(*, prompt: str, model: str) -> str:
        del model
        sentence = (
            prompt.rsplit("<sentence>\n", maxsplit=1)[-1]
            .split("\n</sentence>", maxsplit=1)[0]
            .strip()
        )
        return outputs.get(sentence, "UNSUPPORTED")

    return ReferenceSemanticParser(
        backend=CallableBackend(
            generate,
            provider_name="fake-teacher",
        ),
        config=SemanticParserConfig(
            model_name="teacher-model",
            prompt_version="1.2.0",
        ),
    )


# ── Dataset generation ─────────────────────────────────────────────────────────


class TestSemanticDatasetBuilder:
    def test_separates_accepted_and_rejected_examples(self):
        generator = SemanticDatasetBuilder(make_parser())

        accepted, rejected = generator.generate(
            [
                "A dog is an animal.",
                "Unknown statement.",
                "   ",
            ]
        )

        assert len(accepted) == 1
        assert accepted[0].metta == "(Inheritance dog animal)"
        assert accepted[0].teacher_provider == "fake-teacher"
        assert accepted[0].prompt_version == "1.2.0"

        assert len(rejected) == 2
        assert rejected[0].text == "Unknown statement."
        assert rejected[0].error == "Unsupported sentence: 'Unknown statement.'"
        assert rejected[1].text == "   "
        assert rejected[1].error == "The sentence cannot be empty"

    def test_normalizes_accepted_text_and_keeps_all_returned_atoms(self):
        generator = SemanticDatasetBuilder(make_parser())

        accepted, rejected = generator.generate(["  A dog can bark.  "])

        assert rejected == []
        assert accepted[0].text == "A dog can bark."
        assert accepted[0].metta == ("(Inheritance dog animal)\n(CanDo dog bark)")
        assert accepted[0].teacher_model == "teacher-model"

    def test_writes_jsonl(self, tmp_path):
        generator = SemanticDatasetBuilder(make_parser())
        accepted, _ = generator.generate(["A dog is an animal."])

        output_path = tmp_path / "dataset.jsonl"
        generator.write_jsonl(accepted, output_path)

        lines = output_path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])

        assert record["text"] == "A dog is an animal."
        assert record["metta"] == "(Inheritance dog animal)"
        assert record["teacher_model"] == "teacher-model"

    def test_writes_an_empty_file_and_creates_parent_directories(self, tmp_path):
        output_path = tmp_path / "nested" / "empty.jsonl"

        SemanticDatasetBuilder.write_jsonl([], output_path)

        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == ""
