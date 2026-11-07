"""Student batches preserve complete targets and mask only prompt/padding."""

import pytest
from torch.utils.data import DataLoader

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_student import validate_catalog_student_pair
from parser.semantic.dataset import StudentBatchCollator, StudentDataset
from parser.student.student_prompt import build_student_prompt
from tests.semantic_fixtures import pair as canonical_pair


class CharacterTokenizer:
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) + 2 for char in text]}


def test_variable_length_batch_preserves_targets():
    tokenizer = CharacterTokenizer()
    pairs = [
        canonical_pair("Mira left."),
        canonical_pair("Mira departed earlier."),
    ]
    max_length = max(
        len(build_student_prompt(pair["input"]))
        + len(validate_catalog_student_pair(pair, catalog=load_ontology_catalog())[1])
        + 1
        for pair in pairs
    )
    dataset = StudentDataset(pairs, tokenizer, max_length=max_length)
    batch = next(
        iter(DataLoader(dataset, batch_size=2, collate_fn=StudentBatchCollator(0)))
    )
    assert batch["input_ids"].shape[0] == 2
    for index, pair in enumerate(pairs):
        prompt_length = len(build_student_prompt(pair["input"]))
        _, label, _ = validate_catalog_student_pair(
            pair, catalog=load_ontology_catalog()
        )
        expected = tokenizer(label)["input_ids"] + [1]
        labels = batch["labels"][index].tolist()

        actual_targets = [value for value in labels if value != -100]

        assert actual_targets == expected

        assert all(value == -100 for value in labels[:prompt_length])

        length = len(dataset[index]["input_ids"])

        assert all(value == -100 for value in labels[length:])

        attention_mask = batch["attention_mask"][index].tolist()
        input_ids = batch["input_ids"][index].tolist()

        assert all(value == 0 for value in attention_mask[length:])
        assert all(value == 0 for value in input_ids[length:])


def test_oversized_target_is_rejected_instead_of_truncated():
    tokenizer = CharacterTokenizer()
    pair = canonical_pair()
    _, label, _ = validate_catalog_student_pair(pair, catalog=load_ontology_catalog())
    length = len(build_student_prompt(pair["input"])) + len(label) + 1
    assert len(StudentDataset([pair], tokenizer, max_length=length)) == 1
    with pytest.raises(ValueError, match="never truncated"):
        StudentDataset([pair], tokenizer, max_length=length - 1)


def test_automatic_length_measures_complete_examples():
    records = [canonical_pair("Mira left."), canonical_pair("Mira left much earlier.")]
    dataset = StudentDataset(records, CharacterTokenizer())
    assert len(dataset) == 2
    assert dataset.required_max_length == max(
        len(dataset[index]["input_ids"]) for index in range(len(dataset))
    )


@pytest.mark.parametrize("invalid", [None, [], 42, {"input": "Dog", "label": "{}"}])
def test_malformed_or_historical_record_is_rejected(invalid):
    with pytest.raises(ValueError):
        StudentDataset([invalid], CharacterTokenizer())
