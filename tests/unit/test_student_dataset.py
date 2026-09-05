"""Student batches preserve complete targets and mask only prompt/padding."""

import torch
from torch.utils.data import DataLoader

from parser.semantic.dataset import StudentBatchCollator, StudentDataset
from parser.semantic.student_prompt import build_student_prompt


class CharacterTokenizer:
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) + 2 for char in text]}


def test_variable_length_batch_preserves_targets():
    tokenizer = CharacterTokenizer()
    pairs = [
        {"input": "Dog.", "label": '{"assertions": []}'},
        {"input": "A longer sentence.", "label": '{"assertions": [1, 2]}'},
    ]
    dataset = StudentDataset(pairs, tokenizer, max_length=512)
    batch = next(
        iter(DataLoader(dataset, batch_size=2, collate_fn=StudentBatchCollator(0)))
    )
    assert batch["input_ids"].shape[0] == 2
    for index, pair in enumerate(pairs):
        prompt_length = len(build_student_prompt(pair["input"]))
        expected = tokenizer(pair["label"])["input_ids"] + [1]
        labels = batch["labels"][index]
        assert labels[labels != -100].tolist() == expected
        assert torch.all(labels[:prompt_length] == -100)
        length = len(dataset[index]["input_ids"])
        assert torch.all(labels[length:] == -100)
        assert torch.all(batch["attention_mask"][index, length:] == 0)
        assert torch.all(batch["input_ids"][index, length:] == 0)


def test_oversized_target_is_rejected_instead_of_truncated():
    tokenizer = CharacterTokenizer()
    pair = {"input": "Dog.", "label": '{"assertions": []}'}
    length = len(build_student_prompt(pair["input"])) + len(pair["label"]) + 1
    assert len(StudentDataset([pair], tokenizer, max_length=length)) == 1
    dataset = StudentDataset([pair], tokenizer, max_length=length - 1)
    assert len(dataset) == 0
    assert dataset.oversized == 1
