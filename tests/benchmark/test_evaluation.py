from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.evaluation import aggregate_lm_rows, evaluate_long_range
from runtime import data_loader

ROOT = Path(__file__).resolve().parents[2]


def test_lm_aggregation_is_token_weighted() -> None:
    result = aggregate_lm_rows(
        [
            {
                "tokens": 1,
                "loss_sum": 1.0,
                "base_loss_sum": 2.0,
                "kl_sum": 0.1,
                "correct": 1,
            },
            {
                "tokens": 3,
                "loss_sum": 9.0,
                "base_loss_sum": 12.0,
                "kl_sum": 0.9,
                "correct": 0,
            },
        ]
    )

    expected = {
        "tokens": 4,
        "loss": 2.5,
        "base_loss": 3.5,
        "perplexity": math.exp(2.5),
        "base_perplexity": math.exp(3.5),
        "kl": 0.25,
        "token_accuracy": 0.25,
    }
    assert result["status"] == "measured"
    assert {key: result[key] for key in expected} == pytest.approx(expected)


def test_lm_aggregation_reports_no_evaluated_tokens() -> None:
    assert aggregate_lm_rows([]) == {"status": "insufficient_data", "tokens": 0}
    assert aggregate_lm_rows(
        [
            {
                "tokens": 0,
                "loss_sum": 0.0,
                "base_loss_sum": 0.0,
                "kl_sum": 0.0,
                "correct": 0,
            }
        ]
    ) == {"status": "insufficient_data", "tokens": 0}


@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_lm_aggregation_rejects_invalid_token_counts(tokens: Any) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        aggregate_lm_rows(
            [
                {
                    "tokens": tokens,
                    "loss_sum": 0.0,
                    "base_loss_sum": 0.0,
                    "kl_sum": 0.0,
                    "correct": 0,
                }
            ]
        )


def _long_range_item(**overrides: Any) -> dict[str, Any]:
    item = {
        "id": "near",
        "family": "relation",
        "distance": "near",
        "context": "context",
        "question": " question",
        "choices": [" correct", " wrong"],
        "answer": 0,
    }
    item.update(overrides)
    return item


def test_long_range_reports_groups_and_expands_inline_filler() -> None:
    prompts: list[str] = []
    config = {
        "schema": "long-range/v1",
        "items": [
            _long_range_item(),
            _long_range_item(
                id="far",
                distance="far",
                filler=" filler",
                filler_repeat=2,
            ),
        ],
    }

    def score(prompt: str, choice: str) -> float:
        prompts.append(prompt)
        if "filler" in prompt:
            return 1.0 if choice == " wrong" else 0.0
        return 1.0 if choice == " correct" else 0.0

    result = evaluate_long_range(config, score)

    assert result["accuracy"] == 0.5
    assert result["by_family"]["relation"]["accuracy"] == 0.5
    assert result["by_distance"] == {"far": 0.0, "near": 1.0}
    assert result["near_far_gap"] == 1.0
    assert prompts[-1] == "context filler filler question"
    assert result["items"][1]["predicted"] == 1


def test_long_range_rejects_invalid_answers_and_scores() -> None:
    with pytest.raises(ValueError, match="out of range"):
        evaluate_long_range(
            {"schema": "long-range/v1", "items": [_long_range_item(answer=2)]},
            lambda _prompt, _choice: 0.0,
        )

    with pytest.raises(ValueError, match="finite number"):
        evaluate_long_range(
            {"schema": "long-range/v1", "items": [_long_range_item()]},
            lambda _prompt, _choice: float("nan"),
        )


def test_shipped_runtime_config_keeps_external_lm_protocols() -> None:
    config = data_loader.load_runtime_config(ROOT / "configs/runtime_datasets.yaml")

    assert set(config["evaluation"]) == {"anchor", "long_range"}
    assert config["evaluation"]["anchor"]["name"] == "general_language_anchor"
    assert len(config["evaluation"]["anchor"]["texts"]) == 5
    assert config["evaluation"]["long_range"]["schema"] == "long-range/v1"
    assert isinstance(config["evaluation"]["long_range"]["items"], list)
    assert all(
        isinstance(dataset.get("revision"), str) and len(dataset["revision"]) == 40
        for dataset in config["datasets"]
    )


class _Dataset:
    column_names = ["body"]

    def __getitem__(self, field: str) -> list[str]:
        assert field == "body"
        return [" first ", "", "second"]


def test_lm_text_loading_forwards_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def load_dataset(name: str, **kwargs: Any) -> _Dataset:
        calls.append((name, kwargs))
        return _Dataset()

    monkeypatch.setattr(data_loader, "load_dataset", load_dataset)
    texts = data_loader.load_split_texts(
        {
            "dataset": "owner/corpus",
            "config": "subset",
            "revision": "a" * 40,
            "text_field": "body",
        },
        "train[:10]",
    )

    assert texts == ["first", "second"]
    assert calls == [
        (
            "owner/corpus",
            {
                "name": "subset",
                "split": "train[:10]",
                "revision": "a" * 40,
                "streaming": False,
            },
        )
    ]


def test_array_fingerprint_captures_layout_independent_values_shape_and_dtype() -> None:
    blocks = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    fingerprint = data_loader.array_fingerprint(blocks)

    assert data_loader.array_fingerprint(np.asfortranarray(blocks)) == fingerprint
    assert data_loader.array_fingerprint(blocks.reshape(1, 4)) != fingerprint
    assert data_loader.array_fingerprint(blocks.astype(np.int64)) != fingerprint


@dataclass(frozen=True)
class _Task:
    name: str
    train_blocks: np.ndarray
    eval_set: tuple[Any, ...]
    epochs: int


def test_runtime_data_applies_task_order_heldout_and_joint_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "runtime:\n"
        "  source: separate\n"
        "  task_order: [second, first]\n"
        "  training_mode: joint\n"
        "  heldout_tasks: [second]\n"
        "datasets:\n"
        "  - {name: first, type: separate}\n"
        "  - {name: second, type: separate}\n",
        encoding="utf-8",
    )
    first_blocks = np.arange(6, dtype=np.int32).reshape(3, 2)
    second_blocks = np.arange(8, dtype=np.int32).reshape(4, 2)

    def build_tasks(*_args: Any, **_kwargs: Any):
        tasks = (
            _Task("first", first_blocks, (), 2),
            _Task("second", second_blocks, (), 3),
        )
        return tasks, {"first": first_blocks, "second": second_blocks}, {}

    monkeypatch.setattr(data_loader, "build_separate_tasks", build_tasks)
    monkeypatch.setattr(data_loader, "Domain", _Task)

    runtime = data_loader.load_runtime_data(
        config_path,
        tokenizer=object(),
        sequence_length=2,
        batch_size=2,
        eval_batches=1,
    )

    assert [task.name for task in runtime.tasks] == ["first"]
    assert runtime.task_order == ("first",)
    assert runtime.heldout_tasks == ("second",)
    assert runtime.metadata["training_mode"] == "joint"
    assert runtime.training_tasks is not None
    assert [task.name for task in runtime.training_tasks] == ["joint"]
    np.testing.assert_array_equal(
        runtime.training_tasks[0].train_blocks,
        np.tile(first_blocks[:2], (2, 1)),
    )
    assert set(runtime.test_blocks) == {"first", "second"}

    sequential = data_loader.load_runtime_data(
        config_path,
        tokenizer=object(),
        task_order=("first", "second"),
        training_mode="sequential",
        heldout_tasks=(),
        sequence_length=2,
        batch_size=2,
        eval_batches=1,
    )
    assert sequential.training_tasks is sequential.tasks
    assert sequential.task_order == ("first", "second")
