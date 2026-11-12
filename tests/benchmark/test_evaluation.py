from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from benchmark.evaluation import (
    aggregate_lm_rows,
    evaluate_long_range,
    evaluate_symbolic_rankings,
    evaluate_task_probe,
    fit_task_probe,
    semantic_factor_effects,
)
from runtime import data_loader
from runtime.data_loader import LabeledSplit


def _split(name: str, targets: np.ndarray, *, multilabel: bool) -> LabeledSplit:
    count = len(targets)
    width = targets.shape[1]
    ids = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
    return LabeledSplit(
        task_name=name,
        split="test",
        input_ids=ids,
        attention_mask=np.ones_like(ids, dtype=bool),
        transformed_input_ids=ids,
        transformed_attention_mask=np.ones_like(ids, dtype=bool),
        targets=targets,
        label_names=tuple(str(index) for index in range(width)),
        factor_names=tuple(f"{name}:{index}" for index in range(width)),
        factor_mask=targets.astype(bool),
        example_ids=tuple(str(index) for index in range(count)),
        multilabel=multilabel,
        fingerprint=f"{name}-fingerprint",
    )


def test_frozen_multilabel_probe_is_operational() -> None:
    targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
    train = _split("eurlex", targets, multilabel=True)
    validation = _split("eurlex", targets, multilabel=True)
    features = np.asarray([[2, 0], [1, 0], [0, 1], [0, 2]], dtype=float)
    probe, _ = fit_task_probe(
        train,
        validation,
        features,
        features,
        alpha=1e-3,
        bins=5,
        acquisition_checkpoint=0,
    )
    metrics, probabilities, losses = evaluate_task_probe(probe, validation, features)
    assert metrics["micro_f1"] == 1.0
    assert probabilities.shape == targets.shape
    assert losses.shape == targets.shape


def test_semantic_effects_use_each_labels_own_loss() -> None:
    factors = np.asarray([[1, 0], [1, 1]], dtype=bool)
    full = np.asarray([[0.1, 0.2], [0.2, 0.3]])
    ablated = np.asarray([[[0.6, 9.0], [0.7, 8.0]]])
    result = semantic_factor_effects(
        full,
        ablated,
        factors,
        min_support=1,
    )
    assert result["signed_effects"][0, 0] == np.mean([0.5, 0.5])
    assert result["signed_effects"][0, 1] == 7.7


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
    assert result["loss"] == pytest.approx(2.5)
    assert result["token_accuracy"] == pytest.approx(0.25)


def test_long_range_reports_families_and_distance() -> None:
    config = {
        "schema": "long-range/v1",
        "items": [
            {
                "id": "x",
                "family": "relation",
                "distance": "near",
                "context": "ctx",
                "question": " q",
                "choices": [" correct", " wrong"],
                "answer": 0,
            }
        ],
    }
    result = evaluate_long_range(
        config, lambda prompt, choice: 1.0 if "correct" in choice else 0.0
    )
    assert result["accuracy"] == 1.0
    assert result["by_family"]["relation"]["accuracy"] == 1.0


def test_symbolic_rankings_are_operational(tmp_path: Path) -> None:
    path = tmp_path / "rankings.jsonl"
    path.write_text(
        json.dumps({"retrieved": ["a", "b"], "relevant": ["a"]}) + "\n",
        encoding="utf-8",
    )
    result = evaluate_symbolic_rankings(path, cutoffs=(1, 2))
    assert result["status"] == "measured"
    assert result["precision"]["@1"] == 1.0


@dataclass
class _ClassLabel:
    names: list[str]


@dataclass
class _Sequence:
    feature: _ClassLabel


class _Dataset(list):
    def __init__(self, rows, feature):
        super().__init__(rows)
        self.column_names = list(rows[0])
        self.features = {"text": object(), "labels": feature, "label": feature}


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        length = kwargs["max_length"]
        values = [min(ord(char), 255) for char in text][:length]
        mask = [1] * len(values)
        values += [0] * (length - len(values))
        mask += [0] * (length - len(mask))
        return {
            "input_ids": np.asarray([values], dtype=np.int32),
            "attention_mask": np.asarray([mask], dtype=np.int32),
        }


def _config(multilabel: bool) -> dict:
    return {
        "name": "eurlex" if multilabel else "scotus",
        "dataset": "coastalcph/lex_glue",
        "config": "eurlex" if multilabel else "scotus",
        "revision": "abc123",
        "text_field": "text",
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "legal_task": {
            "label_field": "labels" if multilabel else "label",
            "multilabel": multilabel,
            "expected_num_labels": 3,
        },
    }


def test_loads_multilabel_targets_and_semantic_factors(monkeypatch) -> None:
    dataset = _Dataset(
        [{"text": "alpha", "labels": [0, 2]}, {"text": "beta", "labels": [1]}],
        _Sequence(_ClassLabel(["a", "b", "c"])),
    )
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(data_loader, "load_dataset_split", load)
    spec = data_loader.parse_legal_task_spec(_config(True))
    assert spec is not None
    split = data_loader.load_labeled_split(
        spec, "test", _Tokenizer(), sequence_length=8
    )
    np.testing.assert_array_equal(split.targets, [[1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(split.factor_mask, split.targets.astype(bool))
    assert split.factor_names == ("eurlex:a", "eurlex:b", "eurlex:c")
    assert len(split.fingerprint) == 64
    assert calls[0][1]["revision"] == "abc123"
    assert calls[0][0][1] == "test"


def test_loads_single_label_targets(monkeypatch) -> None:
    dataset = _Dataset(
        [{"text": "one", "label": 2}, {"text": "two", "label": 0}],
        _ClassLabel(["x", "y", "z"]),
    )
    monkeypatch.setattr(
        data_loader, "load_dataset_split", lambda *args, **kwargs: dataset
    )
    spec = data_loader.parse_legal_task_spec(_config(False))
    assert spec is not None
    split = data_loader.load_labeled_split(
        spec, "validation", _Tokenizer(), sequence_length=6
    )
    np.testing.assert_array_equal(split.probe_targets(), [2, 0])
    assert split.multilabel is False


def test_legal_task_requires_pinned_revision() -> None:
    config = _config(False)
    del config["revision"]
    with pytest.raises(ValueError, match="pinned revision"):
        data_loader.parse_legal_task_spec(config)
