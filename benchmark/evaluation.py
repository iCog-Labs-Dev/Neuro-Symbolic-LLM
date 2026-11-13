"""Model-agnostic helpers for external language-model evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def aggregate_lm_rows(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    """Aggregate additive language-model statistics by evaluated token count."""
    if not rows:
        return {"status": "insufficient_data", "tokens": 0}

    tokens = 0
    totals = {name: [] for name in ("loss_sum", "base_loss_sum", "kl_sum", "correct")}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"LM row {index} must be a mapping")
        missing = {"tokens", *totals} - set(row)
        if missing:
            raise ValueError(f"LM row {index} is missing {sorted(missing)}")

        row_tokens = row["tokens"]
        if isinstance(row_tokens, bool) or not isinstance(row_tokens, Integral):
            raise ValueError(f"LM row {index} tokens must be a nonnegative integer")
        row_tokens = int(row_tokens)
        if row_tokens < 0:
            raise ValueError(f"LM row {index} tokens must be a nonnegative integer")

        values = {
            name: _finite_float(row[name], field=f"LM row {index} {name}")
            for name in totals
        }
        if not 0.0 <= values["correct"] <= row_tokens:
            raise ValueError(f"LM row {index} correct must be between 0 and tokens")

        tokens += row_tokens
        for name, value in values.items():
            totals[name].append(value)

    if tokens == 0:
        return {"status": "insufficient_data", "tokens": 0}

    loss = math.fsum(totals["loss_sum"]) / tokens
    base_loss = math.fsum(totals["base_loss_sum"]) / tokens
    try:
        perplexity = math.exp(loss)
        base_perplexity = math.exp(base_loss)
    except OverflowError as exc:
        raise ValueError("aggregated LM loss is too large to exponentiate") from exc

    return {
        "status": "measured",
        "tokens": tokens,
        "loss": loss,
        "base_loss": base_loss,
        "perplexity": perplexity,
        "base_perplexity": base_perplexity,
        "kl": math.fsum(totals["kl_sum"]) / tokens,
        "token_accuracy": math.fsum(totals["correct"]) / tokens,
    }


def evaluate_long_range(
    config: Mapping[str, Any],
    score_choice: Callable[[str, str], float],
) -> dict[str, Any]:
    """Evaluate inline multiple-choice prompts with an external scoring callback."""
    if not isinstance(config, Mapping):
        raise ValueError("long-range configuration must be a mapping")
    if config.get("schema") != "long-range/v1":
        raise ValueError("long-range probes must use schema 'long-range/v1'")
    if not callable(score_choice):
        raise ValueError("score_choice must be callable")

    items = config.get("items", [])
    if not isinstance(items, list):
        raise ValueError("long-range items must be a list")
    if not items:
        return {"status": "not_configured", "reason": "no long-range probes"}

    required = {
        "id",
        "family",
        "distance",
        "context",
        "question",
        "choices",
        "answer",
    }
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"long-range item {index} must be a mapping")
        missing = required - set(item)
        if missing:
            raise ValueError(f"long-range item is missing {sorted(missing)}")

        text_fields = ("id", "family", "distance", "context", "question")
        for field in text_fields:
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"long-range item {index} {field} must be nonempty text"
                )
        item_id = item["id"]
        if item_id in seen_ids:
            raise ValueError(f"long-range item id {item_id!r} is duplicated")
        seen_ids.add(item_id)

        choices = item["choices"]
        if (
            not isinstance(choices, list)
            or len(choices) < 2
            or any(not isinstance(choice, str) or not choice for choice in choices)
        ):
            raise ValueError(
                f"long-range item {item_id!r} requires at least two text choices"
            )

        answer = item["answer"]
        if isinstance(answer, bool) or not isinstance(answer, Integral):
            raise ValueError(f"long-range item {item_id!r} answer must be an index")
        answer = int(answer)
        if not 0 <= answer < len(choices):
            raise ValueError(f"long-range item {item_id!r} answer is out of range")

        filler = item.get("filler", "")
        if not isinstance(filler, str):
            raise ValueError(f"long-range item {item_id!r} filler must be text")
        repeat = item.get("filler_repeat", 0)
        if isinstance(repeat, bool) or not isinstance(repeat, Integral) or repeat < 0:
            raise ValueError(
                f"long-range item {item_id!r} filler_repeat must be a nonnegative integer"
            )

        prompt = item["context"] + (filler * int(repeat)) + item["question"]
        scores = [
            _finite_float(
                score_choice(prompt, choice),
                field=f"long-range item {item_id!r} score {choice!r}",
            )
            for choice in choices
        ]
        predicted = max(range(len(scores)), key=scores.__getitem__)
        distractors = [score for choice, score in enumerate(scores) if choice != answer]
        rows.append(
            {
                "id": item_id,
                "family": item["family"],
                "distance": item["distance"],
                "answer": answer,
                "predicted": predicted,
                "correct": predicted == answer,
                "margin": scores[answer] - max(distractors),
                "scores": scores,
            }
        )

    by_family: dict[str, dict[str, float]] = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "accuracy": math.fsum(row["correct"] for row in selected) / len(selected),
            "mean_margin": math.fsum(row["margin"] for row in selected) / len(selected),
        }

    by_distance: dict[str, float] = {}
    for distance in sorted({row["distance"] for row in rows}):
        selected = [row for row in rows if row["distance"] == distance]
        by_distance[distance] = math.fsum(row["correct"] for row in selected) / len(
            selected
        )

    near_far_gap = None
    if "near" in by_distance and "far" in by_distance:
        near_far_gap = by_distance["near"] - by_distance["far"]

    return {
        "status": "measured",
        "accuracy": math.fsum(row["correct"] for row in rows) / len(rows),
        "mean_margin": math.fsum(row["margin"] for row in rows) / len(rows),
        "by_family": by_family,
        "by_distance": by_distance,
        "near_far_gap": near_far_gap,
        "items": rows,
    }


__all__ = ["aggregate_lm_rows", "evaluate_long_range"]
