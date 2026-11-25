"""Unit tests for the continual-learning and timing metrics. No model needed."""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from metrics.performance import (
    mean_forgetting,
    mean_seen_accuracy,
    training_step_time,
)

# A[t, j]: accuracy on task j after training task t. The upper triangle
# (j > t) is the zero-shot accuracy on tasks not yet trained.
ACC = np.array(
    [
        [0.90, 0.20, 0.10],
        [0.70, 0.80, 0.15],
        [0.60, 0.75, 0.85],
    ]
)


def test_mean_seen_accuracy() -> None:
    assert float(mean_seen_accuracy(ACC, 0)) == pytest.approx(0.90)
    assert float(mean_seen_accuracy(ACC, 1)) == pytest.approx((0.70 + 0.80) / 2)
    assert float(mean_seen_accuracy(ACC, 2)) == pytest.approx((0.60 + 0.75 + 0.85) / 3)


def test_mean_forgetting_accuracy() -> None:
    # t=1: task 0 dropped 0.90 -> 0.70.
    assert float(mean_forgetting(ACC, 1)) == pytest.approx(0.20)
    # t=2: task 0 peak 0.90 -> 0.60, task 1 peak 0.80 -> 0.75.
    assert float(mean_forgetting(ACC, 2)) == pytest.approx((0.30 + 0.05) / 2)


def test_mean_forgetting_loss_matches_negated_accuracy() -> None:
    # For a loss, lower is better; forgetting is the rise above the best loss.
    loss = -ACC
    for t in (1, 2):
        np.testing.assert_allclose(
            float(mean_forgetting(loss, t, higher_is_better=False)),
            float(mean_forgetting(ACC, t)),
            rtol=1e-6,
        )


def test_mean_forgetting_ignores_rows_before_the_task_was_trained() -> None:
    # Task 1 scores 0.99 zero-shot (row 0) but peaks at 0.50 once trained.
    # The pre-training row is transfer, not a peak to forget from.
    acc = np.array(
        [
            [0.9, 0.99, 0.0],
            [0.9, 0.50, 0.0],
            [0.9, 0.50, 0.9],
        ]
    )
    assert float(mean_forgetting(acc, 2)) == pytest.approx(0.0)


def test_no_forgetting_and_improvement_are_non_positive() -> None:
    improving = np.array([[0.5, 0.0], [0.7, 0.8]])
    assert float(mean_forgetting(improving, 1)) == pytest.approx(-0.2)


def test_accepts_jax_arrays() -> None:
    assert float(mean_forgetting(jnp.asarray(ACC), 2)) == pytest.approx(0.175)


@pytest.mark.parametrize("t", [-1, 3, 1.0])
def test_invalid_t_raises(t: object) -> None:
    with pytest.raises(ValueError):
        mean_seen_accuracy(ACC, t)  # type: ignore[arg-type]


def test_forgetting_undefined_at_t0() -> None:
    with pytest.raises(ValueError, match="undefined at t=0"):
        mean_forgetting(ACC, 0)


def test_non_matrix_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mean_seen_accuracy(np.ones(3), 0)


def test_training_step_time_returns_output_and_duration() -> None:
    def step(x: float, *, scale: float) -> dict[str, Any]:
        time.sleep(0.01)
        return {"y": jnp.asarray(x) * scale}

    out, seconds = training_step_time(step, 2.0, scale=3.0)
    assert float(out["y"]) == pytest.approx(6.0)
    assert isinstance(seconds, float)
    assert seconds >= 0.01
