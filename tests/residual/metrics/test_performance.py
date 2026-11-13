"""Unit tests for language-model, continual-learning, and timing metrics."""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from metrics.performance import (
    calibration_stats,
    cross_entropy_loss,
    ece_from_stats,
    expected_calibration_error,
    kl_to_base,
    mean_forgetting,
    mean_seen_loss,
    paired_deltas,
    perplexity,
    sample_mean_std,
    token_accuracy,
    training_step_time,
)

LOSS = np.array(
    [
        [1.00, 0.20, 0.10],
        [1.20, 0.80, 0.15],
        [1.40, 1.00, 0.70],
    ]
)


def test_cross_entropy_loss_shifts_and_masks_tokens() -> None:
    logits = jnp.asarray([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    labels = jnp.asarray([[0, 1, -100]])

    assert float(cross_entropy_loss(logits, labels)) == pytest.approx(
        np.log1p(np.exp(2.0))
    )


def test_perplexity_exponentiates_loss() -> None:
    assert float(perplexity(np.log(4.0))) == pytest.approx(4.0)


def test_token_accuracy_shifts_and_masks_tokens() -> None:
    logits = jnp.zeros((1, 4, 3))
    logits = logits.at[0, 0, 1].set(3.0)
    logits = logits.at[0, 1, 0].set(3.0)
    logits = logits.at[0, 2, 0].set(3.0)
    labels = jnp.asarray([[0, 1, -100, 2]])

    assert float(token_accuracy(logits, labels)) == pytest.approx(0.5)


def test_kl_to_base_is_zero_for_identical_logits() -> None:
    logits = jnp.asarray([[[1.0, -1.0], [0.5, 0.0]]])
    assert float(kl_to_base(logits, logits)) == pytest.approx(0.0, abs=1e-7)


def test_token_ece_is_finite_when_bins_are_empty() -> None:
    logits = jnp.zeros((2, 5, 10), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(50.0)
    labels = jnp.zeros((2, 5), dtype=jnp.int32)

    value = float(expected_calibration_error(logits, labels, bins=15))

    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-3)


def test_token_ece_stats_can_be_pooled_across_batches() -> None:
    logits_a = jnp.asarray([[[3.0, 0.0], [0.0, 3.0], [1.0, 0.0]]])
    logits_b = jnp.asarray([[[0.0, 2.0], [2.0, 0.0], [0.0, 1.0]]])
    labels_a = jnp.asarray([[0, 0, 1]])
    labels_b = jnp.asarray([[1, 1, 0]])

    stats_a = calibration_stats(logits_a, labels_a, bins=5)
    stats_b = calibration_stats(logits_b, labels_b, bins=5)
    pooled = float(
        ece_from_stats(*(a + b for a, b in zip(stats_a, stats_b, strict=True)))
    )
    direct = float(
        expected_calibration_error(
            jnp.concatenate((logits_a, logits_b)),
            jnp.concatenate((labels_a, labels_b)),
            bins=5,
        )
    )

    assert pooled == pytest.approx(direct)


def test_paired_deltas_match_only_shared_seeds() -> None:
    values = {
        "base": {0: 1.0, 1: 2.0},
        "adapter": {0: 0.75, 2: 3.0},
    }

    assert paired_deltas(values, "base") == {"adapter": {0: -0.25}}


def test_sample_mean_std_uses_sample_standard_deviation() -> None:
    assert sample_mean_std([]) == (None, None)
    assert sample_mean_std([2.0]) == (2.0, None)
    mean, std = sample_mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)


def test_mean_seen_loss() -> None:
    assert float(mean_seen_loss(LOSS, 0)) == pytest.approx(1.00)
    assert float(mean_seen_loss(LOSS, 1)) == pytest.approx((1.20 + 0.80) / 2)
    assert float(mean_seen_loss(LOSS, 2)) == pytest.approx((1.40 + 1.00 + 0.70) / 3)


def test_mean_forgetting_uses_best_prior_loss() -> None:
    assert float(mean_forgetting(LOSS, 1)) == pytest.approx(0.20)
    assert float(mean_forgetting(LOSS, 2)) == pytest.approx((0.40 + 0.20) / 2)


def test_mean_forgetting_is_nan_aware() -> None:
    loss = np.array(
        [
            [np.nan, 0.20, 0.10],
            [np.nan, 0.80, 0.15],
            [np.nan, 0.75, 0.90],
        ]
    )
    assert np.isnan(float(mean_forgetting(loss, 1)))
    assert float(mean_forgetting(loss, 2)) == pytest.approx(-0.05)


def test_mean_forgetting_ignores_rows_before_a_task_was_trained() -> None:
    loss = np.array(
        [
            [0.9, 0.01, 0.0],
            [0.9, 0.50, 0.0],
            [0.9, 0.50, 0.9],
        ]
    )
    assert float(mean_forgetting(loss, 2)) == pytest.approx(0.0)


def test_improving_loss_has_negative_forgetting() -> None:
    improving = np.array([[0.5, 0.0], [0.3, 0.2]])
    assert float(mean_forgetting(improving, 1)) == pytest.approx(-0.2)


def test_loss_metrics_accept_jax_arrays() -> None:
    assert float(mean_forgetting(jnp.asarray(LOSS), 2)) == pytest.approx(0.3)


@pytest.mark.parametrize("t", [-1, 3, 1.0])
def test_invalid_t_raises(t: object) -> None:
    with pytest.raises(ValueError):
        mean_seen_loss(LOSS, t)


def test_forgetting_is_undefined_at_t0() -> None:
    with pytest.raises(ValueError, match="undefined at t=0"):
        mean_forgetting(LOSS, 0)


def test_non_matrix_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mean_seen_loss(np.ones(3), 0)


def test_training_step_time_returns_output_and_duration() -> None:
    def step(x: float, *, scale: float) -> dict[str, Any]:
        time.sleep(0.01)
        return {"y": jnp.asarray(x) * scale}

    out, seconds = training_step_time(step, 2.0, scale=3.0)
    assert float(out["y"]) == pytest.approx(6.0)
    assert isinstance(seconds, float)
    assert seconds >= 0.01
