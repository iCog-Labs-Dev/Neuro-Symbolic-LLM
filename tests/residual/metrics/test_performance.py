"""Unit tests for the continual-learning and timing metrics. No model needed."""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from metrics.performance import (
    accuracy_forgetting,
    apply_multilabel_platt,
    classification_metrics,
    fit_frozen_scaler,
    fit_multilabel_platt,
    fit_ridge_probe,
    fit_temperature,
    mean_forgetting,
    mean_seen_accuracy,
    multilabel_calibration_error,
    multilabel_metrics,
    multilabel_task_losses,
    per_example_task_losses,
    pool_top_class_calibration,
    pooled_top_class_ece,
    ridge_probe_probabilities,
    ridge_probe_scores,
    select_multilabel_threshold,
    single_label_metrics,
    single_label_task_losses,
    temperature_scale,
    top_class_calibration_stats,
    top_class_ece,
    top_class_ece_from_stats,
    training_step_time,
)


def test_multilabel_calibration_error_is_zero_for_perfect_probabilities() -> None:
    targets = np.asarray([[1, 0], [0, 1]])
    pooled, macro = multilabel_calibration_error(targets, targets.astype(float))
    assert pooled == pytest.approx(0.0)
    assert macro == pytest.approx(0.0)


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
    assert float(mean_forgetting(ACC, 1)) == pytest.approx(0.20)
    assert float(mean_forgetting(ACC, 2)) == pytest.approx((0.30 + 0.05) / 2)
    assert float(accuracy_forgetting(ACC, 2)) == pytest.approx((0.30 + 0.05) / 2)


def test_mean_forgetting_nan_aware() -> None:
    acc = np.array(
        [
            [np.nan, 0.20, 0.10],
            [np.nan, 0.80, 0.15],
            [np.nan, 0.75, 0.90],
        ]
    )
    assert np.isnan(float(mean_forgetting(acc, 1)))
    assert np.isnan(float(accuracy_forgetting(acc, 1)))
    assert float(mean_forgetting(acc, 2)) == pytest.approx(0.05)
    assert float(accuracy_forgetting(acc, 2)) == pytest.approx(0.05)


def test_classification_metrics() -> None:
    metrics = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_classification_metrics_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        classification_metrics([0, 1], [0])


def test_mean_forgetting_loss_matches_negated_accuracy() -> None:
    loss = -ACC
    for t in (1, 2):
        np.testing.assert_allclose(
            float(mean_forgetting(loss, t, higher_is_better=False)),
            float(mean_forgetting(ACC, t)),
            rtol=1e-6,
        )


def test_mean_forgetting_ignores_rows_before_the_task_was_trained() -> None:
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
        mean_seen_accuracy(ACC, t)


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


def test_frozen_scaler_handles_constant_columns_without_refitting() -> None:
    train = np.array([[10.0, 3.0], [12.0, 3.0], [14.0, 3.0]])
    scaler = fit_frozen_scaler(train)

    np.testing.assert_allclose(scaler.mean, [12.0, 3.0])
    np.testing.assert_allclose(scaler.scale, [np.std([10.0, 12.0, 14.0]), 1.0])
    np.testing.assert_allclose(
        scaler.transform([[112.0, 3.0]])[0, 0], 100.0 / scaler.scale[0]
    )
    with pytest.raises(ValueError):
        scaler.mean[0] = 0.0


def test_ridge_probe_is_deterministic_and_keeps_unregularized_intercept() -> None:
    features = np.array(
        [
            [10.0, 2.0],
            [11.0, 2.0],
            [12.0, 2.0],
            [20.0, 2.0],
            [21.0, 2.0],
            [22.0, 2.0],
        ]
    )
    targets = np.array(["old", "old", "old", "new", "new", "new"])

    first = fit_ridge_probe(features, targets, alpha=0.1)
    second = fit_ridge_probe(features, targets, alpha=0.1)

    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.intercept, second.intercept)
    np.testing.assert_allclose(first.intercept, [0.5, 0.5])
    np.testing.assert_array_equal(first.predict(features), targets)
    assert first.scaler.mean[0] == pytest.approx(16.0)
    assert ridge_probe_scores(first, [[100.0, 2.0]]).shape == (1, 2)
    probabilities = ridge_probe_probabilities(first, features)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)


def test_ridge_probe_supports_high_dimensional_multilabel_targets() -> None:
    features = np.arange(20.0).reshape(4, 5)
    targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

    probe = fit_ridge_probe(features, targets, alpha=0.5)
    scores = ridge_probe_scores(probe, features)
    probabilities = ridge_probe_probabilities(probe, features)

    assert probe.multilabel
    assert scores.shape == targets.shape
    assert np.all(np.isfinite(scores))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_single_label_metrics_match_hand_calculation() -> None:
    targets = np.array([0, 1, 2, 2])
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.40, 0.35, 0.25],
            [0.10, 0.20, 0.70],
            [0.20, 0.50, 0.30],
        ]
    )

    metrics = single_label_metrics(targets, probabilities, bins=2)

    assert set(metrics) == {"accuracy", "macro_f1", "micro_f1", "nll", "brier", "ece"}
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)
    assert metrics["micro_f1"] == pytest.approx(0.5)
    assert metrics["nll"] == pytest.approx(-np.mean(np.log([0.70, 0.35, 0.70, 0.30])))
    one_hot = np.eye(3)[targets]
    assert metrics["brier"] == pytest.approx(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )
    assert metrics["ece"] == pytest.approx(0.125)


def test_single_label_metrics_include_absent_explicit_classes() -> None:
    targets = np.array(["court", "contract"])
    probabilities = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])

    metrics = single_label_metrics(
        targets,
        probabilities,
        labels=["contract", "court", "tax"],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(1.0)


def test_single_label_per_example_losses_and_logits_path() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    probabilities = temperature_scale(logits, 1.0)
    expected = -np.log(probabilities[[0, 1], [0, 1]])

    np.testing.assert_allclose(
        single_label_task_losses([0, 1], probabilities), expected
    )
    np.testing.assert_allclose(
        per_example_task_losses([0, 1], logits, multilabel=False, from_logits=True),
        expected,
    )


def test_single_label_metrics_reject_invalid_probabilities_and_targets() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        single_label_metrics([0], [[0.2, 0.2]])
    with pytest.raises(ValueError, match="not present"):
        single_label_metrics(["unknown"], [[0.5, 0.5]], labels=["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        fit_frozen_scaler([[0.0, np.nan]])


def test_multilabel_metrics_match_hand_calculation() -> None:
    targets = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]])
    probabilities = np.array(
        [
            [0.9, 0.1, 0.8],
            [0.6, 0.8, 0.2],
            [0.7, 0.4, 0.3],
            [0.2, 0.1, 0.1],
        ]
    )

    metrics = multilabel_metrics(targets, probabilities, threshold=0.5)

    assert metrics["micro_precision"] == pytest.approx(4 / 5)
    assert metrics["micro_recall"] == pytest.approx(4 / 5)
    assert metrics["micro_f1"] == pytest.approx(4 / 5)
    assert metrics["macro_precision"] == pytest.approx((2 / 3 + 1 + 1) / 3)
    assert metrics["macro_recall"] == pytest.approx((1 + 1 / 2 + 1) / 3)
    assert metrics["macro_f1"] == pytest.approx((4 / 5 + 2 / 3 + 1) / 3)
    assert metrics["subset_accuracy"] == pytest.approx(0.5)
    expected_losses = -np.mean(
        targets * np.log(probabilities) + (1 - targets) * np.log1p(-probabilities),
        axis=1,
    )
    assert metrics["bce"] == pytest.approx(np.mean(expected_losses))
    assert metrics["brier"] == pytest.approx(np.mean((probabilities - targets) ** 2))
    np.testing.assert_allclose(
        multilabel_task_losses(targets, probabilities), expected_losses
    )
    np.testing.assert_allclose(
        per_example_task_losses(targets, probabilities, multilabel=True),
        expected_losses,
    )


def test_multilabel_metrics_are_finite_with_no_positive_labels_or_predictions() -> None:
    metrics = multilabel_metrics(np.zeros((3, 2)), np.zeros((3, 2)))

    for name in (
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        assert metrics[name] == 0.0
    assert metrics["subset_accuracy"] == 1.0
    assert np.isfinite(metrics["bce"])


def test_multilabel_metrics_reject_nonbinary_targets() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        multilabel_metrics([[0.5, 0.0]], [[0.5, 0.5]])


def test_validation_threshold_maximizes_global_micro_f1() -> None:
    targets = np.array([[1, 1], [1, 0]])
    probabilities = np.array([[0.9, 0.4], [0.6, 0.2]])

    selected = select_multilabel_threshold(targets, probabilities)

    assert selected == pytest.approx(0.4)
    assert multilabel_metrics(targets, probabilities, threshold=selected)[
        "micro_f1"
    ] == pytest.approx(1.0)


def test_validation_threshold_tie_prefers_nearest_half_then_lower() -> None:
    targets = np.array([[1, 0], [1, 0]])
    probabilities = np.array([[0.8, 0.2], [0.7, 0.1]])

    selected = select_multilabel_threshold(
        targets, probabilities, thresholds=[0.6, 0.4]
    )

    assert selected == pytest.approx(0.4)


def test_temperature_scaling_is_deterministic_and_does_not_increase_validation_nll() -> (
    None
):
    logits = np.array([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    targets = np.array([0, 1, 1, 1])
    before = np.mean(single_label_task_losses(targets, temperature_scale(logits, 1.0)))

    first = fit_temperature(logits, targets)
    second = fit_temperature(logits, targets)
    after = np.mean(single_label_task_losses(targets, temperature_scale(logits, first)))

    assert first == second
    assert first > 1.0
    assert after <= before
    np.testing.assert_allclose(np.sum(temperature_scale(logits, first), axis=1), 1.0)


def test_temperature_scaling_uses_identity_for_uninformative_logits() -> None:
    logits = np.zeros((4, 3))
    assert fit_temperature(logits, [0, 1, 2, 0]) == pytest.approx(1.0)


def test_independent_platt_calibration_is_safe_for_constant_labels() -> None:
    base_score = np.linspace(-2.0, 2.0, 20)
    scores = np.column_stack([base_score, base_score, -base_score])
    targets = np.column_stack(
        [base_score > 0.0, np.zeros(20, dtype=bool), np.ones(20, dtype=bool)]
    ).astype(int)
    uncalibrated = 1.0 / (1.0 + np.exp(-scores[:, :1]))
    before = np.mean(multilabel_task_losses(targets[:, :1], uncalibrated))

    calibrator = fit_multilabel_platt(scores, targets)
    repeated = fit_multilabel_platt(scores, targets)
    calibrated = apply_multilabel_platt(calibrator, scores)
    after = np.mean(multilabel_task_losses(targets[:, :1], calibrated[:, :1]))

    np.testing.assert_array_equal(calibrator.slopes, repeated.slopes)
    np.testing.assert_array_equal(calibrator.intercepts, repeated.intercepts)
    np.testing.assert_array_equal(calibrator.constant_labels, [False, True, True])
    assert calibrator.slopes[0] > 0.0
    assert after <= before
    np.testing.assert_allclose(calibrated[:, 1], 0.5 / 21.0)
    np.testing.assert_allclose(calibrated[:, 2], 20.5 / 21.0)
    assert np.all(np.isfinite(calibrated))


def test_pooled_top_class_ece_uses_global_sufficient_statistics() -> None:
    target_batches = [np.array([0]), np.array([1, 1, 0])]
    probability_batches = [
        np.array([[0.9, 0.1]]),
        np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]]),
    ]
    stats = [
        top_class_calibration_stats(targets, probabilities, bins=2)
        for targets, probabilities in zip(
            target_batches, probability_batches, strict=True
        )
    ]

    pooled_stats = pool_top_class_calibration(stats)
    pooled = top_class_ece_from_stats(pooled_stats)
    direct = top_class_ece(
        np.concatenate(target_batches), np.concatenate(probability_batches), bins=2
    )
    convenience = pooled_top_class_ece(target_batches, probability_batches, bins=2)
    mean_batch_ece = np.mean([top_class_ece_from_stats(item) for item in stats])

    assert pooled == pytest.approx(0.175)
    assert pooled == pytest.approx(direct)
    assert pooled == pytest.approx(convenience)
    assert pooled != pytest.approx(mean_batch_ece)
