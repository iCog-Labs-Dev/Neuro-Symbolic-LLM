"""Task-performance, drift, continual-learning and compute metrics.

Continual-learning metrics take ``A[t, j]``: the metric on task ``j`` after
training task ``t`` (0-based, training order).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")

_PROBABILITY_EPS = 1e-15


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal cross-entropy loss with next-token shifting and padding (-100) masking."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    mask = shift_labels != -100
    safe_labels = jnp.where(mask, shift_labels, 0)

    gathered_log_probs = jnp.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1
    ).squeeze(-1)

    loss = -jnp.sum(gathered_log_probs * mask) / jnp.maximum(1.0, jnp.sum(mask))
    return loss


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    log_p = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(adapted_logits.astype(jnp.float32), axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))


def perplexity(loss: jax.Array | float) -> jax.Array:
    return jnp.exp(jnp.asarray(loss, dtype=jnp.float32))


def token_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal next-token top-1 accuracy, ignoring ``-100`` labels."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = target != -100
    predicted = jnp.argmax(logits[:, :-1], axis=-1)
    correct = (predicted == jnp.where(mask, target, 0)) & mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)


def classification_metrics(
    targets: Any, predictions: Any, *, labels: Any | None = None
) -> dict[str, float]:
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            f"targets and predictions must have equal shapes, got "
            f"{target_array.shape} and {prediction_array.shape}"
        )
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError("targets and predictions must be non-empty 1-D arrays")
    classes = (
        list(labels)
        if labels is not None
        else sorted(
            set(target_array.tolist()) | set(prediction_array.tolist()), key=str
        )
    )
    if not classes:
        raise ValueError("labels must contain at least one class")
    true_positive = []
    false_positive = []
    false_negative = []
    for label in classes:
        target_mask = target_array == label
        prediction_mask = prediction_array == label
        true_positive.append(int(np.count_nonzero(target_mask & prediction_mask)))
        false_positive.append(int(np.count_nonzero(~target_mask & prediction_mask)))
        false_negative.append(int(np.count_nonzero(target_mask & ~prediction_mask)))
    precision = [
        tp / (tp + fp) if tp + fp else 0.0
        for tp, fp in zip(true_positive, false_positive, strict=True)
    ]
    recall = [
        tp / (tp + fn) if tp + fn else 0.0
        for tp, fn in zip(true_positive, false_negative, strict=True)
    ]
    f1 = [
        2 * p * r / (p + r) if p + r else 0.0
        for p, r in zip(precision, recall, strict=True)
    ]
    return {
        "accuracy": float(np.mean(target_array == prediction_array)),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
    }


def _float_matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_targets(
    targets: Any, *, shape: tuple[int, int] | None = None
) -> np.ndarray:
    array = _float_matrix(targets, name="targets")
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"targets and scores must have equal shapes, got {array.shape} and {shape}"
        )
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError("multilabel targets must contain only 0 and 1")
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _probability_matrix(
    values: Any, *, name: str, from_logits: bool, multilabel: bool
) -> np.ndarray:
    array = _float_matrix(values, name=name)
    if from_logits:
        return _sigmoid(array) if multilabel else _softmax(array)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    if not multilabel:
        row_sums = np.sum(array, axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-12):
            raise ValueError(f"rows of {name} must sum to 1")
        array = array / row_sums
    return array


def _class_array(labels: Any) -> np.ndarray:
    classes = np.asarray(list(labels))
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("labels must be a non-empty 1-D sequence")
    for index, label in enumerate(classes):
        if np.count_nonzero(classes == label) != 1:
            raise ValueError(f"labels contains duplicate class {label!r}")
        if index and isinstance(label, float) and np.isnan(label):
            raise ValueError("labels must not contain NaN")
    return classes


def _inferred_classes(targets: np.ndarray) -> np.ndarray:
    try:
        classes = np.unique(targets)
    except TypeError:
        classes = np.asarray(
            sorted(
                set(targets.tolist()),
                key=lambda value: (type(value).__name__, repr(value)),
            ),
            dtype=object,
        )
    if classes.size == 0:
        raise ValueError("targets must be non-empty")
    return classes


def _classes_for_probabilities(
    targets: np.ndarray, class_count: int, labels: Any | None
) -> np.ndarray:
    if labels is not None:
        classes = _class_array(labels)
        if classes.size != class_count:
            raise ValueError(
                f"labels has {classes.size} classes, but probabilities has "
                f"{class_count} columns"
            )
        return classes

    if np.issubdtype(targets.dtype, np.number):
        numeric = np.asarray(targets, dtype=np.float64)
        if (
            np.all(np.isfinite(numeric))
            and np.all(numeric == np.floor(numeric))
            and np.all((numeric >= 0) & (numeric < class_count))
        ):
            return np.arange(class_count)

    classes = _inferred_classes(targets)
    if classes.size != class_count:
        raise ValueError(
            "labels is required when probability columns include classes absent "
            "from targets"
        )
    return classes


def _target_indices(targets: Any, classes: np.ndarray) -> np.ndarray:
    target_array = np.asarray(targets)
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError("targets must be a non-empty 1-D array")
    matches = np.column_stack([target_array == label for label in classes])
    match_count = np.sum(matches, axis=1)
    if not np.all(match_count == 1):
        unknown = target_array[np.flatnonzero(match_count != 1)[0]]
        raise ValueError(f"target {unknown!r} is not present exactly once in labels")
    return np.argmax(matches, axis=1)


def _readonly_array(values: Any, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FrozenStandardScaler:
    """Training-set feature standardization reused unchanged on later splits."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly_array(self.mean)
        scale = _readonly_array(self.scale)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.size == 0:
            raise ValueError("scaler mean and scale must be equal non-empty vectors")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("scaler parameters must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("scaler scale must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, features: Any) -> np.ndarray:
        array = _float_matrix(features, name="features")
        if array.shape[1] != self.mean.size:
            raise ValueError(
                f"features has {array.shape[1]} columns, expected {self.mean.size}"
            )
        return (array - self.mean) / self.scale


def fit_frozen_scaler(features: Any) -> FrozenStandardScaler:
    """Fit a standard scaler once; constant features retain scale one."""
    array = _float_matrix(features, name="features")
    mean = np.mean(array, axis=0)
    scale = np.std(array, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return FrozenStandardScaler(mean=mean, scale=scale)


@dataclass(frozen=True)
class RidgeProbe:
    """Closed-form ridge probe with an unregularized intercept."""

    weights: np.ndarray
    intercept: np.ndarray
    scaler: FrozenStandardScaler
    classes: np.ndarray | None
    multilabel: bool

    def __post_init__(self) -> None:
        weights = _readonly_array(self.weights)
        intercept = _readonly_array(self.intercept)
        if weights.ndim != 2 or intercept.ndim != 1:
            raise ValueError("probe weights must be 2-D and intercept must be 1-D")
        if weights.shape != (self.scaler.mean.size, intercept.size):
            raise ValueError("probe and scaler dimensions do not agree")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(intercept)):
            raise ValueError("probe parameters must be finite")
        if self.multilabel and self.classes is not None:
            raise ValueError("a multilabel probe must not define classes")
        if not self.multilabel and self.classes is None:
            raise ValueError("a single-label probe must define classes")
        classes = (
            None
            if self.classes is None
            else _readonly_array(self.classes, dtype=np.asarray(self.classes).dtype)
        )
        if classes is not None and (
            classes.ndim != 1 or classes.size != intercept.size
        ):
            raise ValueError("probe classes and outputs do not agree")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "classes", classes)

    def decision_function(self, features: Any) -> np.ndarray:
        return self.scaler.transform(features) @ self.weights + self.intercept

    def predict_proba(self, features: Any) -> np.ndarray:
        scores = self.decision_function(features)
        return _sigmoid(scores) if self.multilabel else _softmax(scores)

    def predict(self, features: Any, *, threshold: float = 0.5) -> np.ndarray:
        scores = self.decision_function(features)
        if self.multilabel:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be in [0, 1]")
            return (_sigmoid(scores) >= threshold).astype(np.int64)
        assert self.classes is not None
        return self.classes[np.argmax(scores, axis=1)]


def fit_ridge_probe(
    features: Any,
    targets: Any,
    *,
    alpha: float = 1.0,
    labels: Any | None = None,
) -> RidgeProbe:
    """Fit a deterministic one-vs-rest or multilabel ridge probe.

    The scaler and intercept are fit only from these inputs. The intercept is
    not regularized; ``alpha`` applies only to feature weights.
    """
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and >= 0")
    feature_array = _float_matrix(features, name="features")
    target_array = np.asarray(targets)
    classes: np.ndarray | None
    if target_array.ndim == 1:
        if target_array.size != feature_array.shape[0] or target_array.size == 0:
            raise ValueError("one target is required for each feature row")
        classes = (
            _inferred_classes(target_array) if labels is None else _class_array(labels)
        )
        target_indices = _target_indices(target_array, classes)
        response = np.eye(classes.size, dtype=np.float64)[target_indices]
        multilabel = False
    elif target_array.ndim == 2:
        if labels is not None:
            raise ValueError("labels is only valid for single-label targets")
        response = _binary_targets(target_array)
        if response.shape[0] != feature_array.shape[0]:
            raise ValueError("one target row is required for each feature row")
        classes = None
        multilabel = True
    else:
        raise ValueError("targets must be a non-empty 1-D or 2-D array")

    scaler = fit_frozen_scaler(feature_array)
    standardized = scaler.transform(feature_array)
    intercept = np.mean(response, axis=0)
    centered_response = response - intercept
    if alpha == 0.0:
        weights = np.linalg.lstsq(standardized, centered_response, rcond=None)[0]
    elif standardized.shape[1] <= standardized.shape[0]:
        system = standardized.T @ standardized
        system.flat[:: system.shape[0] + 1] += alpha
        weights = np.linalg.solve(system, standardized.T @ centered_response)
    else:
        system = standardized @ standardized.T
        system.flat[:: system.shape[0] + 1] += alpha
        weights = standardized.T @ np.linalg.solve(system, centered_response)
    return RidgeProbe(
        weights=weights,
        intercept=intercept,
        scaler=scaler,
        classes=classes,
        multilabel=multilabel,
    )


def ridge_probe_scores(probe: RidgeProbe, features: Any) -> np.ndarray:
    """Return uncalibrated probe scores using the frozen training scaler."""
    if not isinstance(probe, RidgeProbe):
        raise TypeError("probe must be a RidgeProbe")
    return probe.decision_function(features)


def ridge_probe_probabilities(probe: RidgeProbe, features: Any) -> np.ndarray:
    """Convert ridge scores to softmax or independent sigmoid probabilities."""
    if not isinstance(probe, RidgeProbe):
        raise TypeError("probe must be a RidgeProbe")
    return probe.predict_proba(features)


@dataclass(frozen=True)
class TopClassCalibrationStats:
    """Additive sufficient statistics for top-class ECE."""

    confidence_sum: np.ndarray
    correct_sum: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        confidence_sum = _readonly_array(self.confidence_sum)
        correct_sum = _readonly_array(self.correct_sum)
        counts = _readonly_array(self.counts)
        if (
            confidence_sum.ndim != 1
            or correct_sum.shape != confidence_sum.shape
            or counts.shape != confidence_sum.shape
            or confidence_sum.size == 0
        ):
            raise ValueError("calibration statistics must be equal non-empty vectors")
        if not np.all(np.isfinite(confidence_sum)) or not np.all(
            np.isfinite(correct_sum)
        ):
            raise ValueError("calibration sums must be finite")
        if np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
            raise ValueError("calibration counts must be finite and non-negative")
        object.__setattr__(self, "confidence_sum", confidence_sum)
        object.__setattr__(self, "correct_sum", correct_sum)
        object.__setattr__(self, "counts", counts)


def top_class_calibration_stats(
    targets: Any,
    probabilities: Any,
    *,
    labels: Any | None = None,
    bins: int = 15,
    from_logits: bool = False,
) -> TopClassCalibrationStats:
    """Build additive equal-width-bin statistics for multiclass predictions."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be an integer >= 1")
    probability_array = _probability_matrix(
        probabilities,
        name="probabilities",
        from_logits=from_logits,
        multilabel=False,
    )
    target_array = np.asarray(targets)
    if target_array.ndim != 1 or target_array.size != probability_array.shape[0]:
        raise ValueError("one target is required for each probability row")
    classes = _classes_for_probabilities(
        target_array, probability_array.shape[1], labels
    )
    target_indices = _target_indices(target_array, classes)
    predicted = np.argmax(probability_array, axis=1)
    confidence = probability_array[np.arange(probability_array.shape[0]), predicted]
    correct = (predicted == target_indices).astype(np.float64)
    bin_indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    return TopClassCalibrationStats(
        confidence_sum=np.bincount(
            bin_indices, weights=confidence, minlength=bins
        ).astype(np.float64),
        correct_sum=np.bincount(bin_indices, weights=correct, minlength=bins).astype(
            np.float64
        ),
        counts=np.bincount(bin_indices, minlength=bins).astype(np.float64),
    )


def pool_top_class_calibration(
    statistics: Any,
) -> TopClassCalibrationStats:
    """Sum calibration statistics before computing dataset-level ECE."""
    stats = list(statistics)
    if not stats:
        raise ValueError("at least one calibration-statistics object is required")
    if not all(isinstance(item, TopClassCalibrationStats) for item in stats):
        raise TypeError("statistics must contain TopClassCalibrationStats objects")
    shape = stats[0].counts.shape
    if any(item.counts.shape != shape for item in stats[1:]):
        raise ValueError("all calibration statistics must use the same bins")
    return TopClassCalibrationStats(
        confidence_sum=np.sum([item.confidence_sum for item in stats], axis=0),
        correct_sum=np.sum([item.correct_sum for item in stats], axis=0),
        counts=np.sum([item.counts for item in stats], axis=0),
    )


def top_class_ece_from_stats(statistics: TopClassCalibrationStats) -> float:
    """Compute ECE after all batch statistics have been pooled."""
    if not isinstance(statistics, TopClassCalibrationStats):
        raise TypeError("statistics must be TopClassCalibrationStats")
    total = float(np.sum(statistics.counts))
    if total == 0.0:
        return 0.0
    return float(
        np.sum(np.abs(statistics.correct_sum - statistics.confidence_sum)) / total
    )


def top_class_ece(
    targets: Any,
    probabilities: Any,
    *,
    labels: Any | None = None,
    bins: int = 15,
    from_logits: bool = False,
) -> float:
    """Return multiclass ECE based on top-class confidence and correctness."""
    statistics = top_class_calibration_stats(
        targets,
        probabilities,
        labels=labels,
        bins=bins,
        from_logits=from_logits,
    )
    return top_class_ece_from_stats(statistics)


def pooled_top_class_ece(
    target_batches: Any,
    probability_batches: Any,
    *,
    labels: Any | None = None,
    bins: int = 15,
    from_logits: bool = False,
) -> float:
    """Compute ECE from pooled batch counts, correctness and confidence sums."""
    targets = list(target_batches)
    probabilities = list(probability_batches)
    if len(targets) != len(probabilities):
        raise ValueError(
            "target_batches and probability_batches must have equal length"
        )
    statistics = [
        top_class_calibration_stats(
            batch_targets,
            batch_probabilities,
            labels=labels,
            bins=bins,
            from_logits=from_logits,
        )
        for batch_targets, batch_probabilities in zip(
            targets, probabilities, strict=True
        )
    ]
    return top_class_ece_from_stats(pool_top_class_calibration(statistics))


def single_label_task_losses(
    targets: Any,
    probabilities: Any,
    *,
    labels: Any | None = None,
    from_logits: bool = False,
) -> np.ndarray:
    """Return one multiclass negative log-likelihood for each example."""
    probability_array = _probability_matrix(
        probabilities,
        name="probabilities",
        from_logits=from_logits,
        multilabel=False,
    )
    target_array = np.asarray(targets)
    if target_array.ndim != 1 or target_array.size != probability_array.shape[0]:
        raise ValueError("one target is required for each probability row")
    classes = _classes_for_probabilities(
        target_array, probability_array.shape[1], labels
    )
    target_indices = _target_indices(target_array, classes)
    true_probabilities = probability_array[
        np.arange(probability_array.shape[0]), target_indices
    ]
    return -np.log(np.clip(true_probabilities, _PROBABILITY_EPS, 1.0))


def multilabel_task_losses(
    targets: Any, probabilities: Any, *, from_logits: bool = False
) -> np.ndarray:
    """Return mean binary cross-entropy across labels for each example."""
    return np.mean(
        multilabel_label_losses(targets, probabilities, from_logits=from_logits),
        axis=1,
    )


def multilabel_label_losses(
    targets: Any, probabilities: Any, *, from_logits: bool = False
) -> np.ndarray:
    """Return unreduced binary cross-entropy for every example and label."""
    probability_array = _probability_matrix(
        probabilities,
        name="probabilities",
        from_logits=from_logits,
        multilabel=True,
    )
    target_array = _binary_targets(targets, shape=probability_array.shape)
    clipped = np.clip(probability_array, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    losses = -target_array * np.log(clipped) - (1.0 - target_array) * np.log1p(-clipped)
    return losses


def per_example_task_losses(
    targets: Any,
    probabilities: Any,
    *,
    multilabel: bool,
    labels: Any | None = None,
    from_logits: bool = False,
) -> np.ndarray:
    """Dispatch to the task-appropriate per-example probabilistic loss."""
    if multilabel:
        if labels is not None:
            raise ValueError("labels is only valid for single-label tasks")
        return multilabel_task_losses(targets, probabilities, from_logits=from_logits)
    return single_label_task_losses(
        targets,
        probabilities,
        labels=labels,
        from_logits=from_logits,
    )


def single_label_metrics(
    targets: Any,
    probabilities: Any,
    *,
    labels: Any | None = None,
    bins: int = 15,
    from_logits: bool = False,
) -> dict[str, float]:
    """Compute robust multiclass task and calibration metrics."""
    probability_array = _probability_matrix(
        probabilities,
        name="probabilities",
        from_logits=from_logits,
        multilabel=False,
    )
    target_array = np.asarray(targets)
    if target_array.ndim != 1 or target_array.size != probability_array.shape[0]:
        raise ValueError("one target is required for each probability row")
    classes = _classes_for_probabilities(
        target_array, probability_array.shape[1], labels
    )
    target_indices = _target_indices(target_array, classes)
    predicted = np.argmax(probability_array, axis=1)
    true_positive = np.asarray(
        [
            np.count_nonzero((target_indices == index) & (predicted == index))
            for index in range(classes.size)
        ],
        dtype=np.float64,
    )
    false_positive = np.asarray(
        [
            np.count_nonzero((target_indices != index) & (predicted == index))
            for index in range(classes.size)
        ],
        dtype=np.float64,
    )
    false_negative = np.asarray(
        [
            np.count_nonzero((target_indices == index) & (predicted != index))
            for index in range(classes.size)
        ],
        dtype=np.float64,
    )
    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0.0,
    )
    micro_denominator = float(
        2.0 * np.sum(true_positive) + np.sum(false_positive) + np.sum(false_negative)
    )
    micro_f1 = (
        2.0 * float(np.sum(true_positive)) / micro_denominator
        if micro_denominator > 0.0
        else 0.0
    )
    losses = single_label_task_losses(target_array, probability_array, labels=classes)
    one_hot = np.eye(classes.size, dtype=np.float64)[target_indices]
    return {
        "accuracy": float(np.mean(predicted == target_indices)),
        "macro_f1": float(np.mean(class_f1)),
        "micro_f1": micro_f1,
        "nll": float(np.mean(losses)),
        "brier": float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1))),
        "ece": top_class_ece(
            target_array, probability_array, labels=classes, bins=bins
        ),
    }


def _multilabel_counts(
    targets: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_positive = np.sum((targets == 1.0) & predictions, axis=0, dtype=np.float64)
    false_positive = np.sum((targets == 0.0) & predictions, axis=0, dtype=np.float64)
    false_negative = np.sum((targets == 1.0) & ~predictions, axis=0, dtype=np.float64)
    return true_positive, false_positive, false_negative


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def multilabel_metrics(
    targets: Any,
    probabilities: Any,
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> dict[str, float]:
    """Compute multilabel task metrics with explicit zero-division behavior."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    probability_array = _probability_matrix(
        probabilities,
        name="probabilities",
        from_logits=from_logits,
        multilabel=True,
    )
    target_array = _binary_targets(targets, shape=probability_array.shape)
    predicted = probability_array >= threshold
    true_positive, false_positive, false_negative = _multilabel_counts(
        target_array, predicted
    )
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    class_f1 = _safe_ratio(
        2.0 * true_positive,
        2.0 * true_positive + false_positive + false_negative,
    )
    total_true_positive = float(np.sum(true_positive))
    total_false_positive = float(np.sum(false_positive))
    total_false_negative = float(np.sum(false_negative))
    micro_precision_denominator = total_true_positive + total_false_positive
    micro_recall_denominator = total_true_positive + total_false_negative
    micro_f1_denominator = (
        2.0 * total_true_positive + total_false_positive + total_false_negative
    )
    losses = multilabel_task_losses(target_array, probability_array)
    pooled_ece, macro_ece = multilabel_calibration_error(
        target_array, probability_array
    )
    return {
        "micro_precision": (
            total_true_positive / micro_precision_denominator
            if micro_precision_denominator > 0.0
            else 0.0
        ),
        "micro_recall": (
            total_true_positive / micro_recall_denominator
            if micro_recall_denominator > 0.0
            else 0.0
        ),
        "micro_f1": (
            2.0 * total_true_positive / micro_f1_denominator
            if micro_f1_denominator > 0.0
            else 0.0
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(class_f1)),
        "subset_accuracy": float(np.mean(np.all(predicted == target_array, axis=1))),
        "bce": float(np.mean(losses)),
        "brier": float(np.mean((probability_array - target_array) ** 2)),
        "ece": pooled_ece,
        "macro_ece": macro_ece,
    }


def multilabel_calibration_error(
    targets: Any,
    probabilities: Any,
    *,
    bins: int = 15,
) -> tuple[float, float]:
    """Return pooled binary ECE and macro per-label ECE."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be an integer >= 1")
    probability_array = _probability_matrix(
        probabilities,
        name="probabilities",
        from_logits=False,
        multilabel=True,
    )
    target_array = _binary_targets(targets, shape=probability_array.shape)

    def binary_ece(expected: np.ndarray, confidence: np.ndarray) -> float:
        indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
        counts = np.bincount(indices, minlength=bins).astype(np.float64)
        confidence_sum = np.bincount(indices, weights=confidence, minlength=bins)
        positive_sum = np.bincount(indices, weights=expected, minlength=bins)
        total = float(np.sum(counts))
        return (
            float(np.sum(np.abs(confidence_sum - positive_sum)) / total)
            if total
            else float("nan")
        )

    pooled = binary_ece(target_array.ravel(), probability_array.ravel())
    per_label = [
        binary_ece(target_array[:, index], probability_array[:, index])
        for index in range(target_array.shape[1])
    ]
    return pooled, float(np.mean(per_label))


def select_multilabel_threshold(
    validation_targets: Any,
    validation_probabilities: Any,
    *,
    thresholds: Any | None = None,
    from_logits: bool = False,
) -> float:
    """Select one validation-only threshold maximizing multilabel micro-F1.

    Ties prefer the candidate nearest 0.5, then the lower candidate. When no
    candidate grid is supplied, every validation-score breakpoint is examined.
    """
    probabilities = _probability_matrix(
        validation_probabilities,
        name="validation_probabilities",
        from_logits=from_logits,
        multilabel=True,
    )
    targets = _binary_targets(validation_targets, shape=probabilities.shape)
    if thresholds is None:
        candidates = np.unique(
            np.concatenate((probabilities.ravel(), np.asarray([0.0, 0.5, 1.0])))
        )
    else:
        candidates = np.asarray(thresholds, dtype=np.float64)
        if candidates.ndim != 1 or candidates.size == 0:
            raise ValueError("thresholds must be a non-empty 1-D sequence")
        candidates = np.unique(candidates)
    if not np.all(np.isfinite(candidates)) or np.any(
        (candidates < 0.0) | (candidates > 1.0)
    ):
        raise ValueError("threshold candidates must be finite and in [0, 1]")

    flat_probabilities = probabilities.ravel()
    flat_targets = targets.ravel()
    order = np.argsort(-flat_probabilities, kind="stable")
    sorted_probabilities = flat_probabilities[order]
    cumulative_positive = np.cumsum(flat_targets[order], dtype=np.float64)
    total_positive = float(np.sum(flat_targets))
    predicted_counts = np.searchsorted(
        -sorted_probabilities,
        -candidates,
        side="right",
    )
    true_positive = np.where(
        predicted_counts > 0,
        cumulative_positive[np.maximum(predicted_counts - 1, 0)],
        0.0,
    )
    false_positive = predicted_counts - true_positive
    false_negative = total_positive - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0.0,
    )
    best_score = float(np.max(scores))
    tied = candidates[np.isclose(scores, best_score, rtol=1e-12, atol=1e-15)]
    distances = np.abs(tied - 0.5)
    nearest = tied[np.isclose(distances, np.min(distances), atol=1e-15, rtol=0.0)]
    return float(np.min(nearest))


def temperature_scale(logits: Any, temperature: float) -> np.ndarray:
    """Apply multiclass temperature scaling and return probabilities."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    logit_array = _float_matrix(logits, name="logits")
    return _softmax(logit_array / temperature)


def fit_temperature(
    validation_logits: Any,
    validation_targets: Any,
    *,
    labels: Any | None = None,
    min_temperature: float = 0.05,
    max_temperature: float = 100.0,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """Fit a scalar temperature on validation NLL by deterministic bisection."""
    logits = _float_matrix(validation_logits, name="validation_logits")
    targets = np.asarray(validation_targets)
    if targets.ndim != 1 or targets.size != logits.shape[0]:
        raise ValueError("one validation target is required for each logit row")
    classes = _classes_for_probabilities(targets, logits.shape[1], labels)
    target_indices = _target_indices(targets, classes)
    if (
        not np.isfinite(min_temperature)
        or not np.isfinite(max_temperature)
        or min_temperature <= 0.0
        or max_temperature < min_temperature
    ):
        raise ValueError("temperature bounds must be finite, positive and ordered")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be an integer >= 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and > 0")

    true_logits = logits[np.arange(logits.shape[0]), target_indices]

    def derivative(inverse_temperature: float) -> float:
        probabilities = _softmax(logits * inverse_temperature)
        expected_logits = np.sum(probabilities * logits, axis=1)
        return float(np.mean(expected_logits - true_logits))

    lower = 1.0 / max_temperature
    upper = 1.0 / min_temperature
    initial = float(np.clip(1.0, lower, upper))
    initial_derivative = derivative(initial)
    if abs(initial_derivative) <= tolerance:
        return 1.0 / initial
    lower_derivative = derivative(lower)
    upper_derivative = derivative(upper)
    if lower_derivative >= 0.0:
        return max_temperature
    if upper_derivative <= 0.0:
        return min_temperature
    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_derivative = derivative(midpoint)
        if abs(midpoint_derivative) <= tolerance:
            lower = upper = midpoint
            break
        if midpoint_derivative < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, midpoint):
            break
    return 1.0 / ((lower + upper) / 2.0)


@dataclass(frozen=True)
class MultilabelPlattCalibrator:
    """Independent affine-logistic calibrators, one per label."""

    slopes: np.ndarray
    intercepts: np.ndarray
    constant_labels: np.ndarray

    def __post_init__(self) -> None:
        slopes = _readonly_array(self.slopes)
        intercepts = _readonly_array(self.intercepts)
        constant_labels = _readonly_array(self.constant_labels, dtype=np.bool_)
        if (
            slopes.ndim != 1
            or intercepts.shape != slopes.shape
            or constant_labels.shape != slopes.shape
            or slopes.size == 0
        ):
            raise ValueError("Platt parameters must be equal non-empty vectors")
        if not np.all(np.isfinite(slopes)) or not np.all(np.isfinite(intercepts)):
            raise ValueError("Platt parameters must be finite")
        object.__setattr__(self, "slopes", slopes)
        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "constant_labels", constant_labels)

    def predict_proba(self, scores: Any) -> np.ndarray:
        score_array = _float_matrix(scores, name="scores")
        if score_array.shape[1] != self.slopes.size:
            raise ValueError(
                f"scores has {score_array.shape[1]} labels, expected {self.slopes.size}"
            )
        return _sigmoid(score_array * self.slopes + self.intercepts)


def _logistic_objective(
    scores: np.ndarray,
    targets: np.ndarray,
    slope: float,
    intercept: float,
    l2: float,
) -> float:
    logits = slope * scores + intercept
    losses = (
        np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    )
    return float(np.mean(losses) + 0.5 * l2 * slope**2)


def fit_multilabel_platt(
    validation_scores: Any,
    validation_targets: Any,
    *,
    l2: float = 1e-6,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> MultilabelPlattCalibrator:
    """Fit independent validation-only Platt calibrators safely per label.

    A label with only one observed class falls back to a finite Jeffreys-smoothed
    prevalence, represented by a zero slope.
    """
    scores = _float_matrix(validation_scores, name="validation_scores")
    targets = _binary_targets(validation_targets, shape=scores.shape)
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and >= 0")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be an integer >= 1")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and > 0")

    label_count = scores.shape[1]
    slopes = np.zeros(label_count, dtype=np.float64)
    intercepts = np.empty(label_count, dtype=np.float64)
    constant_labels = np.zeros(label_count, dtype=np.bool_)
    sample_count = scores.shape[0]
    for label_index in range(label_count):
        label_scores = scores[:, label_index]
        label_targets = targets[:, label_index]
        positive_count = float(np.sum(label_targets))
        smoothed_prevalence = (positive_count + 0.5) / (sample_count + 1.0)
        intercept = float(np.log(smoothed_prevalence / (1.0 - smoothed_prevalence)))
        if positive_count == 0.0 or positive_count == sample_count:
            intercepts[label_index] = intercept
            constant_labels[label_index] = True
            continue

        prevalence = positive_count / sample_count
        intercept = float(np.log(prevalence / (1.0 - prevalence)))
        slope = 0.0
        for _ in range(max_iterations):
            logits = slope * label_scores + intercept
            probabilities = _sigmoid(logits)
            residual = probabilities - label_targets
            gradient = np.asarray(
                [
                    np.mean(residual * label_scores) + l2 * slope,
                    np.mean(residual),
                ]
            )
            if float(np.max(np.abs(gradient))) <= tolerance:
                break
            weights = probabilities * (1.0 - probabilities)
            hessian = np.asarray(
                [
                    [
                        np.mean(weights * label_scores**2) + l2,
                        np.mean(weights * label_scores),
                    ],
                    [np.mean(weights * label_scores), np.mean(weights)],
                ]
            )
            hessian.flat[::3] += 1e-12
            try:
                step_direction = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step_direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            objective = _logistic_objective(
                label_scores, label_targets, slope, intercept, l2
            )
            directional_derivative = float(gradient @ step_direction)
            step_size = 1.0
            accepted = False
            while step_size >= 2.0**-20:
                candidate_slope = slope - step_size * float(step_direction[0])
                candidate_intercept = intercept - step_size * float(step_direction[1])
                candidate_objective = _logistic_objective(
                    label_scores,
                    label_targets,
                    candidate_slope,
                    candidate_intercept,
                    l2,
                )
                if candidate_objective <= objective - (
                    1e-4 * step_size * directional_derivative
                ):
                    slope = candidate_slope
                    intercept = candidate_intercept
                    accepted = True
                    break
                step_size /= 2.0
            if not accepted:
                break
            if float(np.max(np.abs(step_size * step_direction))) <= tolerance:
                break
        if not np.isfinite(slope) or not np.isfinite(intercept):
            slope = 0.0
            intercept = float(np.log(prevalence / (1.0 - prevalence)))
        slopes[label_index] = slope
        intercepts[label_index] = intercept
    return MultilabelPlattCalibrator(
        slopes=slopes,
        intercepts=intercepts,
        constant_labels=constant_labels,
    )


def apply_multilabel_platt(
    calibrator: MultilabelPlattCalibrator, scores: Any
) -> np.ndarray:
    """Apply independently fit Platt calibration to multilabel scores."""
    if not isinstance(calibrator, MultilabelPlattCalibrator):
        raise TypeError("calibrator must be a MultilabelPlattCalibrator")
    return calibrator.predict_proba(scores)


def classification_predictions(
    logits: jax.Array, positions: jax.Array, label_token_ids: jax.Array
) -> jax.Array:
    if logits.ndim != 3 or positions.ndim != 1 or label_token_ids.ndim != 2:
        raise ValueError("invalid logits, positions, or label-token shapes")
    if logits.shape[0] != positions.shape[0]:
        raise ValueError("one position is required for each example")
    if label_token_ids.shape[1] < 1:
        raise ValueError("each class must have at least one label token")
    scores = logits[jnp.arange(logits.shape[0]), positions][:, label_token_ids]
    return jnp.argmax(jnp.max(scores, axis=-1), axis=-1)


def calibration_stats(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-bin (conf_sum, acc_sum, count) for top-1 predictions.

    Uses half-open bins and segment summation so it jits without dynamic shapes
    and remains finite when some bins are empty.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = (target != -100).astype(jnp.float32)
    probs = jax.nn.softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    conf = jnp.max(probs, axis=-1)
    predicted = jnp.argmax(probs, axis=-1)
    safe_target = jnp.where(target != -100, target, 0)
    correct = (predicted == safe_target).astype(jnp.float32)

    idx = jnp.clip((conf * bins).astype(jnp.int32), 0, bins - 1)
    conf_flat = (conf * mask).ravel()
    acc_flat = (correct * mask).ravel()
    mask_flat = mask.ravel()
    idx_flat = idx.ravel()

    conf_sum = jax.ops.segment_sum(conf_flat, idx_flat, num_segments=bins)
    acc_sum = jax.ops.segment_sum(acc_flat, idx_flat, num_segments=bins)
    counts = jax.ops.segment_sum(mask_flat, idx_flat, num_segments=bins)
    return conf_sum, acc_sum, counts


def ece_from_stats(
    conf_sum: jax.Array, acc_sum: jax.Array, counts: jax.Array
) -> jax.Array:
    """Compute ECE from pooled per-bin calibration statistics.

    ECE = sum_b |conf_sum_b - acc_sum_b| / max(N, 1).
    Empty bins contribute 0 (since |0 - 0| = 0) and never produce NaN.
    """
    total = jnp.maximum(counts.sum(), 1.0)
    return jnp.abs(conf_sum - acc_sum).sum() / total


def expected_calibration_error(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> jax.Array:
    """Token-level ECE using confidence of the top-1 next-token prediction."""
    conf_sum, acc_sum, counts = calibration_stats(logits, labels, bins=bins)
    return ece_from_stats(conf_sum, acc_sum, counts)


def paired_deltas(
    values: dict[str, dict[int, float]], reference: str
) -> dict[str, dict[int, float]]:
    """Return method-minus-reference values paired by seed."""
    if reference not in values:
        raise KeyError(reference)
    return {
        method: {
            seed: float(seed_values[seed] - values[reference][seed])
            for seed in seed_values
            if seed in values[reference]
        }
        for method, seed_values in values.items()
        if method != reference
    }


def sample_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return sample mean and sample standard deviation."""
    if not values:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return mean, std


def _task_matrix(acc: Any, t: int) -> jax.Array:
    a = jnp.asarray(acc, dtype=jnp.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix A[t, j], got shape {a.shape}.")
    if not isinstance(t, int) or t < 0:
        raise ValueError(f"t must be an integer >= 0, got {t!r}.")
    if t >= min(a.shape):
        raise ValueError(f"t={t} is out of range for A with shape {a.shape}.")
    return a


def mean_seen_accuracy(acc: Any, t: int) -> jax.Array:
    a = _task_matrix(acc, t)
    seen = a[t, : t + 1]
    valid = jnp.isfinite(seen)
    count = jnp.sum(valid)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid, seen, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


def mean_forgetting(acc: Any, t: int, *, higher_is_better: bool = True) -> jax.Array:
    a = _task_matrix(acc, t)
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    x = a if higher_is_better else -a
    s = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    valid_history = (s >= j) & jnp.isfinite(x[:t, :t])
    peak = jnp.max(jnp.where(valid_history, x[:t, :t], -jnp.inf), axis=0)
    drop = peak - x[t, :t]
    valid_drop = jnp.isfinite(peak) & jnp.isfinite(x[t, :t])
    count = jnp.sum(valid_drop)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid_drop, drop, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


def accuracy_forgetting(accuracy_matrix: Any, t: int) -> jax.Array:
    return mean_forgetting(accuracy_matrix, t, higher_is_better=True)


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


__all__ = [
    "FrozenStandardScaler",
    "MultilabelPlattCalibrator",
    "RidgeProbe",
    "TopClassCalibrationStats",
    "accuracy_forgetting",
    "apply_multilabel_platt",
    "calibration_stats",
    "classification_metrics",
    "classification_predictions",
    "cross_entropy_loss",
    "ece_from_stats",
    "expected_calibration_error",
    "fit_frozen_scaler",
    "fit_multilabel_platt",
    "fit_ridge_probe",
    "fit_temperature",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_accuracy",
    "multilabel_metrics",
    "multilabel_label_losses",
    "multilabel_calibration_error",
    "multilabel_task_losses",
    "paired_deltas",
    "per_example_task_losses",
    "perplexity",
    "pool_top_class_calibration",
    "pooled_top_class_ece",
    "ridge_probe_probabilities",
    "ridge_probe_scores",
    "sample_mean_std",
    "select_multilabel_threshold",
    "single_label_metrics",
    "single_label_task_losses",
    "temperature_scale",
    "token_accuracy",
    "top_class_calibration_stats",
    "top_class_ece",
    "top_class_ece_from_stats",
    "training_step_time",
]
