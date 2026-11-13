"""Language-model, continual-learning, and compute metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal cross-entropy loss with next-token shifting and padding masking."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    mask = shift_labels != -100
    safe_labels = jnp.where(mask, shift_labels, 0)

    gathered_log_probs = jnp.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1
    ).squeeze(-1)

    return -jnp.sum(gathered_log_probs * mask) / jnp.maximum(1.0, jnp.sum(mask))


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    """Mean token-level KL divergence from base to adapted distributions."""
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


def calibration_stats(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-bin confidence, accuracy, and count sums for LM tokens."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = (target != -100).astype(jnp.float32)
    probs = jax.nn.softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    confidence = jnp.max(probs, axis=-1)
    predicted = jnp.argmax(probs, axis=-1)
    safe_target = jnp.where(target != -100, target, 0)
    correct = (predicted == safe_target).astype(jnp.float32)

    indices = jnp.clip((confidence * bins).astype(jnp.int32), 0, bins - 1)
    confidence_sum = jax.ops.segment_sum(
        (confidence * mask).ravel(), indices.ravel(), num_segments=bins
    )
    accuracy_sum = jax.ops.segment_sum(
        (correct * mask).ravel(), indices.ravel(), num_segments=bins
    )
    counts = jax.ops.segment_sum(mask.ravel(), indices.ravel(), num_segments=bins)
    return confidence_sum, accuracy_sum, counts


def ece_from_stats(
    confidence_sum: jax.Array, accuracy_sum: jax.Array, counts: jax.Array
) -> jax.Array:
    """Compute token ECE from pooled per-bin sufficient statistics."""
    total = jnp.maximum(counts.sum(), 1.0)
    return jnp.abs(confidence_sum - accuracy_sum).sum() / total


def expected_calibration_error(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> jax.Array:
    """Token-level ECE using confidence of the top-1 next-token prediction."""
    confidence_sum, accuracy_sum, counts = calibration_stats(logits, labels, bins=bins)
    return ece_from_stats(confidence_sum, accuracy_sum, counts)


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


def _loss_matrix(values: Any, t: int) -> jax.Array:
    loss = jnp.asarray(values, dtype=jnp.float32)
    if loss.ndim != 2:
        raise ValueError(f"Expected a 2-D loss matrix L[t, j], got shape {loss.shape}.")
    if not isinstance(t, int) or t < 0:
        raise ValueError(f"t must be an integer >= 0, got {t!r}.")
    if t >= min(loss.shape):
        raise ValueError(f"t={t} is out of range for L with shape {loss.shape}.")
    return loss


def mean_seen_loss(values: Any, t: int) -> jax.Array:
    """Return mean loss on tasks seen through training step ``t``."""
    loss = _loss_matrix(values, t)
    seen = loss[t, : t + 1]
    valid = jnp.isfinite(seen)
    count = jnp.sum(valid)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid, seen, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


def mean_forgetting(values: Any, t: int) -> jax.Array:
    """Return mean increase from each prior task's best observed loss."""
    loss = _loss_matrix(values, t)
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    training_step = jnp.arange(t)[:, None]
    task = jnp.arange(t)[None, :]
    valid_history = (training_step >= task) & jnp.isfinite(loss[:t, :t])
    best = jnp.min(jnp.where(valid_history, loss[:t, :t], jnp.inf), axis=0)
    increase = loss[t, :t] - best
    valid_increase = jnp.isfinite(best) & jnp.isfinite(loss[t, :t])
    count = jnp.sum(valid_increase)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid_increase, increase, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


__all__ = [
    "calibration_stats",
    "cross_entropy_loss",
    "ece_from_stats",
    "expected_calibration_error",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_loss",
    "paired_deltas",
    "perplexity",
    "sample_mean_std",
    "token_accuracy",
    "training_step_time",
]
