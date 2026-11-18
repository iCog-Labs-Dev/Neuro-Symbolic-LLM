"""Full-bank key alignment and independent hidden-to-value prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AlignmentLoss(NamedTuple):
    total: jax.Array
    key: jax.Array
    value: jax.Array


class ValuePredictor:
    """Linear value prediction, separate from query and writeback parameters.

    Row-vector equation: predicted_value = hidden @ V_pred, with V_pred[d, k].
    This predictor is used for alignment, not residual writeback or retrieval.
    """

    def __init__(self, hidden_size: int, value_dim: int) -> None:
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in (hidden_size, value_dim)
        ):
            raise ValueError("Hidden and value dimensions must be positive integers")
        self.hidden_size = hidden_size
        self.value_dim = value_dim

    def init_params(self, seed: int = 0) -> dict[str, jax.Array]:
        """Initialize an independent matrix with variance 1 / hidden_size."""
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x56414C)
        return {
            "V_pred": jax.random.normal(
                key, (self.hidden_size, self.value_dim), dtype=jnp.float32
            )
            / math.sqrt(self.hidden_size)
        }

    def apply(self, params: Mapping[str, jax.Array], hidden: jax.Array) -> jax.Array:
        if hidden.ndim < 1 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Hidden state must have last dimension {self.hidden_size}"
            )
        if set(params) != {"V_pred"} or params["V_pred"].shape != (
            self.hidden_size,
            self.value_dim,
        ):
            raise ValueError("Value predictor parameter keys or shapes are invalid")
        return hidden.astype(jnp.float32) @ params["V_pred"]


def alignment_loss(
    queries: jax.Array,
    predicted_values: jax.Array,
    bank_keys: jax.Array,
    bank_values: jax.Array,
    target_indices: jax.Array,
) -> AlignmentLoss:
    """Average alignment losses over labelled queries with fixed bank targets.

    One-dimensional targets retain the single-positive full-bank softmax loss.
    Two-dimensional targets are known-positive sets padded with -1: each positive
    competes against unlabelled bank entries, not other known positives. Scores
    are raw dot products without temperature scaling. Value loss sums squared
    errors over dimensions, then averages positives per query and queries.

    Labels must be source-derived, not inferred from retrieval rank. Empty,
    duplicate or out-of-range target sets yield NaN under eager execution and
    JIT; callers must reject them before optimizer updates.
    """
    if (
        queries.ndim != 2
        or predicted_values.shape != queries.shape
        or bank_keys.ndim != 2
        or bank_values.shape != bank_keys.shape
        or bank_keys.shape[1] != queries.shape[1]
        or queries.shape[0] == 0
        or queries.shape[1] == 0
        or bank_keys.shape[0] == 0
    ):
        raise ValueError("Queries, predictions and nonempty bank K/V shapes differ")
    if (
        target_indices.ndim not in (1, 2)
        or target_indices.shape[0] != queries.shape[0]
        or (target_indices.ndim == 2 and target_indices.shape[1] == 0)
    ) or not jnp.issubdtype(target_indices.dtype, jnp.integer):
        raise ValueError("Each example requires an integer target or padded target set")
    if any(
        not jnp.issubdtype(array.dtype, jnp.floating)
        for array in (queries, predicted_values, bank_keys, bank_values)
    ):
        raise TypeError("Queries, predictions and bank vectors must be floating point")

    keys = jax.lax.stop_gradient(bank_keys.astype(jnp.float32))
    values = jax.lax.stop_gradient(bank_values.astype(jnp.float32))
    if target_indices.ndim == 2:
        return _positive_set_loss(
            queries, predicted_values, keys, values, target_indices
        )
    valid = (target_indices >= 0) & (target_indices < keys.shape[0])
    safe_indices = jnp.clip(target_indices, 0, keys.shape[0] - 1)
    logits = queries.astype(jnp.float32) @ keys.T
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    key_errors = -log_probs[jnp.arange(queries.shape[0]), safe_indices]
    value_errors = jnp.sum(
        jnp.square(predicted_values.astype(jnp.float32) - values[safe_indices]),
        axis=-1,
    )
    key_loss = jnp.mean(jnp.where(valid, key_errors, jnp.nan))
    value_loss = jnp.mean(jnp.where(valid, value_errors, jnp.nan))
    return AlignmentLoss(key_loss + value_loss, key_loss, value_loss)


def _positive_set_loss(queries, predicted_values, keys, values, targets):
    """Full-bank positive-versus-unlabelled ranking; not certified negatives."""
    present = targets >= 0
    safe = jnp.clip(targets, 0, keys.shape[0] - 1)
    count = present.sum(axis=1)
    positives = jnp.zeros((queries.shape[0], keys.shape[0]), dtype=jnp.int32)
    positives = (
        positives.at[jnp.arange(queries.shape[0])[:, None], safe].add(
            present.astype(jnp.int32)
        )
        > 0
    )
    valid = (
        jnp.all((targets == -1) | (present & (targets < keys.shape[0])), axis=1)
        & (count > 0)
        & (positives.sum(axis=1) == count)
    )
    logits = queries.astype(jnp.float32) @ keys.T
    has_other = jnp.any(~positives, axis=1)
    # Avoid an all-minus-infinity logsumexp and its undefined derivative.
    other = (~positives) | (~has_other[:, None] & (jnp.arange(keys.shape[0]) == 0))
    normalizer = jax.nn.logsumexp(jnp.where(other, logits, -jnp.inf), axis=1)
    positive_logits = jnp.take_along_axis(logits, safe, axis=1)
    errors = jax.nn.softplus(normalizer[:, None] - positive_logits)
    errors = jnp.where(present & has_other[:, None], errors, 0)
    divisor = jnp.maximum(count, 1)
    key_errors = errors.sum(axis=1) / divisor
    distances = jnp.sum(
        jnp.square(predicted_values[:, None].astype(jnp.float32) - values[safe]),
        axis=-1,
    )
    value_errors = jnp.where(present, distances, 0).sum(axis=1) / divisor
    key_loss = jnp.mean(jnp.where(valid, key_errors, jnp.nan))
    value_loss = jnp.mean(jnp.where(valid, value_errors, jnp.nan))
    return AlignmentLoss(key_loss + value_loss, key_loss, value_loss)


__all__ = ["AlignmentLoss", "ValuePredictor", "alignment_loss"]
