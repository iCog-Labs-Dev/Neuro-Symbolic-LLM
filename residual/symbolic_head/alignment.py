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
    """Average single-positive losses over labelled examples.

    Key loss uses every bank key in the softmax denominator, without cosine
    normalization or temperature scaling. Value loss sums squared errors over
    value dimensions before averaging examples. Bank vectors are fixed targets.

    Each row requires one source-derived label in [0, bank_size). Labels must
    not be inferred from retrieval rank. Out-of-range labels yield NaN under
    eager execution and JIT; callers must reject them before optimizer updates.
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
    if target_indices.shape != (queries.shape[0],) or not jnp.issubdtype(
        target_indices.dtype, jnp.integer
    ):
        raise ValueError("Each example requires one integer target index")
    if any(
        not jnp.issubdtype(array.dtype, jnp.floating)
        for array in (queries, predicted_values, bank_keys, bank_values)
    ):
        raise TypeError("Queries, predictions and bank vectors must be floating point")

    keys = jax.lax.stop_gradient(bank_keys.astype(jnp.float32))
    values = jax.lax.stop_gradient(bank_values.astype(jnp.float32))
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


__all__ = ["AlignmentLoss", "ValuePredictor", "alignment_loss"]
