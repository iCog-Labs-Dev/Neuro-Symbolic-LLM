"""Task-performance, drift, continual-learning and compute metrics.

Continual-learning metrics take ``A[t, j]``: the metric on task ``j`` after
training task ``t`` (0-based, training order).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from frozenllm.substrate import FrozenSubstrate

T = TypeVar("T")


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return FrozenSubstrate.compute_loss(logits, labels)


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    # float32 log-space for stability; sum over vocab, mean over batch and positions.
    log_p = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(adapted_logits.astype(jnp.float32), axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))


def perplexity(loss: jax.Array | float) -> jax.Array:
    return jnp.exp(jnp.asarray(loss, dtype=jnp.float32))


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
    return jnp.mean(a[t, : t + 1])


def mean_forgetting(acc: Any, t: int, *, higher_is_better: bool = True) -> jax.Array:
    a = _task_matrix(acc, t)
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    # Negate losses so "best" is always the max. The peak spans s >= j only
    # (Chaudhry et al., 2018), so negative transfer is not counted as forgetting.
    x = a if higher_is_better else -a
    s = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    peak = jnp.max(jnp.where(s >= j, x[:t, :t], -jnp.inf), axis=0)
    return jnp.mean(peak - x[t, :t])


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    # JAX is async: wait for the outputs before stopping the timer.
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


__all__ = [
    "cross_entropy_loss",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_accuracy",
    "perplexity",
    "training_step_time",
]
