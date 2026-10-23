"""Representation metrics: loss improvement, low- vs high-rank residual compressibility,
and the internal metrics of the A2 plan (residual size, redundancy, gradient conflict)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp


def loss_improvement(
    base_loss: jax.Array | float, loss: jax.Array | float
) -> jax.Array:
    """Return ``L_base - L_adapted``; positive when the residual helps."""
    return jnp.asarray(base_loss, jnp.float32) - jnp.asarray(loss, jnp.float32)


def residual_compressibility(
    delta_small: jax.Array | float,
    delta_large: jax.Array | float,
    eps: float = 1e-8,
) -> jax.Array:
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    return jnp.asarray(delta_small, jnp.float32) / (
        jnp.asarray(delta_large, jnp.float32) + eps
    )


def compressibility_curve(
    deltas: Mapping[int, jax.Array | float], eps: float = 1e-8
) -> dict[int, jax.Array]:
    if not deltas:
        raise ValueError("deltas is empty; expected at least one rank.")
    ranks = sorted(deltas)
    if ranks[0] < 1:
        raise ValueError(f"Ranks must be >= 1, got {ranks[0]}.")
    reference = deltas[ranks[-1]]
    return {r: residual_compressibility(deltas[r], reference, eps) for r in ranks}


def residual_norm_ratio(
    h: jax.Array,
    h_corrected: jax.Array,
    mask: jax.Array | None = None,
    eps: float = 1e-8,
) -> jax.Array:
    """Mean over tokens of ``||h~ - h|| / ||h||``: how large the residual ``R`` is.
    """
    if h.shape != h_corrected.shape:
        raise ValueError(f"shape mismatch: {h.shape} vs {h_corrected.shape}")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    h32 = h.astype(jnp.float32)
    r_norm = jnp.linalg.norm(h_corrected.astype(jnp.float32) - h32, axis=-1)
    ratio = r_norm / (jnp.linalg.norm(h32, axis=-1) + eps)
    if mask is None:
        return jnp.mean(ratio)
    if mask.shape != ratio.shape:
        raise ValueError(f"mask shape {mask.shape} must be {ratio.shape}")
    w = mask.astype(jnp.float32)
    return jnp.sum(ratio * w) / jnp.maximum(jnp.sum(w), 1.0)


def direction_redundancy(vectors: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Mean ``|cos|`` over the pairs of distinct rows of ``vectors`` ``[n, d]``.
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D [n, d] matrix, got shape {vectors.shape}")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    n = vectors.shape[0]
    if n < 2:
        return jnp.zeros((), jnp.float32)
    v = vectors.astype(jnp.float32)
    unit = v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), eps)
    cos = jnp.abs(unit @ unit.T)
    off_diagonal = cos.sum() - jnp.trace(cos)
    return off_diagonal / (n * (n - 1))


def residual_redundancy(a: jax.Array, b: jax.Array) -> dict[str, jax.Array]:
    """Redundancy of one adapter block ``h~ = h + s * sigma(h A) B``.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape[::-1]:
        raise ValueError(f"expected A [d, r] and B [r, d], got {a.shape} and {b.shape}")
    return {"read": direction_redundancy(a.T), "write": direction_redundancy(b)}


def gradient_conflict(grad_a: Any, grad_b: Any, eps: float = 1e-12) -> jax.Array:
    """``[-cos(g_a, g_b)]_+`` of two gradient PyTrees with the same structure.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    leaves_a = jax.tree_util.tree_leaves(grad_a)
    leaves_b = jax.tree_util.tree_leaves(grad_b)
    if jax.tree_util.tree_structure(grad_a) != jax.tree_util.tree_structure(grad_b):
        raise ValueError("the two gradients must have the same structure")
    if not leaves_a:
        raise ValueError("gradients are empty")
    for x, y in zip(leaves_a, leaves_b, strict=True):
        if x.shape != y.shape:
            raise ValueError(f"leaf shape mismatch: {x.shape} vs {y.shape}")
    dot = sum(
        jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32))
        for x, y in zip(leaves_a, leaves_b, strict=True)
    )
    na = jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves_a))
    nb = jnp.sqrt(sum(jnp.sum(jnp.square(y.astype(jnp.float32))) for y in leaves_b))
    cos = dot / jnp.maximum(na * nb, eps)
    return jnp.maximum(-cos, 0.0)


__all__ = [
    "compressibility_curve",
    "direction_redundancy",
    "gradient_conflict",
    "loss_improvement",
    "residual_compressibility",
    "residual_norm_ratio",
    "residual_redundancy",
]