"""Representation metrics and A3 settling penalties.

A2 reporting uses residual size, compressibility, redundancy and gradient conflict.
The settling and HEP helpers remain differentiable for future A3 objectives;
relocating them here does not add them to A2 training. Settling references are
detached and fixed, with reductions over valid tokens, features, steps and sites.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from residual.hep import Highway

_EPS = 1e-8


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
    """Mean over tokens of ``||h~ - h|| / ||h||``: how large the residual ``R`` is."""
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
    """Mean ``|cos|`` over the pairs of distinct rows of ``vectors`` ``[n, d]``."""
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
    """Redundancy of one adapter block ``h~ = h + s * sigma(h A) B``."""
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape[::-1]:
        raise ValueError(f"expected A [d, r] and B [r, d], got {a.shape} and {b.shape}")
    return {"read": direction_redundancy(a.T), "write": direction_redundancy(b)}


def gradient_conflict(grad_a: Any, grad_b: Any, eps: float = 1e-12) -> jax.Array:
    """``[-cos(g_a, g_b)]_+`` of two gradient PyTrees with the same structure."""
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


def _masked_mean(x, mask=None):
    if x.size == 0:
        return jnp.sum(x)
    w = (
        jnp.ones_like(x, dtype=bool)
        if mask is None
        else jnp.broadcast_to(mask.astype(bool), x.shape)
    )
    return jnp.sum(jnp.where(w, x, 0.0)) / jnp.maximum(jnp.sum(w), 1.0)


def v_hidden_size(v):
    return v.shape[-1] if hasattr(v, "shape") else v[1].shape[-1]


def settling_loss(
    trajectories: Mapping[int, jax.Array],
    feedforward: Mapping[int, jax.Array],
    *,
    mask=None,
    anchor_weight=1.0,
    step_weight=0.0,
    nudges: Mapping[int, jax.Array] | None = None,
    nudge_weight=0.0,
):
    """Trajectories are [K+1,B,T,D]; nudges are [K,B,T,D].

    ``feedforward[layer]`` is [B,T,D], captured before inference, never
    recomputed from a settled state. Padding, D, K and site counts are averaged.
    Nudge penalties alone do not define how inference applies a nudge.
    """
    if not trajectories or set(trajectories) != set(feedforward):
        raise ValueError(
            "trajectories and feedforward must have the same nonempty sites"
        )
    if any(
        not math.isfinite(w) or w < 0
        for w in (anchor_weight, step_weight, nudge_weight)
    ):
        raise ValueError("settling weights must be finite and nonnegative")
    parts = []
    for layer, trajectory in trajectories.items():
        z = trajectory.astype(jnp.float32)
        reference = jax.lax.stop_gradient(feedforward[layer].astype(jnp.float32))
        if z.ndim != 4 or z.shape[0] < 1 or z.shape[1:] != reference.shape:
            raise ValueError("expected [K+1,B,T,D] and a fixed [B,T,D] reference")
        if mask is not None and mask.shape != reference.shape[:-1]:
            raise ValueError("mask must be [B,T]")
        anchor = _masked_mean(jnp.mean((z[1:] - reference) ** 2, axis=-1), mask)
        step = _masked_mean(jnp.mean((z[1:] - z[:-1]) ** 2, axis=-1), mask)
        nudge = jnp.zeros((), jnp.float32)
        if nudges is not None:
            value = nudges[layer]
            if value.shape != z[1:].shape:
                raise ValueError("nudges must have shape [K,B,T,D]")
            nudge = _masked_mean(jnp.mean(value**2, axis=-1), mask)
        elif nudge_weight > 0:
            raise ValueError("nudge_weight > 0 requires nudges")
        parts.append(jnp.stack((anchor, step, nudge)))
    anchor, step, nudge = jnp.mean(jnp.stack(parts), axis=0)
    total = anchor_weight * anchor + step_weight * step + nudge_weight * nudge
    return total, {
        "settling_anchor": anchor,
        "settling_step": step,
        "settling_nudge": nudge,
    }


def output_error(logits: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Output error ``eps_y = softmax(logits) - onehot(next token)``."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"expected logits [B, T, V] and labels [B, T], got {logits.shape}, {labels.shape}"
        )
    p = jax.nn.softmax(
        jax.lax.stop_gradient(logits[:, :-1]).astype(jnp.float32), axis=-1
    )
    target = labels[:, 1:]
    mask = target != -100
    onehot = jax.nn.one_hot(
        jnp.where(mask, target, 0), logits.shape[-1], dtype=jnp.float32
    )
    eps = (p - onehot) * mask[..., None]
    return eps, mask


def hep_nudge(eps: jax.Array, v: Highway, alpha: float = 1.0) -> jax.Array:
    """Projected output error, for use by an A3 inference implementation."""
    from residual.hep import apply_highway

    return alpha * apply_highway(v, jax.lax.stop_gradient(eps))


def hep_nudge_loss(
    eps: jax.Array,
    v: Highway,
    alpha: float = 1.0,
    *,
    reference: jax.Array | None = None,
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Projected-error norm penalty for A3.

    Only trainable highway weights receive a gradient. A zero highway minimizes
    this penalty, so reducing it does not demonstrate useful HEP learning.

    L = mean_{...,b,t} || alpha * sg(eps) @ V ||^2
    L = that / ( mean || sg(reference) ||^2 + eps )
    """
    # Keep FabricPC imports out of ordinary representation-metric imports.
    from residual.fpc_energy import check_backend, highway_sq_norm

    if eps.ndim < 3:
        raise ValueError(f"expected eps [..., B, T, d_y], got shape {eps.shape}")
    if not alpha >= 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")
    check_backend(backend)
    if reference is not None and reference.shape[:-1] != eps.shape[:-1]:
        raise ValueError(
            f"reference {reference.shape} must match the nudge positions {eps.shape[:-1]}"
        )
    if backend == "fabricpc":
        # ||alpha * sg(eps) @ V||^2 = ||(alpha * sg(eps)) @ V||^2
        sq = highway_sq_norm(alpha * jax.lax.stop_gradient(eps), v)  # [..., B, T]
    else:
        sq = jnp.sum(jnp.square(hep_nudge(eps, v, alpha)), axis=-1)
    mean_sq = _masked_mean(sq, mask) / v_hidden_size(v)
    metrics = {
        "hep_nudge_norm": _masked_mean(jnp.sqrt(sq + _EPS) - jnp.sqrt(_EPS), mask)
    }
    loss = mean_sq
    if reference is not None:
        ref = jax.lax.stop_gradient(reference.astype(jnp.float32))
        ref_sq = _masked_mean(jnp.mean(jnp.square(ref), axis=-1), mask)
        loss = mean_sq / (ref_sq + _EPS)
        metrics["hep_nudge_ratio"] = jnp.sqrt(mean_sq / (ref_sq + _EPS))
    return loss, metrics


__all__ = [
    "compressibility_curve",
    "direction_redundancy",
    "gradient_conflict",
    "hep_nudge",
    "hep_nudge_loss",
    "loss_improvement",
    "output_error",
    "residual_compressibility",
    "residual_norm_ratio",
    "residual_redundancy",
    "settling_loss",
    "v_hidden_size",
]
