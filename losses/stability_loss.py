"""A3-only settling penalties. Never imported by the A2 objective or runner.

Each site is reduced over valid tokens, features and inference steps before
averaging sites. The feedforward reference is fixed for the entire trajectory.
"""

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from residual.fpc_energy import check_backend, highway_sq_norm
from residual.hep import Highway, apply_highway

_EPS = 1e-8


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
