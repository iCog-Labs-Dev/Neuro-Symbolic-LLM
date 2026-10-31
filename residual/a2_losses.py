"""auxiliary losses on residual-corrected hidden states."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from residual import AdapterParams, ResidualAdapter
from residual.cpc_heads import predict_future, project
from residual.fpc_energy import anchor_sq_distance, check_backend, highway_sq_norm
from residual.hep import Highway, apply_highway
from residual.stability import (
    STEP_KEYS,
    StabilityMonitor,
    residual_smoothness,
    step_metrics,
)

_VAR_FLOOR = 1e-6
A2_LOSS_KEYS = ("task", "kl", "wd", "hid")
STABILITY_METRIC_KEYS = (
    "res_smooth",
    "input_smooth",
    "res_vs_input",
    *STEP_KEYS,
)

_NEG = -1e30
_EPS = 1e-8


def _l2_normalize(x: jax.Array) -> jax.Array:
    return x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + _EPS)


def anchor_positions(num_positions: int, max_anchors: int | None) -> tuple[int, ...]:
    """Deterministic anchor subsampling: every ``stride``-th position from 0."""
    if num_positions < 1:
        raise ValueError(f"num_positions must be >= 1, got {num_positions}")
    if max_anchors is None:
        return tuple(range(num_positions))
    if max_anchors < 1:
        raise ValueError(f"max_anchors must be >= 1 or None, got {max_anchors}")
    stride = math.ceil(num_positions / max_anchors)
    return tuple(range(0, num_positions, stride))


def contrastive_future_loss(
    q_params: Mapping[str, jax.Array],
    a_params: Mapping[str, jax.Array],
    anchors: jax.Array,
    targets: jax.Array,
    *,
    offset: int = 1,
    tau: float = 0.1,
    max_anchors_per_seq: int | None = None,
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """InfoNCE between the predicted future code and the true future code.

    For a sequence ``b`` and position ``t`` (``t + offset < T``):

        anchor     z_{b,t} = normalize( A( q(anchors[b, t]) ) )
        positive   y_{b,t} = normalize( stop_gradient( q(targets[b, t + offset]) ) )

        L = mean_{b,t}  -log  exp(z.y_pos / tau) / ( exp(z.y_pos / tau) + sum_neg exp(z.y_neg / tau) )

    """
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        raise ValueError(f"offset must be an integer >= 1, got {offset!r}")
    if not math.isfinite(tau) or not tau > 0:
        raise ValueError(f"tau must be > 0, got {tau!r}")
    if anchors.shape != targets.shape:
        raise ValueError(
            f"anchors {anchors.shape} and targets {targets.shape} must have equal shapes"
        )
    if anchors.ndim < 3:
        raise ValueError(f"expected [..., B, T, d], got shape {anchors.shape}")
    *lead, batch, seq_len, _ = anchors.shape
    if offset >= seq_len:
        raise ValueError(f"offset={offset} must be < seq_len={seq_len}")
    steps = math.prod(lead)
    if mask is not None and (
        mask.shape[-2:] != (batch, seq_len) or mask.shape[:-2] not in ((), tuple(lead))
    ):
        raise ValueError(
            f"mask must have shape [B, T] = {(batch, seq_len)} or the shape of the "
            f"anchors without the last axis {anchors.shape[:-1]}, got {mask.shape}"
        )

    positions = jnp.asarray(anchor_positions(seq_len - offset, max_anchors_per_seq))
    n_pos = int(positions.shape[0])

    # [S, B, T, d]: leading axes folded together.
    anchors = anchors.reshape(steps, batch, seq_len, -1)
    targets = jax.lax.stop_gradient(targets.reshape(steps, batch, seq_len, -1))

    z = _l2_normalize(
        predict_future(
            a_params, project(q_params, anchors[:, :, positions], backend), backend
        )
    )
    y = _l2_normalize(
        jax.lax.stop_gradient(
            project(q_params, targets[:, :, positions + offset], backend)
        )
    )
    m = z.shape[-1]
    z = z.reshape(-1, m)
    y = y.reshape(-1, m)
    n = z.shape[0]

    # All entries sharing a sequence index (any position, any leading sample) are non-negatives.
    seq_id = jnp.broadcast_to(
        jnp.arange(batch)[None, :, None], (steps, batch, n_pos)
    ).reshape(-1)
    zero = jnp.zeros((), jnp.float32)
    if batch == 1:
        return zero, {
            "cpc_top1": zero,
            "cpc_pos_sim": zero,
            "cpc_neg_sim": zero,
            "cpc_n_anchors": zero,
            "cpc_valid": zero,
        }

    if mask is None:
        valid = jnp.ones((n,), dtype=bool)
    else:
        token_ok = mask.astype(bool)
        m = (
            jnp.broadcast_to(token_ok, (steps, batch, seq_len))
            if token_ok.ndim == 2
            else token_ok.reshape(steps, batch, seq_len)
        )
        valid = (m[:, :, positions] & m[:, :, positions + offset]).reshape(-1)

    sims = z @ y.T  # cosine similarities, [N, N]
    eye = jnp.eye(n, dtype=bool)
    negative = (seq_id[:, None] != seq_id[None, :]) & valid[None, :] & valid[:, None]
    candidate = negative | eye

    logits = jnp.where(candidate, sims / tau, _NEG)
    pos_logit = jnp.diagonal(logits)
    per_row = jax.nn.logsumexp(logits, axis=-1) - pos_logit

    # Rows that count: a real anchor with at least one real negative.
    row_ok = valid & jnp.any(negative, axis=-1)
    w = row_ok.astype(jnp.float32)
    n_ok = jnp.sum(w)
    denom = jnp.maximum(n_ok, 1.0)
    loss = jnp.sum(jnp.where(row_ok, per_row, 0.0)) / denom

    top1 = (jnp.argmax(logits, axis=-1) == jnp.arange(n)).astype(jnp.float32)
    metrics = {
        "cpc_top1": jnp.sum(top1 * w) / denom,
        "cpc_pos_sim": jnp.sum(jnp.diagonal(sims) * w) / denom,
        "cpc_neg_sim": jnp.sum(jnp.where(negative & row_ok[:, None], sims, 0.0))
        / jnp.maximum(jnp.sum(negative & row_ok[:, None]), 1),
        "cpc_n_anchors": n_ok,
        "cpc_valid": (n_ok > 0).astype(jnp.float32),
    }
    return loss.astype(jnp.float32), metrics


def hidden_prediction_loss(
    pred: jax.Array, target: jax.Array, source: jax.Array, config: Any
) -> dict[str, jax.Array]:
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    source = source.astype(jnp.float32)
    if config.skip_first_token:
        pred, target, source = pred[:, 1:], target[:, 1:], source[:, 1:]
    if pred.size == 0:
        zero = jnp.sum(pred)  # finite empty sum, with a zero derivative
        return dict.fromkeys(("loss", "mse", "rel", "copy_rel"), zero)
    mse = jnp.mean(jnp.square(pred - target))
    var = jnp.mean(jnp.var(target, axis=(0, 1))) + _VAR_FLOOR
    copy_mse = jnp.mean(jnp.square(jax.lax.stop_gradient(source) - target))
    rel = mse / var
    return {
        "loss": rel if config.hidden_norm == "target_var" else mse,
        "mse": mse,
        "rel": rel,
        "copy_rel": copy_mse / var,
    }


def collect_stability_metrics(
    monitor: StabilityMonitor,
    adapter: ResidualAdapter,
    adapter_params: AdapterParams,
    site_inputs: Mapping[int, jax.Array],
    corrected: Mapping[int, jax.Array] | None = None,
    mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Collect A2 stability metrics without adding them to the loss."""
    return monitor.metrics(
        adapter,
        adapter_params,
        site_inputs,
        corrected=corrected,
        mask=mask,
    )


# --- stability: anchor loss and HEP nudge


def _masked_mean(x: jax.Array, mask: jax.Array | None) -> jax.Array:
    """Mean of ``x`` ``[..., B, T]`` over valid entries; ``mask`` is ``[B, T]``."""
    if mask is None:
        return jnp.mean(x)
    w = jnp.broadcast_to(mask.astype(jnp.float32), x.shape)
    return jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0)


def anchor_loss(
    z: jax.Array,
    z_anchor: jax.Array,
    *,
    normalize: str = "relative",
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Anchor loss ``L_zanch``: keep the state ``z`` close to its feed-forward anchor.

    L = mean_{...,b,t} || z - sg(z_anchor) ||^2        (``normalize="none"``)
    L = that / ( mean || sg(z_anchor) ||^2 + eps )     (``normalize="relative"``)
    """
    if normalize not in ("relative", "none"):
        raise ValueError(f"normalize must be 'relative' or 'none', got {normalize!r}")
    check_backend(backend)
    if z.ndim < 3:
        raise ValueError(f"expected [..., B, T, d], got shape {z.shape}")
    if z_anchor.ndim > z.ndim or z_anchor.shape != z.shape[z.ndim - z_anchor.ndim :]:
        raise ValueError(
            f"z_anchor {z_anchor.shape} must equal z {z.shape} or its trailing axes"
        )
    z = z.astype(jnp.float32)
    anchor = jax.lax.stop_gradient(z_anchor.astype(jnp.float32))

    if backend == "fabricpc":
        sq = anchor_sq_distance(z, anchor)  # [..., B, T]
    else:
        sq = jnp.sum(jnp.square(z - anchor), axis=-1)
    anchor_sq = jnp.broadcast_to(jnp.sum(jnp.square(anchor), axis=-1), sq.shape)
    mean_sq = _masked_mean(sq, mask)
    mean_anchor_sq = _masked_mean(anchor_sq, mask)
    loss = mean_sq
    if normalize == "relative":
        loss = mean_sq / (mean_anchor_sq + _EPS)
    metrics = {
        "anch_dist": _masked_mean(jnp.sqrt(sq + _EPS) - jnp.sqrt(_EPS), mask),
        "anch_ratio": jnp.sqrt(mean_sq / (mean_anchor_sq + _EPS)),
    }
    return loss, metrics


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
    """Projected error ``alpha * sg(eps) @ V``, ``[..., d_i]``; A2 does not inject it."""
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
    """Projected-error norm-only control; A2 applies no inference nudge.

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
    mean_sq = _masked_mean(sq, mask)
    metrics = {
        "hep_nudge_norm": _masked_mean(jnp.sqrt(sq + _EPS) - jnp.sqrt(_EPS), mask)
    }
    loss = mean_sq
    if reference is not None:
        ref = jax.lax.stop_gradient(reference.astype(jnp.float32))
        ref_sq = _masked_mean(jnp.sum(jnp.square(ref), axis=-1), mask)
        loss = mean_sq / (ref_sq + _EPS)
        metrics["hep_nudge_ratio"] = jnp.sqrt(mean_sq / (ref_sq + _EPS))
    return loss, metrics


__all__ = [
    "A2_LOSS_KEYS",
    "anchor_loss",
    "anchor_positions",
    "contrastive_future_loss",
    "collect_stability_metrics",
    "hep_nudge",
    "hep_nudge_loss",
    "hidden_prediction_loss",
    "output_error",
    "residual_smoothness",
    "STABILITY_METRIC_KEYS",
    "step_metrics",
]
