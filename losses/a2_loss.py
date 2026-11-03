"""Feedforward A2 losses. No settling states or label-conditioned query updates.

Targets are always detached. Query detachment controls column gradients only;
it does not make a noncausal query safe. Callers must supply causal forward states.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from residual import AdapterParams, ResidualAdapter
from residual.cpc_heads import predict_future, project
from residual.fpc_energy import anchor_sq_distance, check_backend
from residual.stability import StabilityMonitor

_VAR_FLOOR = 1e-6
A2_LOSS_KEYS = ("task", "kl", "wd", "hid")
STABILITY_METRIC_KEYS = (
    "res_smooth",
    "input_smooth",
    "res_vs_input",
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
    cpc_detach_query: bool = True,
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
    if cpc_detach_query:
        anchors = jax.lax.stop_gradient(anchors)
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
    pred: jax.Array,
    target: jax.Array,
    source: jax.Array,
    config: Any,
    mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Single hidden-prediction implementation, including valid-token normalization."""
    if mask is None:
        mask = jnp.ones(target.shape[:-1], dtype=bool)
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    source = source.astype(jnp.float32)
    if config.skip_first_token:
        pred, target, source = pred[:, 1:], target[:, 1:], source[:, 1:]
        mask = mask[:, 1:]
    sq = jnp.square(pred - target)
    copy_sq = jnp.square(jax.lax.stop_gradient(source) - target)
    w = mask.astype(jnp.float32)[..., None]  # [B, S, 1]
    n = jnp.maximum(jnp.sum(w), 1.0)
    d = target.shape[-1]
    mse = jnp.sum(w * sq) / (n * d)
    copy_mse = jnp.sum(w * copy_sq) / (n * d)
    mean = jnp.sum(w * target, axis=(0, 1)) / n  # [D]
    var = jnp.mean(jnp.sum(w * jnp.square(target - mean), axis=(0, 1)) / n) + _VAR_FLOOR
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
    return dict(
        monitor.metrics(
            adapter,
            adapter_params,
            site_inputs,
            corrected=corrected,
            mask=mask,
        )
    )


# Direct clean-base regularization; no inference trajectory is involved.


def _masked_mean(x: jax.Array, mask: jax.Array | None) -> jax.Array:
    """Mean of ``x`` ``[..., B, T]`` over valid entries; ``mask`` is ``[B, T]``."""
    if mask is None:
        return jnp.mean(x)
    w = jnp.broadcast_to(mask.astype(jnp.float32), x.shape)
    return jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0)


def anchor_loss(
    corrected: jax.Array,
    base_state: jax.Array,
    *,
    normalize: str = "relative",
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Direct A2 base-state anchor: corrected state versus clean-base state.

    Here ``corrected`` is the residual-corrected state and ``base_state`` is the clean
    frozen-model state. Neither is a settled latent. A3 has a separate anchor.

    L = mean_{...,b,t} || corrected - sg(base_state) ||^2        (``normalize="none"``)
    L = that / ( mean || sg(base_state) ||^2 + eps )     (``normalize="relative"``)
    """
    if normalize not in ("relative", "none"):
        raise ValueError(f"normalize must be 'relative' or 'none', got {normalize!r}")
    check_backend(backend)
    if corrected.ndim < 3:
        raise ValueError(f"expected [..., B, T, d], got shape {corrected.shape}")
    if (
        base_state.ndim > corrected.ndim
        or base_state.shape != corrected.shape[corrected.ndim - base_state.ndim :]
    ):
        raise ValueError(
            f"base_state {base_state.shape} must equal corrected {corrected.shape} or its trailing axes"
        )
    corrected = corrected.astype(jnp.float32)
    anchor = jax.lax.stop_gradient(base_state.astype(jnp.float32))

    if backend == "fabricpc":
        sq = anchor_sq_distance(corrected, anchor)  # [..., B, T]
    else:
        sq = jnp.sum(jnp.square(corrected - anchor), axis=-1)
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


NORMALIZATIONS = ("relative", "none")


def _sq_norm(x: jax.Array, backend: str) -> jax.Array:
    """``||x||^2`` per token, ``[..., B, T]``; on ``"fabricpc"`` the energy of an
    ``IdentityNode`` whose target is clamped to 0 (so ``x`` carries the gradient)."""
    if backend == "fabricpc":
        return anchor_sq_distance(x, jnp.zeros_like(x))
    return jnp.sum(jnp.square(x), axis=-1)


def residual_smoothness_loss(
    correction: jax.Array,
    inputs: jax.Array,
    *,
    normalize: str = "relative",
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Residual-smoothness loss ``L_res-smooth``: neighbouring tokens get similar corrections.

    L = mean_{b, t>1} || R(h_t) - R(h_{t-1}) ||^2                         (``"none"``)
    L = that / ( mean_{b, t>1} || sg(h_t - h_{t-1}) ||^2 + eps )          (``"relative"``)

    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be 'relative' or 'none', got {normalize!r}")
    check_backend(backend)
    if correction.ndim < 3 or correction.shape != inputs.shape:
        raise ValueError(
            f"correction and inputs must share a shape [..., B, T, d], got "
            f"{correction.shape} and {inputs.shape}"
        )
    r = correction.astype(jnp.float32)
    h = jax.lax.stop_gradient(inputs.astype(jnp.float32))
    if r.shape[-2] < 2:
        zero = jnp.zeros((), jnp.float32) + 0.0 * jnp.sum(r)
        return zero, {"smooth_res": zero, "smooth_ratio": zero}
    pair = None if mask is None else (mask[..., 1:] & mask[..., :-1])
    res = _masked_mean(_sq_norm(r[..., 1:, :] - r[..., :-1, :], backend), pair)
    inp = _masked_mean(_sq_norm(h[..., 1:, :] - h[..., :-1, :], backend), pair)
    ratio = res / (inp + _EPS)
    loss = ratio if normalize == "relative" else res
    return loss, {"smooth_res": res, "smooth_ratio": ratio}


@dataclass(frozen=True)
class DetachConfig:
    """First-run gradient controls; targets are unconditionally detached."""

    hid_detach_input: bool = True
    cpc_detach_query: bool = True

    def __post_init__(self) -> None:
        for name in ("hid_detach_input", "cpc_detach_query"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


def hidden_prediction_term(bank, params, layer, source, target, config, mask=None):
    """Detach before the predictor, preserving gradients on its trainable weights."""
    pred = bank.predict(params, layer, source, detach_input=config.hid_detach_input)
    return hidden_prediction_loss(pred, target, source, config, mask=mask)


def drift_metrics(corrected, base, mask=None):
    """Measurement only: direct corrected-to-clean-base drift at each site."""
    parts = []
    metrics = {}
    for layer, value in corrected.items():
        delta = jax.lax.stop_gradient(value - base[layer]).astype(jnp.float32)
        distance = _masked_mean(jnp.linalg.norm(delta, axis=-1), mask)
        metrics[f"drift_l{layer}"] = distance
        parts.append(distance)
    metrics["drift_mean"] = jnp.mean(jnp.stack(parts))
    return metrics
