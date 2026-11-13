"""A4 cross-entropy and full-bank key/value alignment at fixed latents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from residual.symbolic_head.alignment import ValuePredictor, alignment_loss
from residual.symbolic_head.jax_head import HeadParams
from stages.stage_A.a4.model import A4Model


class AlignmentLabels(NamedTuple):
    """One bank target per labelled [batch, token] position at a head layer."""

    positions: jax.Array
    template_indices: jax.Array


def _task_terms_at_latents(
    model: A4Model,
    continuous_params: Any,
    symbolic_params: HeadParams,
    input_ids: jax.Array,
    z_star: Mapping[int, jax.Array],
    *,
    symbolic_enabled: bool = True,
    token_mask: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], dict[int, jax.Array]]:
    z_fixed = jax.tree.map(jax.lax.stop_gradient, z_star)
    continuous_fixed = jax.tree.map(jax.lax.stop_gradient, continuous_params)
    result, states = model.forward_with_states_at_latents(
        continuous_fixed,
        symbolic_params,
        input_ids,
        z_fixed,
        symbolic_enabled=symbolic_enabled,
    )
    labels = input_ids
    if token_mask is not None:
        if token_mask.shape != input_ids.shape or token_mask.dtype != jnp.bool_:
            raise ValueError("token_mask must be boolean and match input_ids")
        labels = jnp.where(token_mask, input_ids, -100)
    task = model.continuous.sub.compute_loss(result.logits, labels, losses=("CE",))
    return {"total": task, "task": task}, states


def objective_at_latents(
    model: A4Model,
    continuous_params: Any,
    symbolic_params: HeadParams,
    value_params: Mapping[str, Mapping[str, jax.Array]],
    input_ids: jax.Array,
    z_star: Mapping[int, jax.Array],
    labels: Mapping[int, AlignmentLabels],
    bank_keys: jax.Array,
    bank_values: jax.Array,
    *,
    key_weight: float = 1.0,
    value_weight: float = 1.0,
    token_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute CE and key/value alignment with fixed residual weights and latents.

    Every symbolic layer requires labelled token positions. Invalid dynamic
    positions yield NaN, which the optimizer step rejects.
    """
    if any(
        not math.isfinite(weight) or weight < 0 for weight in (key_weight, value_weight)
    ):
        raise ValueError("Alignment weights must be finite and nonnegative")
    if set(labels) != set(model.symbolic.layers) or set(value_params) != {
        f"layer_{layer}" for layer in model.symbolic.layers
    }:
        raise ValueError("Labels and value parameters must cover every symbolic layer")
    for positions, indices in labels.values():
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or positions.shape[0] == 0
            or not jnp.issubdtype(positions.dtype, jnp.integer)
            or indices.shape != (positions.shape[0],)
            or not jnp.issubdtype(indices.dtype, jnp.integer)
        ):
            raise ValueError(
                "Each integer [count, 2] position needs one integer target"
            )
    terms, states = _task_terms_at_latents(
        model,
        continuous_params,
        symbolic_params,
        input_ids,
        z_star,
        token_mask=token_mask,
    )
    key_loss = jnp.asarray(0.0, dtype=jnp.float32)
    value_loss = jnp.asarray(0.0, dtype=jnp.float32)
    predictor = ValuePredictor(model.symbolic.hidden_size, model.symbolic.key_dim)
    for layer in model.symbolic.layers:
        positions, indices = labels[layer]
        valid = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < input_ids.shape[0])
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < input_ids.shape[1])
        )
        batch = jnp.clip(positions[:, 0], 0, input_ids.shape[0] - 1)
        token = jnp.clip(positions[:, 1], 0, input_ids.shape[1] - 1)
        if token_mask is not None:
            valid = valid & token_mask[batch, token]
        hidden = states[layer][batch, token]
        name = f"layer_{layer}"
        aligned = alignment_loss(
            model.symbolic.project(symbolic_params[name], hidden),
            predictor.apply(value_params[name], hidden),
            bank_keys,
            bank_values,
            indices,
        )
        key_loss = key_loss + jnp.where(jnp.all(valid), aligned.key, jnp.nan)
        value_loss = value_loss + jnp.where(jnp.all(valid), aligned.value, jnp.nan)
    total = terms["total"] + key_weight * key_loss + value_weight * value_loss
    return total, {
        **terms,
        "total": total,
        "key": key_loss,
        "value": value_loss,
    }


__all__ = ["AlignmentLabels", "objective_at_latents"]
