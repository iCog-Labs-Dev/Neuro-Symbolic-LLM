"""A4 task, predictive-coding spring and full-bank alignment at fixed latents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from fabricpc.core.energy import GaussianEnergy

from residual.symbolic_head.alignment import ValuePredictor, alignment_loss
from residual.symbolic_head.jax_head import HeadParams
from stages.stage_A.a4 import A4Model


class AlignmentLabels(NamedTuple):
    """One bank target per labelled [batch, token] position at a head layer."""

    positions: jax.Array
    template_indices: jax.Array


def objective_at_latents(
    model: A4Model,
    continuous_params: Any,
    symbolic_params: HeadParams,
    value_params: Mapping[str, Mapping[str, jax.Array]],
    input_ids: jax.Array,
    z_star: Mapping[int, jax.Array],
    clean_hidden: Mapping[int, jax.Array],
    labels: Mapping[int, AlignmentLabels],
    bank_keys: jax.Array,
    bank_values: jax.Array,
    *,
    base_logits: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Sum task, optional base KL, spring and per-layer alignment losses.

    Final PC latents and clean-pass targets are fixed outer-step inputs. Task
    loss is evaluated once through the symbolic forward path. No A3 task loss
    or precomputed A3 gradient is added. The spring trains the latent predictor;
    task/alignment losses can train the continuous projection and earlier live
    layers. Bank vectors remain fixed. Each head layer needs explicit labels.

    Invalid dynamic label positions produce NaN; callers must require finite
    loss and gradients before applying optimizer updates.
    """
    if set(clean_hidden) != set(model.continuous.layers):
        raise ValueError("Clean hidden states must cover every continuous layer")
    if set(labels) != set(model.symbolic.layers) or set(value_params) != {
        f"layer_{layer}" for layer in model.symbolic.layers
    }:
        raise ValueError("Labels and value parameters must cover every symbolic layer")
    if model.continuous.cfg.lambda_kl > 0 and base_logits is None:
        raise ValueError("Base logits are required when the KL weight is nonzero")
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
    z_fixed = jax.tree.map(jax.lax.stop_gradient, z_star)
    h_fixed = jax.tree.map(jax.lax.stop_gradient, clean_hidden)
    result, states = model.forward_with_states_at_latents(
        continuous_params, symbolic_params, input_ids, z_fixed
    )
    task = model.continuous.sub.compute_loss(result.logits, input_ids, losses=("CE",))
    kl = jnp.asarray(0.0, dtype=jnp.float32)
    if model.continuous.cfg.lambda_kl > 0:
        assert base_logits is not None
        kl = model.continuous.sub.compute_loss(
            result.logits[:, :-1],
            None,
            losses=("KL",),
            base_logits=jax.lax.stop_gradient(base_logits[:, :-1]),
        )
    predictions = model.continuous.predict_latents(continuous_params, h_fixed)
    n_pred = input_ids.shape[0] * (input_ids.shape[1] - 1)
    spring = (
        sum(
            jnp.sum(
                GaussianEnergy.energy(
                    z_fixed[layer],
                    predictions[layer],
                    {"precision": 1.0 / model.continuous.cfg.sigma2},
                )
            )
            for layer in model.continuous.layers
        )
        / n_pred
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
    total = task + model.continuous.cfg.lambda_kl * kl + spring + key_loss + value_loss
    return total, {
        "total": total,
        "task": task,
        "kl": kl,
        "spring": spring,
        "key": key_loss,
        "value": value_loss,
    }


__all__ = ["AlignmentLabels", "objective_at_latents"]
