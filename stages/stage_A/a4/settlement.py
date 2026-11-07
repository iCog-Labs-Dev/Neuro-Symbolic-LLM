"""Frozen continuous-state inference for head-only A4 training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import build_clamps

from stages.stage_A.a3_tail import residual_modify


def infer_latents(model, params, ids: jax.Array, key: jax.Array, token_mask=None):
    """Settle PC latents without computing parameter gradients."""
    if token_mask is not None:
        return _infer_masked_latents(model, params, ids, token_mask)
    batch, _, _ = model.make_batch(ids)
    clamps = build_clamps(batch, model.structure, clamp_target=True)
    state = initialize_graph_state(
        model.structure, batch["x"].shape[0], key, clamps, params=params
    )
    state = model._refresh_state(params, state, clamps)
    state = run_inference(params, state, clamps, model.structure)
    state = model._refresh_state(params, state, clamps)
    return {
        layer: jax.lax.stop_gradient(state.nodes[f"z{layer}"].z_latent)
        for layer in model.layers
    }


def _infer_masked_latents(model, params, ids, token_mask):
    """K-step A3 energy descent with right-padding excluded from every term."""
    ids = model._check_ids(ids)
    if token_mask.shape != ids.shape or token_mask.dtype != jnp.bool_:
        raise ValueError("token_mask must be boolean and match ids")
    clean, base = model.clean_pass(ids)
    predicted = model.predict_latents(params, clean)
    projections = {layer: model.projection(params, layer) for layer in model.layers}
    labels = jnp.where(token_mask, ids, -100)
    valid_next = token_mask[:, :-1] & token_mask[:, 1:]
    labels = labels.at[:, 1:].set(jnp.where(valid_next, ids[:, 1:], -100))
    count = jnp.sum(valid_next)
    base_logp = (
        jax.nn.log_softmax(base[:, :-1].astype(jnp.float32), axis=-1)
        if model.cfg.lambda_kl > 0
        else None
    )

    def energy(latents):
        logits = model.sub.run_with_interception(
            ids,
            modify_fn=residual_modify(latents, projections, model.scale),
            intercept_layers=model.layers,
        ).logits
        task = model.sub.compute_loss(logits, labels, losses=("CE",)) * count
        spring = sum(
            jnp.sum(
                GaussianEnergy.energy(
                    jnp.where(token_mask[..., None], latents[layer], predicted[layer]),
                    predicted[layer],
                    {"precision": 1.0 / model.cfg.sigma2},
                )
            )
            for layer in model.layers
        )
        if base_logp is not None:
            logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
            kl = jnp.sum(jnp.exp(base_logp) * (base_logp - logp), axis=-1)
            task = task + model.cfg.lambda_kl * jnp.sum(jnp.where(valid_next, kl, 0.0))
        return task + spring

    def settle(_, latents):
        grads = jax.grad(energy)(latents)
        return {
            layer: latents[layer] - model.cfg.eta_z * grads[layer]
            for layer in model.layers
        }

    settled = jax.lax.fori_loop(0, model.cfg.infer_steps, settle, predicted)
    valid_layout = jnp.all(token_mask[:, :-1] | ~token_mask[:, 1:]) & jnp.all(
        jnp.sum(token_mask, axis=1) >= 2
    )
    return jax.tree.map(
        lambda value: jax.lax.stop_gradient(jnp.where(valid_layout, value, jnp.nan)),
        settled,
    )
