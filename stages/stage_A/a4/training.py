"""Symbolic-head training on a frozen continuous residual."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from residual.symbolic_head.alignment import ValuePredictor
from stages.results import RunWriter
from stages.stage_A.a4.config import A4Config
from stages.stage_A.a4.data import A4Data
from stages.stage_A.a4.model import A4Model
from stages.stage_A.a4.objective import AlignmentLabels, objective_at_latents
from stages.stage_A.a4.settlement import infer_latents
from stages.stage_base import make_eval_set

log = logging.getLogger("stage_a4")


def init_params(model: A4Model, seed: int) -> dict[str, Any]:
    """Initialize query, writeback and value-prediction parameters."""
    predictor = ValuePredictor(model.symbolic.hidden_size, model.symbolic.key_dim)
    return {
        "symbolic": model.symbolic.init_params(seed),
        "value": {
            f"layer_{layer}": predictor.init_params(seed + layer)
            for layer in model.symbolic.layers
        },
    }


def residual_fingerprint(model: A4Model, params: Any) -> str:
    """Hash the inherited predictor and projection matrices."""
    digest = hashlib.sha256()
    for layer, weights in sorted(model.continuous.adapter_params(params).items()):
        for name, value in sorted(weights.items()):
            digest.update(f"{layer}/{name}".encode())
            digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def make_train_step(model, frozen_residual, optimizer, bank_keys, bank_values, config):
    """Build an optimizer step over symbolic-head parameters only."""
    frozen_residual = jax.tree.map(jax.lax.stop_gradient, frozen_residual)

    def step(params, state, ids, key, positions, targets, token_mask=None):
        if set(params) != {"symbolic", "value"}:
            raise ValueError(
                "A4 optimizer parameters must contain only symbolic and value groups"
            )
        latents = infer_latents(model.continuous, frozen_residual, ids, key, token_mask)

        def loss(candidate):
            return objective_at_latents(
                model,
                frozen_residual,
                candidate["symbolic"],
                candidate["value"],
                ids,
                latents,
                {
                    layer: AlignmentLabels(positions, targets)
                    for layer in model.symbolic.layers
                },
                bank_keys,
                bank_values,
                key_weight=config.key_weight,
                value_weight=config.value_weight,
                token_mask=token_mask,
            )

        (_, terms), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, next_state = optimizer.update(grads, state, params)
        candidate = optax.apply_updates(params, updates)
        finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(leaf))
                    for leaf in jax.tree.leaves((terms, grads, candidate, next_state))
                ]
            )
        )
        candidate, next_state = jax.lax.cond(
            finite, lambda: (candidate, next_state), lambda: (params, state)
        )
        terms = {**terms, "gradient_norm": optax.global_norm(grads)}
        for name in ("W_sym", "b_sym", "U_out"):
            terms[f"{name}_grad_norm"] = optax.global_norm(
                [layer[name] for layer in grads["symbolic"].values()]
            )
        terms["V_pred_grad_norm"] = optax.global_norm(grads["value"])
        return candidate, next_state, terms, finite

    return jax.jit(step) if config.training.jit else step


def evaluate(model, frozen_residual, params, blocks, training, token_masks=None):
    """Evaluate label-free deployment cross-entropy and perplexity."""

    def loss(head_params, ids, mask):
        clean, _ = model.continuous.clean_pass(ids)
        latents = model.continuous.predict_latents(frozen_residual, clean)
        result = model.forward_at_latents(
            frozen_residual, head_params["symbolic"], ids, latents
        )
        labels = jnp.where(mask, ids, -100)
        return model.continuous.sub.compute_loss(result.logits, labels, losses=("CE",))

    evaluate_batch = jax.jit(loss) if training.jit else loss
    batches = make_eval_set(blocks, training.batch_size, training.eval_batches)
    if token_masks is None:
        token_masks = np.ones_like(blocks, dtype=bool)
    total, predictions = 0.0, 0
    for index, ids in enumerate(batches):
        start = index * training.batch_size
        mask = token_masks[start : start + len(ids)]
        count = int(np.sum(mask[:, 1:] & mask[:, :-1]))
        total += float(evaluate_batch(params, ids, jnp.asarray(mask))) * count
        predictions += count
    if not predictions:
        raise ValueError("Evaluation requires real next-token targets")
    mean = total / predictions
    if not math.isfinite(mean):
        raise FloatingPointError("Nonfinite evaluation loss")
    return {
        "loss": mean,
        "ppl": math.exp(mean),
        "batches": len(batches),
        "tokens": predictions,
    }


def train(
    model: A4Model,
    frozen_residual: Any,
    data: A4Data,
    bank: Any,
    config: A4Config,
    writer: RunWriter,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Train the symbolic head and save its checkpoint and evaluation metrics."""
    training = config.training
    fingerprint = residual_fingerprint(model, frozen_residual)
    params = init_params(model, training.seed)
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adam(training.learning_rate)
    )
    state = optimizer.init(params)
    step = make_train_step(
        model,
        frozen_residual,
        optimizer,
        jnp.asarray(np.stack([record.key_vector for record in bank.records])),
        jnp.asarray(np.stack([record.val_vector for record in bank.records])),
        config,
    )
    validation = evaluate(
        model, frozen_residual, params, data.validation, training, data.validation_mask
    )
    initial_validation = dict(validation)
    writer.log("eval", split="validation", epoch=0, **validation)
    updates = 0
    for epoch in range(training.epochs):
        order = np.random.default_rng(training.seed + epoch).permutation(
            len(data.train)
        )
        for start in range(
            0, len(order) - training.batch_size + 1, training.batch_size
        ):
            indices = order[start : start + training.batch_size]
            positions, targets = data.batch_labels(indices)
            started = time.perf_counter()
            params, state, terms, finite = step(
                params,
                state,
                jnp.asarray(data.train[indices]),
                jax.random.PRNGKey(training.seed + updates),
                jnp.asarray(positions),
                jnp.asarray(targets),
                None
                if data.train_mask is None
                else jnp.asarray(data.train_mask[indices]),
            )
            jax.block_until_ready((params, state, terms, finite))
            if not bool(finite):
                raise FloatingPointError("Nonfinite loss, gradient or optimizer update")
            updates += 1
            writer.log(
                "train",
                epoch=epoch + 1,
                step=updates,
                step_time=time.perf_counter() - started,
                **{name: float(value) for name, value in terms.items()},
            )
            if updates == 1 or updates % training.log_every == 0:
                log.info(
                    "epoch=%d step=%d CE=%.6f key=%.6f value=%.6f",
                    epoch + 1,
                    updates,
                    float(terms["task"]),
                    float(terms["key"]),
                    float(terms["value"]),
                )
        validation = evaluate(
            model,
            frozen_residual,
            params,
            data.validation,
            training,
            data.validation_mask,
        )
        writer.log("eval", split="validation", epoch=epoch + 1, **validation)
    if residual_fingerprint(model, frozen_residual) != fingerprint:
        raise RuntimeError("Frozen continuous residual changed")
    frozen = model.continuous.sub.verify_frozen()
    if not frozen["params_unchanged"]:
        raise RuntimeError("Frozen language model changed")
    results = {
        "stage": "A4",
        "updates": updates,
        "frozen_residual_sha256": fingerprint,
        "frozen": frozen,
        "validation": validation,
        "initial_validation": initial_validation,
        "validation_ce_change": validation["loss"] - initial_validation["loss"],
        "training_examples": len(data.train),
        "alignment": data.provenance.get("alignment", {}),
    }
    if data.test is not None:
        results["test"] = evaluate(
            model, frozen_residual, params, data.test, training, data.test_mask
        )
    writer.save_params("a3-frozen", model.continuous.adapter_params(frozen_residual))
    writer.save_checkpoint(
        "a4-final",
        {
            f"{group}.{layer}": weights
            for group, layers in params.items()
            for layer, weights in layers.items()
        },
        opt_state=state,
        updates=updates,
        frozen_residual_sha256=fingerprint,
        bank_snapshot_sha256=bank.input_snapshot_sha256,
    )
    writer.write_json("results.json", results)
    return results
