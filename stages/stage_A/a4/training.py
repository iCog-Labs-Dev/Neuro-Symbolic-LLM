"""Symbolic-head training on a frozen continuous residual."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from metrics.a4.console import training_row
from residual.symbolic_head.alignment import ValuePredictor
from stages.results import RunWriter
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE, A4Config
from stages.stage_A.a4.data import A4Data
from stages.stage_A.a4.model import A4Model
from stages.stage_A.a4.objective import AlignmentLabels, objective_at_latents
from stages.stage_base import make_eval_set

log = logging.getLogger("stage_a4")


@contextmanager
def _copy_sample(retriever, enabled):
    """Read existing native CUDA events around one real optimizer update."""
    sample = {}

    if enabled:
        with retriever.transfer_timing():
            before = retriever.transfer_stats()
            yield sample
            after = retriever.transfer_stats()
            sample.update({key: after[key] - before[key] for key in before})
    else:
        yield sample


def _communication_summary(samples):
    """Aggregate recorded copy intervals without rerunning or logging work."""
    if not samples:
        return {"status": "not_measured", "samples": 0}

    steps = math.fsum(row["step_seconds"] for row in samples)
    d2h = math.fsum(row["d2h_gpu_seconds"] for row in samples)
    h2d = math.fsum(row["h2d_gpu_seconds"] for row in samples)

    return {
        "status": "measured",
        "samples": len(samples),
        "gpu_to_cpu_mean_ms": 1000 * d2h / len(samples),
        "cpu_to_gpu_mean_ms": 1000 * h2d / len(samples),
        "copy_mean_ms": 1000 * (d2h + h2d) / len(samples),
        "sampled_step_mean_ms": 1000 * steps / len(samples),
        "copy_ratio_of_means_percent": 100 * (d2h + h2d) / steps,
        "scope": "CUDA-event copy intervals in real weight-updating steps",
        "denominator": (
            "Same instrumented training steps; excludes first 10 updates "
            "and first use of each input shape"
        ),
        "excluded": (
            "CPU search and host dispatch/waits outside CUDA event brackets; "
            "not a full bridge overhead qualification"
        ),
    }


def _log_communication(report):
    if report["status"] != "measured":
        log.info(
            "CUDA copies: no sampled updates "
            "(disabled, unsupported backend, or run too short)"
        )
        return

    log.info(
        "CUDA copies during training: %d sampled updates | GPU->CPU %.3f ms | "
        "CPU->GPU %.3f ms | copies %.3f ms / instrumented step %.3f ms = %.2f%%",
        report["samples"],
        report["gpu_to_cpu_mean_ms"],
        report["cpu_to_gpu_mean_ms"],
        report["copy_mean_ms"],
        report["sampled_step_mean_ms"],
        report["copy_ratio_of_means_percent"],
    )


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
    """Build an A4 optimizer step for runtime-only or supervised training."""
    frozen_residual = jax.tree.map(jax.lax.stop_gradient, frozen_residual)

    alignment_enabled = config.key_weight > 0 or config.value_weight > 0

    def step(params, state, ids, positions=None, targets=None, token_mask=None):
        if set(params) != {"symbolic", "value"}:
            raise ValueError(
                "A4 optimizer parameters must contain only symbolic and value groups"
            )

        if alignment_enabled and (positions is None or targets is None):
            raise ValueError(
                "Alignment-supervised training requires positions and targets"
            )

        labels = (
            {
                layer: AlignmentLabels(positions, targets)
                for layer in model.symbolic.layers
            }
            if alignment_enabled
            else None
        )

        clean, _ = model.continuous.clean_pass(ids)

        latents = jax.tree.map(
            jax.lax.stop_gradient,
            model.continuous.predict_latents(frozen_residual, clean),
        )

        def loss(symbolic_params):
            total, terms = objective_at_latents(
                model,
                frozen_residual,
                symbolic_params,
                params["value"],
                ids,
                latents,
                labels,
                bank_keys,
                bank_values,
                key_weight=config.key_weight,
                value_weight=config.value_weight,
                token_mask=token_mask,
            )
            return total, terms

        if alignment_enabled:
            def supervised_loss(candidate):
                return objective_at_latents(
                    model,
                    frozen_residual,
                    candidate["symbolic"],
                    candidate["value"],
                    ids,
                    latents,
                    labels,
                    bank_keys,
                    bank_values,
                    key_weight=config.key_weight,
                    value_weight=config.value_weight,
                    token_mask=token_mask,
                )

            (_, terms), grads = jax.value_and_grad(
                supervised_loss, has_aux=True
            )(params)
        else:
            (_, terms), symbolic_grads = jax.value_and_grad(
                loss, has_aux=True
            )(params["symbolic"])

            grads = {
                "symbolic": symbolic_grads,
                "value": jax.tree.map(
                    jnp.zeros_like, params["value"]
                ),
            }

        updates, next_state = optimizer.update(grads, state, params)
        candidate = optax.apply_updates(params, updates)

        finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(leaf))
                    for leaf in jax.tree.leaves(
                        (terms, grads, candidate, next_state)
                    )
                ]
            )
        )

        candidate, next_state = jax.lax.cond(
            finite,
            lambda: (candidate, next_state),
            lambda: (params, state),
        )

        terms = {
            **terms,
            "gradient_norm": optax.global_norm(grads),
        }

        for name in ("W_sym", "b_sym", "U_out"):
            terms[f"{name}_grad_norm"] = optax.global_norm(
                [
                    layer[name]
                    for layer in grads["symbolic"].values()
                ]
            )

        terms["V_pred_grad_norm"] = optax.global_norm(
            grads["value"]
        )

        return candidate, next_state, terms, finite

    return jax.jit(step) if config.training.jit else step


def evaluate(
    model,
    frozen_residual,
    params,
    blocks,
    training,
    token_masks=None,
    *,
    eval_batches=None,
):
    """Evaluate label-free deployment cross-entropy and perplexity.

    eval_batches allows validation and test evaluation to use
    different batch counts.
    """

    def loss(head_params, ids, mask):
        clean, _ = model.continuous.clean_pass(ids)

        latents = model.continuous.predict_latents(frozen_residual, clean)

        result = model.forward_at_latents(
            frozen_residual,
            head_params["symbolic"],
            ids,
            latents,
        )

        labels = jnp.where(mask, ids, -100)

        return model.continuous.sub.compute_loss(
            result.logits,
            labels,
            losses=("CE",),
        )

    evaluate_batch = jax.jit(loss) if training.jit else loss

    num_batches = training.eval_batches if eval_batches is None else eval_batches

    batches = make_eval_set(
        blocks,
        training.batch_size,
        num_batches,
    )

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
    latency_every: int = 100,
) -> dict[str, Any]:
    """Train the symbolic head and save its checkpoint and evaluation metrics."""

    training = config.training

    if latency_every < 0:
        raise ValueError("latency_every must be nonnegative")

    retriever = model.symbolic.retrieve

    timing_enabled = bool(
        latency_every
        and training.jit
        and callable(getattr(retriever, "transfer_timing", None))
        and callable(getattr(retriever, "transfer_stats", None))
    )

    communication_samples = []
    seen_shapes = set()

    fingerprint = residual_fingerprint(model, frozen_residual)

    params = init_params(model, training.seed)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(training.learning_rate),
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

    log.info("Task 1/1: A4 symbolic-head training")

    if timing_enabled:
        log.info(
            "CUDA copy timing: every %d updates after 10-update "
            "and input-shape warmup",
            latency_every,
        )
    else:
        log.info(
            "Communication timing disabled or unavailable "
            "(requires compiled native CUDA retrieval)"
        )

    log.info("Initial validation...")

    validation = evaluate(
        model,
        frozen_residual,
        params,
        data.validation,
        training,
        data.validation_mask,
    )

    initial_validation = dict(validation)

    writer.log("eval", split="validation", epoch=0, **validation)

    log.info(
        "Validation: task loss=%.6f PPL=%.4f",
        validation["loss"],
        validation["ppl"],
    )

    updates = 0
    batches_per_epoch = len(data.train) // training.batch_size
    step_times = []

    columns = (
        f"{'Batch':>12} {'Total loss':>14} {'Task loss':>14} "
        f"{'Symbolic loss':>14} {'PPL':>12}"
    )

    for epoch in range(training.epochs):
        epoch_samples = []

        log.info("Epoch %d/%d", epoch + 1, training.epochs)
        log.info("%s", columns)

        order = np.random.default_rng(training.seed + epoch).permutation(
            len(data.train)
        )

        for start in range(
            0,
            len(order) - training.batch_size + 1,
            training.batch_size,
        ):
            indices = order[start : start + training.batch_size]

            positions, targets = data.batch_labels(indices)

            inputs = (
                jnp.asarray(data.train[indices]),
                jnp.asarray(positions),
                jnp.asarray(targets),
                (
                    None
                    if data.train_mask is None
                    else jnp.asarray(data.train_mask[indices])
                ),
            )

            shape = tuple(
                None if value is None else (value.shape, str(value.dtype))
                for value in inputs
            )

            measure = (
                timing_enabled
                and updates >= 10
                and shape in seen_shapes
                and (updates + 1) % latency_every == 0
            )

            with _copy_sample(retriever, measure) as sample:
                started = time.perf_counter()

                params, state, terms, finite = step(
                    params,
                    state,
                    *inputs,
                )

                jax.block_until_ready((params, state, terms, finite))

                if measure:
                    jax.effects_barrier()

                elapsed = time.perf_counter() - started

            seen_shapes.add(shape)

            if not bool(finite):
                raise FloatingPointError("Nonfinite loss, gradient or optimizer update")

            if measure:
                if sample["calls"] != len(model.symbolic.layers):
                    raise RuntimeError("CUDA timing did not cover every symbolic layer")

                sample.update(
                    step_seconds=elapsed,
                    step=updates + 1,
                    epoch=epoch + 1,
                )

                communication_samples.append(sample)
                epoch_samples.append(sample)

            step_times.append(elapsed)

            metrics: dict[str, Any] = {
                name: float(value) for name, value in terms.items()
            }

            metrics["symbolic"] = (
                config.key_weight * metrics["key"]
                + config.value_weight * metrics["value"]
            )

            if measure:
                metrics["communication"] = sample

            try:
                metrics["ppl"] = math.exp(metrics["task"])
            except OverflowError:
                metrics["ppl"] = None

            updates += 1

            writer.log(
                "train",
                epoch=epoch + 1,
                step=updates,
                step_time=elapsed,
                **metrics,
            )

            batch = start // training.batch_size + 1

            if (
                batch == 1
                or updates % training.log_every == 0
                or batch == batches_per_epoch
            ):
                log.info(
                    "%s",
                    training_row(batch, batches_per_epoch, metrics),
                )

        log.info("%s", columns)

        epoch_communication = _communication_summary(epoch_samples)

        writer.log(
            "communication",
            epoch=epoch + 1,
            **epoch_communication,
        )

        _log_communication(epoch_communication)

        validation = evaluate(
            model,
            frozen_residual,
            params,
            data.validation,
            training,
            data.validation_mask,
        )

        writer.log(
            "eval",
            split="validation",
            epoch=epoch + 1,
            **validation,
        )

        log.info(
            "Validation: task loss=%.6f PPL=%.4f",
            validation["loss"],
            validation["ppl"],
        )

    if residual_fingerprint(model, frozen_residual) != fingerprint:
        raise RuntimeError("Frozen continuous residual changed")

    frozen = model.continuous.sub.verify_frozen()

    if not frozen["params_unchanged"]:
        raise RuntimeError("Frozen language model changed")

    results = {
        "stage": "A4",
        "training_state_source": TRAINING_STATE_SOURCE,
        "updates": updates,
        "frozen_residual_sha256": fingerprint,
        "frozen": frozen,
        "validation": validation,
        "initial_validation": initial_validation,
        "validation_ce_change": (validation["loss"] - initial_validation["loss"]),
        "training_examples": len(data.train),
        "alignment": data.provenance.get("alignment", {}),
        "communication": _communication_summary(communication_samples),
        "training_step_mean_ms": (
            1000 * math.fsum(step_times[1:]) / (len(step_times) - 1)
            if len(step_times) > 1
            else None
        ),
    }

    if data.test is not None:
        results["test"] = evaluate(
            model,
            frozen_residual,
            params,
            data.test,
            training,
            data.test_mask,
            eval_batches=training.test_batches,
        )

    writer.save_params(
        "a3-frozen",
        model.continuous.adapter_params(frozen_residual),
    )

    writer.save_checkpoint(
        "a4-final",
        {
            f"{group}.{layer}": weights
            for group, layers in params.items()
            for layer, weights in layers.items()
        },
        opt_state=state,
        updates=updates,
        training_state_source=TRAINING_STATE_SOURCE,
        frozen_residual_sha256=fingerprint,
        bank_snapshot_sha256=bank.input_snapshot_sha256,
    )

    writer.write_json("results.json", results)

    log.info(
        "Summary: %d epochs, %d updates",
        training.epochs,
        updates,
    )

    log.info(
        "Final validation: task loss=%.6f PPL=%.4f",
        validation["loss"],
        validation["ppl"],
    )

    mean_ms = results["training_step_mean_ms"]

    if mean_ms is not None:
        log.info(
            "Latency: mean training step=%.3f ms " "(excluding first update)",
            mean_ms,
        )
    else:
        log.info(
            "Latency: insufficient updates for a mean " "excluding the first update"
        )

    _log_communication(results["communication"])

    log.info("Saved run: %s", writer.dir)

    return results


def train_sequential(
    model: A4Model,
    frozen_residual: Any,
    data: Any,
    bank: Any,
    config: A4Config,
    writer: RunWriter,
    *,
    max_grad_norm: float = 1.0,
    latency_every: int = 100,
) -> dict[str, Any]:
    """Train A4 sequentially on shared runtime tasks.

    The LLM and A3 residual remain frozen. Only symbolic
    head parameters are updated using next-token CE.
    """
    del latency_every  # Timing integration can be added later.

    training = config.training

    if config.key_weight != 0 or config.value_weight != 0:
        raise ValueError(
            "Runtime sequential training requires zero "
            "key/value alignment weights"
        )

    tasks = data.tasks
    if not tasks:
        raise ValueError("No runtime tasks available")

    # Validate task data before starting expensive model updates.
    for task in tasks:
        blocks = np.asarray(task.train_blocks)

        if blocks.ndim != 2:
            raise ValueError(
                f"Task {task.name}: expected [samples, seq_len] blocks"
            )

        if blocks.shape[1] != training.seq_len:
            raise ValueError(
                f"Task {task.name}: sequence length mismatch: "
                f"{blocks.shape[1]} != {training.seq_len}"
            )

        if len(blocks) < training.batch_size:
            raise ValueError(
                f"Task {task.name}: insufficient training blocks "
                f"for batch size {training.batch_size}"
            )

        if not task.eval_set:
            raise ValueError(
                f"Task {task.name}: no validation batches"
            )

    fingerprint = residual_fingerprint(model, frozen_residual)
    params = init_params(model, training.seed)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(training.learning_rate),
    )
    state = optimizer.init(params)

    bank_keys = jnp.asarray(
        np.stack([record.key_vector for record in bank.records])
    )
    bank_values = jnp.asarray(
        np.stack([record.val_vector for record in bank.records])
    )

    step = make_train_step(
        model,
        frozen_residual,
        optimizer,
        bank_keys,
        bank_values,
        config,
    )

    names = [task.name for task in tasks]
    eval_blocks = {
        task.name: np.concatenate(
            [np.asarray(batch) for batch in task.eval_set],
            axis=0,
        )
        for task in tasks
    }

    def evaluate_all(after_task):
        results = {}
        for task in tasks:
            results[task.name] = evaluate(
                model,
                frozen_residual,
                params,
                eval_blocks[task.name],
                training,
                eval_batches=len(task.eval_set),
            )
            writer.log(
                "eval",
                after_task=after_task,
                domain=task.name,
                split="validation",
                **results[task.name],
            )
        return results

    initial = evaluate_all(-1)
    history = []
    updates = 0

    for task_index, task in enumerate(tasks):
        log.info(
            "A4 Task %d/%d: %s",
            task_index + 1,
            len(tasks),
            task.name,
        )

        if task_index > 0 and training.reset_optimizer:
            state = optimizer.init(params)

        for epoch in range(task.epochs):
            rng = np.random.default_rng(
                training.seed + task_index * 1000 + epoch
            )
            order = rng.permutation(len(task.train_blocks))
            batch_size = training.batch_size

            for start in range(
                0,
                len(order) - batch_size + 1,
                batch_size,
            ):
                indices = order[start : start + batch_size]
                ids = jnp.asarray(task.train_blocks[indices])

                params, state, terms, finite = step(
                    params,
                    state,
                    ids,
                )

                jax.block_until_ready((params, terms))

                if not bool(finite):
                    raise FloatingPointError(
                        f"Nonfinite A4 update in task {task.name}"
                    )

                updates += 1

                if updates % training.log_every == 0:
                    metrics = {
                        name: float(value)
                        for name, value in terms.items()
                    }
                    writer.log(
                        "train",
                        domain=task.name,
                        epoch=epoch + 1,
                        step=updates,
                        **metrics,
                    )
                    log.info(
                        "%s | epoch=%d | step=%d | CE=%.4f",
                        task.name,
                        epoch + 1,
                        updates,
                        metrics["task"],
                    )

        results = evaluate_all(task_index)
        history.append(results)

        writer.save_checkpoint(
            f"task_{task_index}_{task.name}",
            {
                f"{group}.{layer}": weights
                for group, layers in params.items()
                for layer, weights in layers.items()
            },
            opt_state=state,
            updates=updates,
            bank_snapshot_sha256=bank.input_snapshot_sha256,
            frozen_residual_sha256=fingerprint,
        )

    if residual_fingerprint(model, frozen_residual) != fingerprint:
        raise RuntimeError("Frozen A3 residual changed")

    frozen = model.continuous.sub.verify_frozen()
    if not frozen["params_unchanged"]:
        raise RuntimeError("Frozen LLM changed")

    from stages.stage_base import mean_forgetting

    loss_matrix = np.asarray(
        [
            [row[name]["loss"] for name in names]
            for row in history
        ],
        dtype=float,
    )

    forgetting = [
        None if index == 0 else float(
            mean_forgetting(
                loss_matrix,
                index,
                higher_is_better=False,
            )
        )
        for index in range(len(history))
    ]

    report = {
        "stage": "A4",
        "mode": "runtime_sequential",
        "task_order": names,
        "updates": updates,
        "initial": initial,
        "after_task": history,
        "loss_matrix": loss_matrix.tolist(),
        "mean_forgetting": forgetting,
        "frozen": frozen,
        "frozen_residual_sha256": fingerprint,
        "bank_snapshot_sha256": bank.input_snapshot_sha256,
    }

    writer.save_params(
        "a3-frozen",
        model.continuous.adapter_params(frozen_residual),
    )

    writer.save_checkpoint(
        "a4-final",
        {
            f"{group}.{layer}": weights
            for group, layers in params.items()
            for layer, weights in layers.items()
        },
        opt_state=state,
        updates=updates,
        bank_snapshot_sha256=bank.input_snapshot_sha256,
        frozen_residual_sha256=fingerprint,
    )

    writer.write_json("results.json", report)
    return report


__all__ = [
    "evaluate",
    "init_params",
    "make_train_step",
    "residual_fingerprint",
    "train",
    "train_sequential",
]
