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
from stages.stage_A.a4.checkpoint import pack_head_params
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE, A4Config
from stages.stage_A.a4.data import (
    HELD_OUT_GROUP_INTERPRETATION,
    HELD_OUT_GROUP_SCOPE,
    ROCSTORIES_EVAL_GROUPS,
    A4Data,
    SequentialAlignmentData,
)
from stages.stage_A.a4.model import A4Model
from stages.stage_A.a4.objective import AlignmentLabels, objective_at_latents
from stages.stage_base import (
    log_continual_summary,
    make_eval_set,
    summarize_continual_losses,
)

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


def make_train_step(
    model,
    frozen_residual,
    optimizer,
    bank_keys,
    bank_values,
    config,
    *,
    detach_alignment_states=False,
):
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
                    detach_alignment_states=detach_alignment_states,
                )

            (_, terms), grads = jax.value_and_grad(supervised_loss, has_aux=True)(
                params
            )
        else:
            (_, terms), symbolic_grads = jax.value_and_grad(loss, has_aux=True)(
                params["symbolic"]
            )

            grads = {
                "symbolic": symbolic_grads,
                "value": jax.tree.map(jnp.zeros_like, params["value"]),
            }

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
                [layer[name] for layer in grads["symbolic"].values()]
            )

        terms["V_pred_grad_norm"] = optax.global_norm(grads["value"])

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
    symbolic_enabled=True,
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
            symbolic_enabled=symbolic_enabled,
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

    log.info("\nTask 1/1")

    for epoch in range(training.epochs):
        epoch_samples = []

        log.info("  Epoch %d/%d", epoch + 1, training.epochs)

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
                or batch % training.log_every == 0
                or batch == batches_per_epoch
            ):
                log.info(
                    "%s",
                    training_row(batch, batches_per_epoch, metrics),
                )

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
        "mode": "standalone_aligned",
        "seed": training.seed,
        "checkpoints": ["checkpoints/a4-final.npz"],
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

    results["frozen_a3_baseline"] = {
        "validation": evaluate(
            model,
            frozen_residual,
            params,
            data.validation,
            training,
            data.validation_mask,
            symbolic_enabled=False,
        )
    }
    if data.test is not None:
        results["frozen_a3_baseline"]["test"] = evaluate(
            model,
            frozen_residual,
            params,
            data.test,
            training,
            data.test_mask,
            eval_batches=training.test_batches,
            symbolic_enabled=False,
        )

    writer.save_params(
        "a3-frozen",
        model.continuous.adapter_params(frozen_residual),
    )

    writer.save_checkpoint(
        "a4-final",
        pack_head_params(params),
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
    """Train one symbolic head across runtime or alignment-supervised tasks."""
    del latency_every  # Timing integration can be added later.

    training = config.training

    aligned = isinstance(data, SequentialAlignmentData)
    if aligned and (config.key_weight <= 0 or config.value_weight <= 0):
        raise ValueError("Sequential alignment requires positive key and value weights")
    if not aligned and (config.key_weight != 0 or config.value_weight != 0):
        raise ValueError(
            "Runtime sequential training requires zero " "key/value alignment weights"
        )

    tasks = data.tasks
    if not tasks:
        raise ValueError("No sequential tasks available")

    # Validate task data before starting expensive model updates.
    for task in tasks:
        blocks = np.asarray(task.train_blocks)

        if blocks.ndim != 2:
            raise ValueError(f"Task {task.name}: expected [samples, seq_len] blocks")

        if blocks.shape[1] != training.seq_len:
            raise ValueError(
                f"Task {task.name}: sequence length mismatch: "
                f"{blocks.shape[1]} != {training.seq_len}"
            )

        minimum_batch = 1 if aligned else training.batch_size
        if len(blocks) < minimum_batch:
            raise ValueError(
                f"Task {task.name}: insufficient training blocks "
                f"(requires at least {minimum_batch})"
            )

        if not aligned and not task.eval_set:
            raise ValueError(f"Task {task.name}: no validation batches")

    fingerprint = residual_fingerprint(model, frozen_residual)
    params = init_params(model, training.seed)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(training.learning_rate),
    )
    state = optimizer.init(params)

    bank_keys = jnp.asarray(np.stack([record.key_vector for record in bank.records]))
    bank_values = jnp.asarray(np.stack([record.val_vector for record in bank.records]))

    step = make_train_step(
        model,
        frozen_residual,
        optimizer,
        bank_keys,
        bank_values,
        config,
        detach_alignment_states=aligned,
    )

    names = [task.name for task in tasks]
    validation_masks, test_masks, alignment_metadata = {}, {}, {}
    if aligned:
        task_eval = data.task_eval
        eval_group_names = ROCSTORIES_EVAL_GROUPS
        if len(tasks) != len(eval_group_names) or set(task_eval) != set(
            eval_group_names
        ):
            raise ValueError(
                "Sequential alignment requires three tasks and three fixed held-out groups"
            )
        validation_scope = HELD_OUT_GROUP_SCOPE
        eval_sets = {
            "shared_validation": (data.validation, data.validation_mask),
            **{name: task_eval[name] for name in eval_group_names},
        }
        eval_blocks = {name: blocks for name, (blocks, _) in eval_sets.items()}
        eval_batch_counts = dict.fromkeys(eval_sets, training.eval_batches)
        validation_masks = {name: mask for name, (_, mask) in eval_sets.items()}
        test_blocks = {} if data.test is None else {"shared_test": data.test}
        test_masks = {"shared_test": data.test_mask}
        alignment_metadata = {
            "data_mode": "sequential-aligned",
            "task_order": names,
            "alignment_counts": data.provenance["alignment_counts"],
            "source_fingerprints": data.provenance["source_fingerprints"],
            "task_mapping": data.provenance["task_mapping"],
            "loss_weights": {
                "ce": 1.0,
                "key": config.key_weight,
                "value": config.value_weight,
            },
            "reset_optimizer": training.reset_optimizer,
            "training_state_source": TRAINING_STATE_SOURCE,
        }
    else:
        eval_group_names = tuple(names)
        validation_scope = data.metadata["validation_scope"]
        eval_sets = (
            {"shared_validation": tasks[0].eval_set}
            if validation_scope == "shared"
            else {task.name: task.eval_set for task in tasks}
        )
        eval_blocks = {
            name: np.concatenate(batches, axis=0) for name, batches in eval_sets.items()
        }
        eval_batch_counts = {name: len(batches) for name, batches in eval_sets.items()}

    shared_validation = validation_scope == "shared"
    # Aligned groups are separated by source document, and shared validation
    # can contain those same documents.
    if not aligned and not shared_validation:
        seen: set[bytes] = set()
        for blocks in eval_blocks.values():
            fingerprints = {row.tobytes() for row in blocks}
            if seen.intersection(fingerprints):
                raise ValueError(
                    "Task-specific evaluation requires distinct validation sets"
                )
            seen.update(fingerprints)

    def evaluate_all(after_task):
        results = {}
        for name, blocks in eval_blocks.items():
            results[name] = evaluate(
                model,
                frozen_residual,
                params,
                blocks,
                training,
                token_masks=validation_masks.get(name),
                eval_batches=eval_batch_counts[name],
            )
            writer.log(
                "eval",
                after_task=after_task,
                domain=name,
                split="validation",
                **results[name],
            )
        if aligned:
            step = (
                "before training"
                if after_task < 0
                else f"after Task {after_task + 1}/{len(tasks)}"
            )
            scores = " | ".join(
                f"{name} CE: {row['loss']:.4f} PPL: {row['ppl']:.2f}"
                for name, row in results.items()
            )
            log.info("  Validation %s | %s", step, scores)
        return results

    if shared_validation and not aligned:
        log.info("Forgetting metrics: N/A | Shared validation only.")
    initial = evaluate_all(-1)
    history = []
    updates = 0
    task_metrics = {}

    for task_index, task in enumerate(tasks):
        log.info("\nTask %d/%d", task_index + 1, len(tasks))

        optimizer_reset = task_index > 0 and training.reset_optimizer
        if optimizer_reset:
            state = optimizer.init(params)

        start_updates = updates
        examples = 0
        loss_sums = dict.fromkeys(
            ("task", "key", "value", "total", "gradient_norm"), 0.0
        )
        batch_size = training.batch_size
        num_examples = len(task.train_blocks)
        # A smaller final batch keeps all labels and can use its own JIT shape.
        stop = num_examples if aligned else num_examples - batch_size + 1
        batch_starts = range(0, stop, batch_size)
        batches_per_epoch = len(batch_starts)
        for epoch in range(task.epochs):
            log.info("  Epoch %d/%d", epoch + 1, task.epochs)
            seed = (
                np.random.SeedSequence([training.seed, task_index, epoch])
                if aligned
                else training.seed + task_index * 1000 + epoch
            )
            rng = np.random.default_rng(seed)
            order = rng.permutation(num_examples)

            for batch, start in enumerate(batch_starts, start=1):
                indices = order[start : start + batch_size]
                inputs = (
                    tuple(jnp.asarray(value) for value in task.batch_inputs(indices))
                    if aligned
                    else (jnp.asarray(task.train_blocks[indices]),)
                )

                params, state, terms, finite = step(
                    params,
                    state,
                    *inputs,
                )

                jax.block_until_ready((params, terms))

                if not bool(finite):
                    raise FloatingPointError(f"Nonfinite A4 update in task {task.name}")

                updates += 1

                record_metrics = updates % training.log_every == 0
                log_batch = (
                    batch == 1
                    or batch % training.log_every == 0
                    or batch == batches_per_epoch
                )
                if aligned or record_metrics or log_batch:
                    metrics = {
                        name: float(value)
                        for name, value in jax.device_get(terms).items()
                    }

                    metrics["symbolic"] = (
                        config.key_weight * metrics["key"]
                        + config.value_weight * metrics["value"]
                    )

                    ppl: float | None
                    try:
                        ppl = math.exp(metrics["task"])
                    except OverflowError:
                        ppl = None
                    log_metrics = {**metrics, "ppl": ppl}
                if aligned:
                    examples += len(indices)
                    for name in loss_sums:
                        loss_sums[name] += metrics[name]

                if record_metrics:
                    writer.log(
                        "train",
                        domain=task.name,
                        epoch=epoch + 1,
                        step=updates,
                        **log_metrics,
                    )
                if log_batch:
                    log.info(
                        "%s",
                        training_row(batch, batches_per_epoch, log_metrics),
                    )

        task_checkpoint_metadata = {}
        if aligned:
            task_updates = updates - start_updates
            summary = {
                "updates": task_updates,
                "examples_processed": examples,
                "optimizer_reset": optimizer_reset,
                "mean_per_update": {
                    name: total / task_updates if task_updates else None
                    for name, total in loss_sums.items()
                },
            }
            task_metrics[task.name] = summary
            writer.log("task_training", domain=task.name, **summary)
            task_checkpoint_metadata = {
                **alignment_metadata,
                "task": task.name,
                "task_index": task_index,
                "task_metrics": summary,
            }

        history.append(evaluate_all(task_index))

        writer.save_checkpoint(
            f"task_{task_index}_{task.name}",
            pack_head_params(params),
            opt_state=state,
            updates=updates,
            bank_snapshot_sha256=bank.input_snapshot_sha256,
            frozen_residual_sha256=fingerprint,
            **task_checkpoint_metadata,
        )

    if residual_fingerprint(model, frozen_residual) != fingerprint:
        raise RuntimeError("Frozen A3 residual changed")

    frozen = model.continuous.sub.verify_frozen()
    if not frozen["params_unchanged"]:
        raise RuntimeError("Frozen LLM changed")

    if not aligned:
        test_blocks = data.test_blocks
    report = {
        "stage": "A4",
        "mode": "sequential_aligned" if aligned else "runtime_sequential",
        "training_state_source": TRAINING_STATE_SOURCE,
        "seed": training.seed,
        "task_order": names,
        "task_epochs": {task.name: task.epochs for task in tasks},
        "validation_scope": validation_scope,
        "updates": updates,
        "initial": initial,
        "after_task": history,
        "test": {
            name: evaluate(
                model,
                frozen_residual,
                params,
                blocks,
                training,
                token_masks=test_masks.get(name),
                eval_batches=training.test_batches,
            )
            for name, blocks in test_blocks.items()
        },
        "frozen": frozen,
        "frozen_residual_sha256": fingerprint,
        "bank_snapshot_sha256": bank.input_snapshot_sha256,
        "checkpoints": [
            *(
                f"checkpoints/task_{index}_{name}.npz"
                for index, name in enumerate(names)
            ),
            "checkpoints/a4-final.npz",
        ],
    }
    if aligned:
        report.update(
            alignment_metadata,
            task_metrics=task_metrics,
            continual_evaluation={
                **data.provenance.get("continual_evaluation", {}),
                "scope": validation_scope,
                "interpretation": HELD_OUT_GROUP_INTERPRETATION,
                "group_order": list(eval_group_names),
                "shared_generalization_metric": "shared_validation",
            },
        )
    if not shared_validation:
        losses = np.asarray(
            [[row[name]["loss"] for name in eval_group_names] for row in history]
        )
        report["loss_matrix"] = losses.tolist()
        report["perplexity_matrix"] = [
            [row[name]["ppl"] for name in eval_group_names] for row in history
        ]
        continual_losses = summarize_continual_losses(losses)
        report["mean_forgetting"] = continual_losses["mean_forgetting"]
        if aligned:
            report["mean_seen_loss"] = continual_losses["mean_seen_loss"]
        if len(names) > 1:
            # Lower final CE on earlier tasks gives positive backward transfer.
            report["backward_transfer_ce"] = float(
                np.mean(losses.diagonal()[:-1] - losses[-1, :-1])
            )
        writer.log(
            "continual_evaluation",
            after_task=len(tasks) - 1,
            validation_scope=validation_scope,
            mean_seen_loss=continual_losses["mean_seen_loss"][-1],
            mean_forgetting=continual_losses["mean_forgetting"][-1],
            backward_transfer_ce=report.get("backward_transfer_ce"),
        )

    writer.save_params(
        "a3-frozen",
        model.continuous.adapter_params(frozen_residual),
    )

    writer.save_checkpoint(
        "a4-final",
        pack_head_params(params),
        opt_state=state,
        updates=updates,
        bank_snapshot_sha256=bank.input_snapshot_sha256,
        frozen_residual_sha256=fingerprint,
        **({**alignment_metadata, "task_metrics": task_metrics} if aligned else {}),
    )

    writer.write_json("results.json", report)
    if aligned:
        log_continual_summary(log, report, interpretation=HELD_OUT_GROUP_INTERPRETATION)
    return report


__all__ = [
    "evaluate",
    "init_params",
    "make_train_step",
    "residual_fingerprint",
    "train",
    "train_sequential",
]
