"""Shared sequential training engine for A1, A2 and A3."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
import optax

from metrics.performance import perplexity
from stages.results import RunWriter
from stages.stage_base import (
    TrainingConfig,
    batches,
    summarize,
)

log = logging.getLogger(__name__)


def _to_scalars(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert scalar metrics to Python floats."""
    return {
        name: float(value) for name, value in metrics.items() if np.ndim(value) == 0
    }


def _format_metrics(metrics: Mapping[str, float]) -> str:
    """Format total, task and stage-specific auxiliary losses."""

    total_loss = metrics.get("total", metrics.get("total_loss"))
    task_loss = metrics.get("task", metrics.get("task_loss"))

    parts = []

    if total_loss is not None:
        parts.append(f"Total loss: {total_loss:.4f}")

    if task_loss is not None:
        parts.append(f"Task loss: {task_loss:.4f}")

    auxiliary_metrics = {
        "pc_energy": "PC energy",
        "hid": "Hidden loss",
        "cpc": "CPC loss",
        "stab": "Stability loss",
        "kl": "KL loss",
    }

    for key, label in auxiliary_metrics.items():
        if key in metrics:
            parts.append(f"{label}: {metrics[key]:.6f}")

    if task_loss is not None:
        parts.append(f"PPL: {float(perplexity(task_loss)):.2f}")

    return " | ".join(parts)


def train(
    *,
    params: Any,
    tasks: Sequence[Any],
    forward_fn: Callable | None,
    loss_fn: Callable | None,
    loss_config: Any,
    optimizer: optax.GradientTransformation,
    config: TrainingConfig,
    writer: RunWriter,
    eval_fn: Callable,
    evaluation_tasks: Sequence[Any] | None = None,
    pc_gradient_fn: Callable | None = None,
    checkpoint_fn: Callable | None = None,
    after_task_fn: Callable | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train a stage using shared sequential training orchestration.

    Standard path (A1/A2):
        predictions = forward_fn(params, batch, key)
        loss_fn(predictions, batch, params, loss_config)
        gradients are computed using JAX autodiff.

    Predictive-coding path (A3):
        pc_gradient_fn(params, batch, key)
        returns gradients and stage-specific metrics.

    The optimizer update and training lifecycle are shared.
    """

    if not tasks:
        raise ValueError("At least one training task is required.")

    if pc_gradient_fn is None and (forward_fn is None or loss_fn is None):
        raise ValueError("forward_fn and loss_fn are required for standard training.")

    eval_tasks = (
        tuple(evaluation_tasks) if evaluation_tasks is not None else tuple(tasks)
    )
    if not eval_tasks:
        raise ValueError("At least one evaluation task is required.")
    names = [task.name for task in eval_tasks]

    optimizer_state = optimizer.init(params)

    base_key = jax.random.PRNGKey(config.seed + 100)

    global_step = 0
    total_tokens = 0
    step_times = []

    task_results = []

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    def evaluate_all(after_task: int):
        """Evaluate all tasks and save metrics without console spam."""

        results = {}

        for task in eval_tasks:
            result = eval_fn(
                params,
                task.eval_set,
            )

            results[task.name] = result

            writer.log(
                "eval",
                after_task=after_task,
                domain=task.name,
                **result,
            )

        return results

    # Initial evaluation is retained for forgetting calculations.
    initial_results = evaluate_all(-1)

    # -------------------------------------------------------------
    # Standard JAX gradient path (A1/A2)
    # -------------------------------------------------------------

    def standard_objective(p, batch, key):
        predictions = forward_fn(p, batch, key)

        return loss_fn(
            predictions,
            batch,
            p,
            loss_config,
        )

    grad_fn = jax.value_and_grad(
        standard_objective,
        has_aux=True,
    )

    def standard_step(p, batch, key):
        (_, metrics), grads = grad_fn(
            p,
            batch,
            key,
        )

        return grads, metrics

    # -------------------------------------------------------------
    # Select standard or predictive-coding gradient path
    # -------------------------------------------------------------

    train_step: Any = pc_gradient_fn if pc_gradient_fn is not None else standard_step

    if config.jit:
        train_step = jax.jit(train_step)

    # -------------------------------------------------------------
    # Sequential task training
    # -------------------------------------------------------------

    total_tasks = len(tasks)

    for task_idx, task in enumerate(tasks):
        if task_idx > 0 and config.reset_optimizer:
            optimizer_state = optimizer.init(params)

        epochs = task.epochs

        if epochs < 0:
            raise ValueError(f"Task {task.name!r}: epochs must be >= 0.")

        if epochs > 0 and len(task.train_blocks) < config.batch_size:
            raise ValueError(f"Task {task.name!r} has insufficient training blocks.")

        batches_per_epoch = (
            len(task.train_blocks) // config.batch_size if epochs > 0 else 0
        )

        task_step = 0
        task_seed = int.from_bytes(
            hashlib.sha256(task.name.encode()).digest()[:4], "big"
        )

        # Print task number only, not dataset/task names.
        log.info(
            "\nTask %d/%d",
            task_idx + 1,
            total_tasks,
        )

        # ---------------------------------------------------------
        # Epoch training
        # ---------------------------------------------------------

        for epoch in range(1, epochs + 1):
            log.info(
                "  Epoch %d/%d",
                epoch,
                epochs,
            )

            epoch_start = time.time()

            epoch_metrics = []
            epoch_step_times = []
            epoch_tokens = 0

            stream = batches(
                task.train_blocks,
                config.batch_size,
                config.seed + task_seed + epoch - 1,
            )

            # -----------------------------------------------------
            # Batch training
            # -----------------------------------------------------

            for batch_idx, batch in enumerate(
                stream,
                start=1,
            ):
                task_step += 1
                global_step += 1

                key = jax.random.fold_in(
                    jax.random.fold_in(base_key, task_seed),
                    task_step,
                )

                step_start = time.perf_counter()
                grads, metrics = train_step(params, batch, key)

                updates, optimizer_state = optimizer.update(
                    grads,
                    optimizer_state,
                    params,
                )

                params = optax.apply_updates(
                    params,
                    updates,
                )
                jax.block_until_ready((params, metrics))
                step_time = time.perf_counter() - step_start

                scalars = _to_scalars(metrics)

                epoch_metrics.append(scalars)
                epoch_step_times.append(step_time)

                batch_tokens = int(batch.size)

                total_tokens += batch_tokens
                epoch_tokens += batch_tokens

                step_times.append(step_time)

                # -------------------------------------------------
                # Save detailed batch metrics
                # -------------------------------------------------

                writer_metrics = {
                    f"loss_{name}": value for name, value in scalars.items()
                }

                task_loss = scalars.get(
                    "task",
                    scalars.get("task_loss"),
                )

                if task_loss is not None:
                    writer_metrics["ppl"] = float(perplexity(task_loss))

                writer.log(
                    "train",
                    task=task_idx,
                    domain=task.name,
                    epoch=epoch,
                    batch=batch_idx,
                    global_step=global_step,
                    tokens=total_tokens,
                    step_time=step_time,
                    **writer_metrics,
                )

                # -------------------------------------------------
                # Concise console batch logging
                # -------------------------------------------------

                if (
                    batch_idx == 1
                    or batch_idx % config.log_every == 0
                    or batch_idx == batches_per_epoch
                ):
                    log.info(
                        "    Batch %d/%d | %s",
                        batch_idx,
                        batches_per_epoch,
                        _format_metrics(scalars),
                    )

            # -----------------------------------------------------
            # Save epoch metrics (without console logging)
            # -----------------------------------------------------

            if epoch_metrics:
                keys = epoch_metrics[0].keys()

                averages = {
                    name: float(np.mean([row[name] for row in epoch_metrics]))
                    for name in keys
                }

                duration = time.time() - epoch_start

                mean_step_time = float(np.mean(epoch_step_times))

                epoch_writer_metrics = {
                    f"loss_{name}": value for name, value in averages.items()
                }

                task_loss = averages.get(
                    "task",
                    averages.get("task_loss"),
                )

                if task_loss is not None:
                    epoch_writer_metrics["ppl"] = float(perplexity(task_loss))

                writer.log(
                    "epoch",
                    task=task_idx,
                    domain=task.name,
                    epoch=epoch,
                    total_epochs=epochs,
                    global_step=global_step,
                    batches=len(epoch_metrics),
                    epoch_tokens=epoch_tokens,
                    tokens=total_tokens,
                    epoch_time=duration,
                    mean_step_time=mean_step_time,
                    **epoch_writer_metrics,
                )

        # ---------------------------------------------------------
        # Save checkpoint without console logging
        # ---------------------------------------------------------

        saved_params = checkpoint_fn(params) if checkpoint_fn is not None else params

        writer.save_checkpoint(
            f"task_{task_idx}_{task.name}",
            saved_params,
            opt_state=None if checkpoint_fn is not None else optimizer_state,
            global_step=global_step,
            task=task_idx,
            domain=task.name,
            epoch=epochs,
            step=task_step,
            deployment_only=checkpoint_fn is not None,
        )

        # Evaluate after each task for continual-learning metrics.
        task_results.append(evaluate_all(task_idx))
        if after_task_fn is not None:
            after_task_fn(params, task_idx, task)
    # -------------------------------------------------------------
    # Final experiment summary
    # -------------------------------------------------------------

    summary = summarize(
        names,
        initial_results,
        task_results,
    )

    summary.update(
        epochs={task.name: task.epochs for task in tasks},
        steps=global_step,
        tokens=total_tokens,
        mean_step_time=(
            float(np.mean(step_times[1:])) if len(step_times) > 1 else None
        ),
    )

    return params, summary


__all__ = ["train"]
