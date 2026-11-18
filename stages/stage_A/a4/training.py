"""Symbolic-head training on a frozen continuous residual."""

from __future__ import annotations

import hashlib
import json
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
from stages.stage_A.a4.checkpoint import (
    restore_training_checkpoint,
    save_training_checkpoint,
)
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


def init_params(model: A4Model, seed: int, projection=None) -> dict[str, Any]:
    """Initialize query, writeback and value-prediction parameters."""
    predictor = ValuePredictor(model.symbolic.hidden_size, model.symbolic.key_dim)
    params = {
        "symbolic": model.symbolic.init_params(seed),
        "value": {
            f"layer_{layer}": predictor.init_params(seed + layer)
            for layer in model.symbolic.layers
        },
    }
    if projection is not None:
        params["projection"] = {"bank": {"P": jnp.asarray(projection)}}
    return params


def load_projection_artifacts(directory, bank, initialization="learned"):
    """Validate a projection-only experiment against the unchanged source bank."""
    import json

    report = json.loads((directory / "result.json").read_text())
    if (
        report.get("status") != "completed"
        or report.get("bank_snapshot") != bank.input_snapshot_sha256
    ):
        raise ValueError(
            "Projection artifacts belong to another or incomplete bank experiment"
        )
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        features = archive["templates"].copy()
    with np.load(directory / "projection.npz", allow_pickle=False) as archive:
        initial = archive["initial"].copy()
        projection = archive[
            "projection" if initialization == "learned" else "initial"
        ].copy()
    if (
        features.ndim != 2
        or features.shape[0] != len(bank.records)
        or projection.shape != (bank.vector_dim, features.shape[1])
    ):
        raise ValueError("Projection feature dimensions differ from bank")
    if any(not np.isfinite(x).all() for x in (features, initial, projection)):
        raise ValueError("Projection artifacts must be finite")
    np.testing.assert_allclose(
        features @ initial.T,
        np.stack([r.key_vector for r in bank.records]),
        rtol=1e-5,
        atol=2e-4,
    )
    if any(not np.array_equal(r.key_vector, r.val_vector) for r in bank.records):
        raise ValueError(
            "Joint shared projection requires initially equal keys and values"
        )
    np.testing.assert_allclose(
        projection @ projection.T, np.eye(bank.vector_dim), atol=3e-5
    )
    return features.astype(np.float32), projection.astype(np.float32)


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
    train_bank_projection=False,
):
    """Build a head-only or explicitly opted-in joint projection update."""
    frozen_residual = jax.tree.map(jax.lax.stop_gradient, frozen_residual)

    def step(params, state, ids, positions, targets, token_mask=None):
        dynamic = model.bank_features is not None
        if set(params) != (
            {"symbolic", "value", "projection"} if dynamic else {"symbolic", "value"}
        ):
            raise ValueError(
                "A4 optimizer parameters must contain only symbolic and value groups"
            )
        clean, _ = model.continuous.clean_pass(ids)
        latents = jax.tree.map(
            jax.lax.stop_gradient,
            model.continuous.predict_latents(frozen_residual, clean),
        )

        def loss(candidate):
            if dynamic and not train_bank_projection:
                candidate = {
                    **candidate,
                    "projection": jax.tree.map(
                        jax.lax.stop_gradient, candidate["projection"]
                    ),
                }
            vectors = model.bank_vectors(candidate)
            keys, values = vectors if vectors is not None else (bank_keys, bank_values)
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
                keys,
                values,
                key_weight=config.key_weight,
                value_weight=config.value_weight,
                token_mask=token_mask,
                train_bank_keys=train_bank_projection,
                dynamic_bank=dynamic,
            )

        (_, terms), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, next_state = optimizer.update(grads, state, params)
        candidate = optax.apply_updates(params, updates)
        if train_bank_projection:
            p = candidate["projection"]["bank"]["P"]
            orthogonal, triangular = jnp.linalg.qr(p.T, mode="reduced")
            candidate["projection"]["bank"]["P"] = (
                orthogonal * jnp.where(jnp.diag(triangular) < 0, -1.0, 1.0)
            ).T
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
        if dynamic:
            terms["projection_grad_norm"] = optax.global_norm(grads["projection"])
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
            bank_vectors=model.bank_vectors(head_params),
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
    projection_features=None,
    initial_projection=None,
    train_bank_projection: bool = False,
    checkpoint_every: int = 250,
    resume_checkpoint=None,
) -> dict[str, Any]:
    """Train the symbolic head and save its checkpoint and evaluation metrics."""
    training = config.training
    if latency_every < 0:
        raise ValueError("latency_every must be nonnegative")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    retriever = model.symbolic.retrieve
    dynamic = projection_features is not None
    if dynamic != (initial_projection is not None) or (
        train_bank_projection and not dynamic
    ):
        raise ValueError(
            "Projection training requires aligned features and an initial projection"
        )
    if dynamic:
        if not callable(getattr(retriever, "refresh_vectors", None)):
            raise ValueError("Dynamic bank projection requires native vector refresh")
        if projection_features.shape[1] != model.symbolic.hidden_size:
            raise ValueError("Raw template features differ from model hidden size")
        model.bank_features = jnp.asarray(projection_features)
    timing_enabled = bool(
        latency_every
        and not dynamic
        and training.jit
        and callable(getattr(retriever, "transfer_timing", None))
        and callable(getattr(retriever, "transfer_stats", None))
    )
    communication_samples = []
    seen_shapes = set()
    fingerprint = residual_fingerprint(model, frozen_residual)
    params = init_params(model, training.seed, initial_projection)

    def refresh(candidate):
        if dynamic:
            keys, values = (np.asarray(x) for x in model.bank_vectors(candidate))
            retriever.refresh_vectors(keys, values)

    if dynamic:
        retriever.info["no_gpu_bank_cache"] = False
        log.info(
            "Bank projection: %s; native FAISS vectors refreshed between updates; value targets detached",
            "joint" if train_bank_projection else "fixed",
        )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adam(training.learning_rate)
    )
    state = optimizer.init(params)
    settings = config.to_dict()
    for name in ("epochs", "output_dir", "log_every"):
        settings["training"].pop(name, None)
    settings.pop("residual_config")
    signature_inputs = {
        "config": settings,
        "data": data.provenance,
        "bank": bank.input_snapshot_sha256,
        "frozen": fingerprint,
        "layers": model.symbolic.layers,
        "max_grad_norm": max_grad_norm,
        "joint": train_bank_projection,
        "protocol": TRAINING_STATE_SOURCE,
        "features": (
            hashlib.sha256(np.asarray(projection_features).tobytes()).hexdigest()
            if dynamic
            else None
        ),
        "initial_projection": (
            hashlib.sha256(np.asarray(initial_projection).tobytes()).hexdigest()
            if dynamic
            else None
        ),
    }
    signature = hashlib.sha256(
        json.dumps(signature_inputs, sort_keys=True, default=str).encode()
    ).hexdigest()
    batches_per_epoch = len(data.train) // training.batch_size
    updates, first_epoch, first_batch = 0, 0, 0
    recovered = None
    if resume_checkpoint is not None:
        params, state, recovered = restore_training_checkpoint(
            resume_checkpoint, params, optimizer, signature
        )
        updates, first_epoch, first_batch = (
            recovered.get(k) for k in ("updates", "next_epoch", "next_batch")
        )
        if (
            any(
                not isinstance(x, int) or isinstance(x, bool) or x < 0
                for x in (updates, first_epoch, first_batch)
            )
            or first_batch >= batches_per_epoch
            or updates != first_epoch * batches_per_epoch + first_batch
            or first_epoch > training.epochs
            or (first_epoch == training.epochs and first_batch)
        ):
            raise ValueError(
                "Checkpoint batch cursor differs from requested training schedule"
            )
        writer.log(
            "resume",
            checkpoint=str(resume_checkpoint),
            updates=updates,
            next_epoch=first_epoch,
            next_batch=first_batch,
        )
        log.info(
            "Resuming after update %d; next epoch %d batch %d",
            updates,
            first_epoch + 1,
            first_batch + 1,
        )
    refresh(params)
    writer.save_params("a3-frozen", model.continuous.adapter_params(frozen_residual))
    step = make_train_step(
        model,
        frozen_residual,
        optimizer,
        jnp.asarray(np.stack([record.key_vector for record in bank.records])),
        jnp.asarray(np.stack([record.val_vector for record in bank.records])),
        config,
        **({"train_bank_projection": True} if train_bank_projection else {}),
    )
    log.info("Task 1/1: A4 symbolic-head training")
    if timing_enabled:
        log.info(
            "CUDA copy timing: every %d updates after 10-update and input-shape warmup",
            latency_every,
        )
    else:
        log.info(
            "CUDA copy qualification disabled for projection experiments"
            if dynamic
            else "Communication timing disabled or unavailable (requires compiled native CUDA retrieval)"
        )
    log.info("Initial validation...")
    validation = evaluate(
        model, frozen_residual, params, data.validation, training, data.validation_mask
    )
    initial_validation = (
        dict(validation) if recovered is None else recovered["initial_validation"]
    )
    writer.log(
        "eval",
        split="validation",
        epoch=first_epoch,
        resumed=resume_checkpoint is not None,
        **validation,
    )
    log.info(
        "Validation: task loss=%.6f PPL=%.4f", validation["loss"], validation["ppl"]
    )
    step_times = []
    launch_updates = updates
    columns = (
        f"{'Batch':>12} {'Total loss':>14} {'Task loss':>14} "
        f"{'Symbolic loss':>14} {'PPL':>12}"
    )
    if dynamic:
        columns += f" {'P grad norm':>12}"
    for epoch in range(first_epoch, training.epochs):
        epoch_samples = []
        log.info("Epoch %d/%d", epoch + 1, training.epochs)
        log.info("%s", columns)
        order = np.random.default_rng(training.seed + epoch).permutation(
            len(data.train)
        )
        for start in range(
            (first_batch * training.batch_size if epoch == first_epoch else 0),
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
                and updates - launch_updates >= 10
                and shape in seen_shapes
                and (updates + 1) % latency_every == 0
            )
            with _copy_sample(retriever, measure) as sample:
                started = time.perf_counter()
                params, state, terms, finite = step(params, state, *inputs)
                jax.block_until_ready((params, state, terms, finite))
                if measure:
                    jax.effects_barrier()
                elapsed = time.perf_counter() - started
            seen_shapes.add(shape)
            if not bool(finite):
                raise FloatingPointError("Nonfinite loss, gradient or optimizer update")
            if train_bank_projection:
                refresh(params)
                elapsed = time.perf_counter() - started
            if measure:
                if sample["calls"] != len(model.symbolic.layers):
                    raise RuntimeError("CUDA timing did not cover every symbolic layer")
                sample.update(step_seconds=elapsed, step=updates + 1, epoch=epoch + 1)
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
                row = training_row(batch, batches_per_epoch, metrics)
                if dynamic:
                    row += f" {metrics['projection_grad_norm']:12.4f}"
                log.info("%s", row)
            if updates % checkpoint_every == 0 or batch == batches_per_epoch:
                path = save_training_checkpoint(
                    writer,
                    "a4-latest",
                    params,
                    state,
                    resume_version=1,
                    resume_signature=signature,
                    updates=updates,
                    next_epoch=epoch + 1 if batch == batches_per_epoch else epoch,
                    next_batch=0 if batch == batches_per_epoch else batch,
                    initial_validation=initial_validation,
                    training_state_source=TRAINING_STATE_SOURCE,
                    frozen_residual_sha256=fingerprint,
                    bank_snapshot_sha256=bank.input_snapshot_sha256,
                )
                writer.log("checkpoint", step=updates, path=str(path))
                writer.write_json(
                    "status.json",
                    {
                        "stage": "A4",
                        "status": "running",
                        "updates": updates,
                        "checkpoint": str(path),
                    },
                )
                if batch != batches_per_epoch:
                    log.info("Checkpoint saved at update %d: %s", updates, path)
        log.info("%s", columns)
        epoch_communication = _communication_summary(epoch_samples)
        writer.log("communication", epoch=epoch + 1, **epoch_communication)
        _log_communication(epoch_communication)
        validation = evaluate(
            model,
            frozen_residual,
            params,
            data.validation,
            training,
            data.validation_mask,
        )
        writer.log("eval", split="validation", epoch=epoch + 1, **validation)
        log.info(
            "Validation: task loss=%.6f PPL=%.4f", validation["loss"], validation["ppl"]
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
        "resumed_from": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "timing_scope": "Current process segment only; first update after launch excluded",
        "frozen_residual_sha256": fingerprint,
        "frozen": frozen,
        "validation": validation,
        "initial_validation": initial_validation,
        "validation_ce_change": validation["loss"] - initial_validation["loss"],
        "training_examples": len(data.train),
        "alignment": data.provenance.get("alignment", {}),
        "communication": _communication_summary(communication_samples),
        "bank_projection": {
            "mode": (
                "joint"
                if train_bank_projection
                else ("fixed-projected" if dynamic else "fixed-bank")
            ),
            "native_refreshes": getattr(retriever, "info", {}).get(
                "working_bank_refreshes", 0
            ),
            "value_targets_detached": True,
            "step_timing_includes_bank_refresh": train_bank_projection,
            "transfer_only_qualification": False if dynamic else None,
            "projection_change_norm": (
                float(
                    np.linalg.norm(
                        np.asarray(params["projection"]["bank"]["P"])
                        - initial_projection
                    )
                )
                if dynamic
                else None
            ),
        },
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
    writer.save_params("a3-frozen", model.continuous.adapter_params(frozen_residual))
    save_training_checkpoint(
        writer,
        "a4-final",
        params,
        state,
        resume_version=1,
        resume_signature=signature,
        next_epoch=training.epochs,
        next_batch=0,
        initial_validation=initial_validation,
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


__all__ = [
    "evaluate",
    "init_params",
    "make_train_step",
    "residual_fingerprint",
    "train",
]
