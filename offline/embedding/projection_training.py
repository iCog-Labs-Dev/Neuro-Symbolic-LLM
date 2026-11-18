"""Supervised, orthonormal projection learning over cached frozen features."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cosine_scores(projection, queries, templates):
    """Use the same projected cosine score for every candidate template."""
    q, k = queries @ projection.T, templates @ projection.T
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    k /= np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-12)
    return q @ k.T


def retrieval_report(scores, positives, top_k=8):
    """Report exact-ID precision/recall and bounds for tied cutoff scores."""
    if (
        scores.ndim != 2
        or not len(scores)
        or scores.shape != positives.shape
        or positives.dtype != bool
        or not positives.any()
        or not np.isfinite(scores).all()
        or not 0 < top_k <= scores.shape[1]
    ):
        raise ValueError("Scores, labels and top_k must align")
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    hits = np.take_along_axis(positives, ranked, axis=1).sum(axis=1)
    positive_rows = positives.any(axis=1)
    lower, upper = [], []
    for row, truth, selected in zip(scores, positives, ranked, strict=True):
        tied = np.isclose(row, row[selected[-1]], rtol=1e-6, atol=1e-6)
        higher = (row > row[selected[-1]]) & ~tied
        slots = top_k - int(higher.sum())
        certain = int(truth[higher].sum())
        tied_true, tied_false = int(truth[tied].sum()), int((~truth[tied]).sum())
        lower.append(certain + max(0, slots - tied_false))
        upper.append(certain + min(slots, tied_true))
    return {
        "queries": len(scores),
        "positive_queries": int(positive_rows.sum()),
        "top_k": top_k,
        "true_positives": int(hits.sum()),
        "micro_precision": float(hits.sum() / ranked.size),
        "micro_recall": float(hits.sum() / positives.sum()),
        "positive_hit_rate": float((hits[positive_rows] > 0).mean()),
        "tie_precision_lower": float(sum(lower) / ranked.size),
        "tie_precision_upper": float(sum(upper) / ranked.size),
    }


def fit_projection(
    initial,
    queries,
    templates,
    positives,
    allowed,
    validation_queries,
    validation_positives,
    *,
    epochs=20,
    batch_size=16,
    learning_rate=0.001,
    temperature=0.1,
    seed=42,
    report: Callable[[dict], None] | None = None,
):
    """Optimize only P; exclude unknown candidates and prevent scale collapse.

    Each example can have several known positives. Other candidates enter the
    denominator only when supplied as explicit negatives. QR retraction keeps
    P's rows orthonormal after each optimizer update. Validation selects a
    checkpoint but never contributes a gradient.
    """
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_default_matmul_precision", "highest")
    if (
        epochs < 1
        or batch_size < 1
        or not np.isfinite([learning_rate, temperature]).all()
        or learning_rate <= 0
        or temperature <= 0
        or positives.shape != allowed.shape
        or positives.shape != (len(queries), len(templates))
        or not positives.any(axis=1).all()
        or np.any(positives & ~allowed)
        or not (allowed & ~positives).any(axis=1).all()
        or not validation_positives.any()
    ):
        raise ValueError(
            "Projection training requires aligned positive/negative labels"
        )
    initial = np.asarray(initial, dtype=np.float32)
    if initial.ndim != 2 or initial.shape[1] != templates.shape[1]:
        raise ValueError("Projection and frozen features have incompatible shapes")
    np.testing.assert_allclose(initial @ initial.T, np.eye(len(initial)), atol=2e-5)
    if not all(
        np.isfinite(a).all() for a in (initial, queries, templates, validation_queries)
    ):
        raise ValueError("Projection features must be finite")
    bank = jnp.asarray(templates)
    start = jnp.asarray(initial)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    state = optimizer.init(start)

    def loss(p, q, targets, candidates, weights):
        projected_q, projected_k = q @ p.T, bank @ p.T
        projected_q /= jnp.maximum(
            jnp.linalg.norm(projected_q, axis=1, keepdims=True), 1e-12
        )
        projected_k /= jnp.maximum(
            jnp.linalg.norm(projected_k, axis=1, keepdims=True), 1e-12
        )
        scores = projected_q @ projected_k.T / temperature
        denominator = jax.nn.logsumexp(jnp.where(candidates, scores, -jnp.inf), axis=1)
        target_score = (scores * targets).sum(axis=1) / jnp.maximum(
            targets.sum(axis=1), 1
        )
        return ((denominator - target_score) * weights).sum() / weights.sum()

    @jax.jit
    def step(p, state, q, targets, candidates, weights):
        value, gradient = jax.value_and_grad(loss)(p, q, targets, candidates, weights)
        updates, state = optimizer.update(gradient, state, p)
        p = optax.apply_updates(p, updates)
        orthogonal, triangular = jnp.linalg.qr(p.T, mode="reduced")
        signs = jnp.where(jnp.diag(triangular) < 0, -1.0, 1.0)
        return (orthogonal * signs).T, state, value

    evaluate = jax.jit(loss)
    valid = validation_positives.any(axis=1)
    vq = jnp.asarray(validation_queries[valid])
    vp = jnp.asarray(validation_positives[valid])
    va, vw = jnp.ones_like(vp), jnp.ones(len(vq))
    p, best, best_epoch = start, initial.copy(), 0
    best_loss = float(evaluate(p, vq, vp, va, vw))
    history = [{"epoch": 0, "validation_loss": best_loss}]
    if report:
        report(history[0])
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(queries))
        total, count = 0.0, 0
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            real = len(indices)
            indices = np.pad(indices, (0, batch_size - real), mode="edge")
            p, state, value = step(
                p,
                state,
                jnp.asarray(queries[indices]),
                jnp.asarray(positives[indices]),
                jnp.asarray(allowed[indices]),
                jnp.asarray(np.arange(batch_size) < real),
            )
            total += float(value) * real
            count += real
        validation_loss = float(evaluate(p, vq, vp, va, vw))
        if not np.isfinite([total, validation_loss]).all():
            raise FloatingPointError("Nonfinite projection training loss")
        row = {
            "epoch": epoch,
            "training_loss": total / count,
            "validation_loss": validation_loss,
        }
        history.append(row)
        if report:
            report(row)
        if validation_loss < best_loss:
            best_loss, best_epoch, best = validation_loss, epoch, np.asarray(p).copy()
    return best, {"best_epoch": best_epoch, "history": history}
