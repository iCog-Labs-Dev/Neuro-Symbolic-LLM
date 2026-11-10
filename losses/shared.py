"""Loss calculations shared across training stages."""

from __future__ import annotations

import jax

from frozenllm.substrate import FrozenSubstrate


def model_loss_terms(
    substrate: FrozenSubstrate,
    adapted: jax.Array,
    labels: jax.Array,
    lambda_kl: float,
    base: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Calculate task CE and optional KL regularization.

    Returns:
        model_loss: CE plus weighted KL when a base reference is supplied.
        task_loss: CE component.
        kl_loss: Unweighted KL component, or None when not calculated.
    """
    if base is None:
        task = substrate.compute_loss(
            adapted,
            labels,
            losses=("CE",),
        )
        return task, task, None

    total, parts = substrate.compute_loss(
        adapted,
        labels,
        losses=("CE", "KL"),
        base_logits=jax.lax.stop_gradient(base),
        weights={"KL": lambda_kl},
        return_components=True,
    )

    return total, parts["ce"], parts["kl"]


__all__ = ["model_loss_terms"]
