"""Stage A1 loss computation."""

from __future__ import annotations

import jax

from frozenllm.substrate import FrozenSubstrate
from losses.shared import model_loss_terms


def compute_a1_loss(
    substrate: FrozenSubstrate,
    adapted_logits: jax.Array,
    labels: jax.Array,
    *,
    lambda_kl: float,
    lambda_wd: float,
    weight_decay: jax.Array,
    base: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Calculate the complete A1 objective and its individual components."""

    model_loss, task_loss, kl_loss = model_loss_terms(
        substrate,
        adapted_logits,
        labels,
        lambda_kl,
        base,
    )

    total_loss = model_loss + lambda_wd * weight_decay

    terms = {
        "total": total_loss,
        "task": task_loss,
        "wd": weight_decay,
    }

    if kl_loss is not None:
        terms["kl"] = kl_loss

    return total_loss, terms


__all__ = ["compute_a1_loss"]
