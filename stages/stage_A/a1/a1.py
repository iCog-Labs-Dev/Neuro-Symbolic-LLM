"""Stage A1: plain residual baseline (no predictive coding, no symbolic head).

L_A1 = L_task + lambda_kl * KL(p_F0 || p_F0+R) + lambda_wd * ||phi||^2
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax

from frozenllm.substrate import FrozenSubstrate
from residual import AdapterParams, ResidualAdapter
from stages.stage_base import base_logits, checked_kwargs, model_loss_terms


@dataclass(frozen=True)
class A1Config:
    lambda_kl: float = 0.0
    lambda_wd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lambda_kl", "lambda_wd"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A1Config:
        return cls(**checked_kwargs(cls, data))


def a1_objective(
    params: AdapterParams,
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A1Config,
    input_ids: jax.Array,
    labels: jax.Array | None = None,
    base: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    adapted = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(params),
        intercept_layers=adapter.layers,
    ).logits
    target_labels = input_ids if labels is None else labels
    wd = ResidualAdapter.l2_norm_sq(params)

    reference = None
    if config.lambda_kl > 0 or base is not None:
        reference = base_logits(substrate, input_ids) if base is None else base
    model_loss, task, kl = model_loss_terms(
        substrate, adapted, target_labels, config.lambda_kl, reference
    )
    total = model_loss + config.lambda_wd * wd
    terms = {"total": total, "task": task, "wd": wd}
    if kl is not None:
        terms["kl"] = kl
    return total, terms


__all__ = ["A1Config", "a1_objective"]
