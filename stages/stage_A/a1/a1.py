"""Stage A1: standard residual adapter baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax

from frozenllm.substrate import FrozenSubstrate
from losses.a1_loss import compute_a1_loss
from residual import AdapterParams, ResidualAdapter
from stages.stage_base import base_logits, checked_kwargs


@dataclass(frozen=True)
class A1Config:
    lambda_kl: float = 0.0
    lambda_wd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lambda_kl", "lambda_wd"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A1Config:
        return cls(**checked_kwargs(cls, data))


def a1_forward(
    params: AdapterParams,
    batch: jax.Array,
    key: jax.Array | None = None,
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A1Config,
    base: jax.Array | None = None,
) -> dict[str, jax.Array | None]:
    """Run A1 forward pass and return predictions."""

    adapted = substrate.run_with_interception(
        batch,
        modify_fn=adapter.modify_fn(params),
        intercept_layers=adapter.layers,
    ).logits

    reference = base

    if config.lambda_kl > 0 and reference is None:
        reference = base_logits(substrate, batch)

    return {
        "logits": adapted,
        "base_logits": reference,
    }


def a1_loss(
    predictions: Mapping[str, jax.Array | None],
    batch: jax.Array,
    params: AdapterParams,
    config: A1Config,
    *,
    substrate: FrozenSubstrate,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Calculate A1 losses from model predictions."""

    return compute_a1_loss(
        substrate,
        predictions["logits"],
        batch,
        lambda_kl=config.lambda_kl,
        lambda_wd=config.lambda_wd,
        weight_decay=ResidualAdapter.l2_norm_sq(params),
        base=predictions["base_logits"],
    )


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
    """Compatibility objective used by existing A1 and A2 implementations."""

    predictions = a1_forward(
        params,
        input_ids,
        substrate=substrate,
        adapter=adapter,
        config=config,
        base=base,
    )

    return a1_loss(
        predictions,
        input_ids if labels is None else labels,
        params,
        config,
        substrate=substrate,
    )


__all__ = [
    "A1Config",
    "a1_forward",
    "a1_loss",
    "a1_objective",
]
