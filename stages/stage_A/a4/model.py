"""Symbolic attention after continuous residual injection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from frozenllm.substrate import ForwardResult
from residual.symbolic_head.jax_head import HeadParams, SymbolicHead
from stages.stage_A.A3_model import A3Model
from stages.stage_A.a3_tail import residual_modify


class A4Model:
    """Compose a continuous residual with a bank-bound symbolic head."""

    def __init__(self, continuous: A3Model, symbolic: SymbolicHead) -> None:
        if symbolic.hidden_size != continuous.hidden_size:
            raise ValueError("Continuous stage and symbolic head hidden sizes differ")
        if not set(symbolic.layers).issubset(continuous.layers):
            raise ValueError("Symbolic layers must belong to the continuous stage")
        self.continuous = continuous
        self.symbolic = symbolic

    def forward_at_latents(
        self,
        continuous_params: Any,
        symbolic_params: HeadParams,
        input_ids: jax.Array,
        z_star: Mapping[int, jax.Array],
        *,
        symbolic_enabled: bool = True,
    ) -> ForwardResult:
        """Apply the residual and symbolic head at caller-supplied latents.

        symbolic_enabled is static under JIT; False bypasses retrieval.
        """
        result, _ = self.forward_with_states_at_latents(
            continuous_params,
            symbolic_params,
            input_ids,
            z_star,
            symbolic_enabled=symbolic_enabled,
        )
        return result

    def forward_with_states_at_latents(
        self,
        continuous_params: Any,
        symbolic_params: HeadParams,
        input_ids: jax.Array,
        z_star: Mapping[int, jax.Array],
        *,
        symbolic_enabled: bool = True,
    ) -> tuple[ForwardResult, dict[int, jax.Array]]:
        """Return logits and post-residual states at symbolic-head layers."""
        if not isinstance(symbolic_enabled, bool):
            raise TypeError("symbolic_enabled must be a static Python bool")
        if input_ids.ndim != 2 or input_ids.shape[1] != self.continuous.seq_len:
            raise ValueError(
                f"input_ids must have shape [batch, {self.continuous.seq_len}]"
            )
        if set(z_star) != set(self.continuous.layers):
            raise ValueError("Final latents must cover every continuous adapted layer")
        projections = {
            layer: self.continuous.projection(continuous_params, layer)
            for layer in self.continuous.layers
        }
        continuous_modify = residual_modify(z_star, projections, self.continuous.scale)
        states: dict[int, jax.Array] = {}

        def modify(hidden: jax.Array, layer: int) -> jax.Array:
            settled = continuous_modify(hidden, layer)
            if layer in self.symbolic.layers:
                states[layer] = settled
            return (
                self.symbolic.apply(symbolic_params, settled, layer)
                if symbolic_enabled
                else settled
            )

        result = self.continuous.sub.run_with_interception(
            input_ids,
            modify_fn=modify,
            intercept_layers=self.continuous.layers,
        )
        return result, states


__all__ = ["A4Model"]
