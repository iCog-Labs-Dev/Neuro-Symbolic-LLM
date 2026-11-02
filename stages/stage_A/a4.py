"""Symbolic forward composition at final predictive-coding latents.

Accepts final ``aux['z_star']`` latents from ``A3Model.train_grads`` and applies
symbolic attention after continuous residual injection. Latent inference,
alignment losses and optimizer updates are outside this forward interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from frozenllm.substrate import ForwardResult
from residual.symbolic_head.jax_head import HeadParams, SymbolicHead
from stages.stage_A.A3_model import A3Model
from stages.stage_A.a3_tail import residual_modify


class A4Model:
    """Compose a continuous stage with a bank-bound symbolic head.

    Uses the A3 and symbolic-head parameter layouts without conversion.
    Construction performs no I/O. Symbolic layers must be a subset of the
    continuous stage's adapted layers.
    """

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
        """Forward at caller-supplied final latents, without further settlement.

        At each adapted layer: h_pc = h_live + scale * z_star @ B, followed by
        the symbolic query, retrieval, attention and writeback when enabled.
        Earlier modified layers flow into later ones through the live model.

        ``symbolic_enabled`` is a static Python bool when compiling with JAX.
        Disabling it bypasses the head entirely: no query projection or network
        callback is executed, and symbolic parameters are not inspected.

        Returned intermediates contain the substrate's pre-hook hidden states.
        ``z_star`` must correspond to ``input_ids`` and ``continuous_params``.
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
        """Return the forward result and live pre-symbolic post-PC states.

        States are captured at head-bearing layers within the same forward pass,
        preserving earlier-layer corrections and gradients to continuous weights.
        """
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
