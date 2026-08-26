"""Frozen LLM substrate: a reusable JAX wrapper around pretrained causal LMs.

The base model parameters are completely frozen: they are stored as an
immutable Flax parameter PyTree, gradient flow is stopped with
``jax.lax.stop_gradient`` before any forward computation, and every forward is
a pure function of ``(params, input_ids)``. The wrapper only performs forward
computation and hidden-state interception.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .architecture import (
    Architecture,
    detect_architecture,
    validate_interception_layers,
)
from .memory import (
    MemoryStatus,
    check_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)
from .models import run_embeddings, run_lm_head, run_transformer_blocks


@dataclass(frozen=True)
class ForwardResult:
    """Output of a substrate forward pass.

    ``intermediates`` maps each intercepted zero-based layer index to the
    pre-modification hidden state cached at that layer.
    """

    logits: jax.Array
    intermediates: dict[int, jax.Array]

    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.intermediates.keys()))

    def hidden_state(self, layer_idx: int) -> jax.Array:
        if layer_idx not in self.intermediates:
            raise KeyError(
                f"Layer {layer_idx} was not intercepted. Intercepted layers: "
                f"{self.layer_indices()}"
            )
        return self.intermediates[layer_idx]

    def hidden_shapes(self) -> dict[int, tuple[int, ...]]:
        return {i: tuple(h.shape) for i, h in self.intermediates.items()}


jax.tree_util.register_dataclass(
    ForwardResult, data_fields=["logits", "intermediates"], meta_fields=[]
)


class FrozenJAXSubstrate:
    """Reusable frozen substrate around a pretrained GPT-2 or Pythia/GPT-NeoX
    causal LM.

    Args:
        params: Flax-convention parameter PyTree (nested mappings of arrays)
            with HuggingFace-compatible names. Created by
            ``state_dict_to_jax_pytree`` / ``load_substrate_from_hf``.
        config: optional HuggingFace config object used for hyperparameters
            such as head count, rope theta and layernorm epsilon. The layer
            count and hidden size are always auto-detected from ``params``.
        intercept_layers: zero-based layer indices at which hidden states are
            cached and passed through :meth:`intercept_and_modify`.
        modify_hook: optional ``(hidden_state, layer_idx) -> hidden_state``
            callable overriding the default identity interception. Must be
            JIT-trace-safe (pure array math only).
        min_memory_headroom: headroom ratio below which a warning is emitted.
    """

    def __init__(
        self,
        params: Any,
        config: Any = None,
        intercept_layers: Sequence[int] | None = None,
        modify_hook: Callable[[jax.Array, int], jax.Array] | None = None,
        min_memory_headroom: float = 0.5,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping (nested param PyTree)")

        self._architecture = detect_architecture(params, config)
        self._intercept_layers = validate_interception_layers(
            intercept_layers, self._architecture.num_layers
        )
        self._min_memory_headroom = float(min_memory_headroom)
        self._modify_hook = modify_hook
        # Params are stored as an immutable Flax FrozenDict. JAX arrays are
        # immutable, so this reference snapshot also captures the values; no
        # 2x copy is kept (memory-conscious).
        self._params = freeze(params)
        self._pristine = freeze(params)
        self._call_count = 0

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def architecture(self) -> Architecture:
        return self._architecture

    @property
    def intercept_layers(self) -> tuple[int, ...]:
        return self._intercept_layers

    @property
    def params(self) -> Any:
        return unfreeze(self._params)

    def get_params(self) -> Any:
        return unfreeze(self._params)

    def __call__(self, input_ids: jax.Array) -> ForwardResult:
        """Run the frozen substrate forward pass (JAX/JIT compatible)."""
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be a 2D array of shape [batch, seq_len], "
                f"got shape {tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one token position")

        params = jax.tree.map(jax.lax.stop_gradient, self._params)
        hook = self._modify_hook or self.intercept_and_modify
        logits, intermediates = self._run_forward(
            params, self._architecture, self._intercept_layers, hook, input_ids
        )
        self._call_count += 1
        return ForwardResult(logits=logits, intermediates=intermediates)

    # ── interception hook ───────────────────────────────────────────────────

    def intercept_and_modify(
        self, hidden_state: jax.Array, layer_idx: int
    ) -> jax.Array:
        """Default modification hook: an identity operation.

        Intentionally written as ``hidden_state + 0.0`` so it is JIT-trace-safe,
        does not convert tensors to NumPy, and preserves the numerical hidden
        state. Override in a subclass for custom steering.
        """
        return hidden_state + 0.0

    # ── freezing guarantees ─────────────────────────────────────────────────

    def params_unchanged(self) -> bool:
        """True when the params are still the same immutable JAX arrays that
        were captured at construction time. JAX arrays cannot be mutated in
        place, so this verifies that no replacement or reassignment ever
        happened."""
        identical = jax.tree.map(lambda a, b: a is b, self._pristine, self._params)
        return all(jax.tree.leaves(identical))

    def verify_frozen(self) -> dict[str, Any]:
        """Run the original-vs-wrapper param identity check and return a
        report. Base parameters must never be modified."""
        unchanged = self.params_unchanged()
        return {
            "params_unchanged": unchanged,
            "param_leaves": len(jax.tree.leaves(self._params)),
            "architecture": {
                "model_family": self._architecture.model_family,
                "num_layers": self._architecture.num_layers,
                "hidden_size": self._architecture.hidden_size,
            },
        }

    # ── memory monitoring ───────────────────────────────────────────────────

    def memory_status(self, device: jax.Device | None = None) -> MemoryStatus:
        return get_memory_status(device)

    def memory_warnings(self, status: MemoryStatus | None = None) -> list[str]:
        status = status or self.memory_status()
        return check_memory_headroom(status, self._min_memory_headroom)

    def run_with_memory_guard(
        self,
        input_ids: jax.Array,
        min_headroom: float | None = None,
        auto_reduce_batch_size: bool = False,
    ) -> tuple[ForwardResult, dict[str, Any]]:
        """Forward pass plus the memory headroom safety rule.

        When ``auto_reduce_batch_size`` is False (the default) an unsafe
        headroom only produces warnings and the configuration is untouched.
        When True, the batch is halved until the headroom rule is satisfied
        and the reduction is reported — never silently.
        """
        headroom = (
            min_headroom if min_headroom is not None else self._min_memory_headroom
        )
        status = self.memory_status()
        batch_size, reduced, warnings = maybe_reduce_batch_size(
            status, input_ids.shape[0], headroom, auto_reduce_batch_size
        )
        ids = input_ids[:batch_size] if reduced else input_ids
        result = self(ids)
        report = {
            "memory_status": status,
            "warnings": warnings,
            "batch_size_reduced": reduced,
            "effective_batch_size": ids.shape[0],
        }
        return result, report

    # ── forward internals (pure, JIT-safe) ──────────────────────────────────

    @staticmethod
    def _run_forward(
        params: Any,
        arch: Architecture,
        intercept_layers: tuple[int, ...],
        hook: Callable[[jax.Array, int], jax.Array],
        input_ids: jax.Array,
    ) -> tuple[jax.Array, dict[int, jax.Array]]:
        hidden = run_embeddings(params, arch, input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(input_ids.shape[1]), input_ids.shape)
        hidden, intermediates = run_transformer_blocks(
            params, arch, hidden, intercept_layers, hook, position_ids
        )
        logits = run_lm_head(params, arch, hidden)
        return logits, intermediates

    def __repr__(self) -> str:
        return (
            f"FrozenJAXSubstrate(model_family={self._architecture.model_family!r}, "
            f"num_layers={self._architecture.num_layers}, "
            f"hidden_size={self._architecture.hidden_size}, "
            f"intercept_layers={list(self._intercept_layers)})"
        )
