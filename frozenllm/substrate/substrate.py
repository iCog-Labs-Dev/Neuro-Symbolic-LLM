"""Frozen LLM substrate using monolithic TorchAX execution.

The base model parameters theta_0 are strictly frozen (requires_grad=False).
Forward passes execute the monolithic model on TorchAX with per-layer
hidden-state interception via forward hooks, returning JAX-compatible
logits and intermediate states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import torch
from flax.core import freeze, unfreeze

from frozenllm.substrate.architecture import (
    Architecture,
    detect_architecture,
    validate_interception_layers,
)
from frozenllm.substrate.interception import identity_modify, run_with_hooks
from frozenllm.substrate.memory import (
    MemoryStatus,
    check_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)

from .loader import load_tokenizer, load_torchax_model
from .torchax_backend import (
    enable_torchax,
    from_jax_array,
    is_on_torchax_device,
    to_jax_array,
    to_torchax_device,
)


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


class FrozenSubstrate:
    """Top-level frozen LLM substrate using monolithic TorchAX execution.

    PyTorch model / Hugging Face checkpoint
              │
              ▼
           TorchAX
              │
              ▼
         JAX-backed execution

    The base model parameters theta_0 are strictly frozen (requires_grad=False).
    Forward passes execute the monolithic model on TorchAX with per-layer
    hidden-state interception via forward hooks, returning JAX-compatible
    logits and intermediate states.
    """

    def __init__(
        self,
        model_id_or_model: str | torch.nn.Module | Mapping[str, Any] | None = None,
        config: Any = None,
        intercept_layers: Sequence[int] | None = None,
        modify_hook: Callable[[jax.Array, int], jax.Array] | None = None,
        min_memory_headroom: float = 0.5,
        tokenizer: Any = None,
        params: Mapping[str, Any] | None = None,
        dtype: torch.dtype | str | None = None,
        attn_implementation: str | None = "eager",
        torch_dtype: torch.dtype | str | None = None,
    ) -> None:
        if model_id_or_model is None:
            if params is not None:
                model_id_or_model = params
            else:
                raise TypeError(
                    "FrozenSubstrate requires either a model_id, torch.nn.Module, or params Mapping"
                )

        self._min_memory_headroom = float(min_memory_headroom)
        self._modify_hook = modify_hook
        self._call_count = 0
        self._tokenizer = tokenizer

        # Legacy JAX param PyTree support for backward compatibility with older test fixtures
        if isinstance(model_id_or_model, Mapping):
            self._legacy_mode = True
            self._model = None
            self._architecture = detect_architecture(model_id_or_model, config)
            self._intercept_layers = validate_interception_layers(
                intercept_layers, self._architecture.num_layers
            )
            self._params = freeze(model_id_or_model)
            self._pristine = freeze(model_id_or_model)
            return

        self._legacy_mode = False

        # Handle positional params passed as second argument: Substrate(model, params)
        if isinstance(config, Mapping) and params is None:
            params = config
            config = getattr(model_id_or_model, "config", None)

        # 1. Load or accept PyTorch model and tokenizer
        if isinstance(model_id_or_model, str):
            effective_dtype = dtype or torch_dtype or torch.float32
            model, params = load_torchax_model(
                model_id_or_model,
                dtype=effective_dtype,
                attn_implementation=attn_implementation,
            )
            config = config or getattr(model, "config", None)
            if self._tokenizer is None:
                try:
                    self._tokenizer = load_tokenizer(model_id_or_model)
                except Exception:
                    self._tokenizer = None
        elif isinstance(model_id_or_model, torch.nn.Module):
            model = model_id_or_model
            config = config or getattr(model, "config", None)
            enable_torchax()
            model = to_torchax_device(model)
            model.eval()
            params = dict(model.named_parameters()) if params is None else dict(params)
            for p in params.values():
                p.requires_grad_(False)
        else:
            raise TypeError(
                f"Expected model_id (str), torch.nn.Module, or param Mapping, "
                f"got {type(model_id_or_model)}"
            )

        self._model = model
        self._params = dict(params)

        # 2. Detect architecture
        if params is not None and (config is None or not hasattr(config, "model_type")):
            self._architecture = detect_architecture(params, config)
        elif config is not None:
            self._architecture = detect_architecture(config)
        elif hasattr(model, "config") and model.config is not None:
            self._architecture = detect_architecture(model.config)
        else:
            raise ValueError(
                "Model configuration must be provided or available on model.config."
            )

        # 3. Validate interception layers
        self._intercept_layers = validate_interception_layers(
            intercept_layers, self._architecture.num_layers
        )

        # 6. Store pristine parameter snapshot for immutability verification
        self._pristine = {k: v.detach().clone() for k, v in self._params.items()}

    # ── public properties ───────────────────────────────────────────────────

    @property
    def architecture(self) -> Architecture:
        return self._architecture

    @property
    def intercept_layers(self) -> tuple[int, ...]:
        return self._intercept_layers

    @property
    def params(self) -> Any:
        if self._legacy_mode:
            return unfreeze(self._params)
        return self._params

    def get_params(self) -> Any:
        if self._legacy_mode:
            return unfreeze(self._params)
        return self._params

    @property
    def tokenizer(self) -> Any:
        """Tokenizer associated with this substrate (if loaded or provided)."""
        return self._tokenizer

    def tokenize(
        self,
        text: str | list[str],
        return_tensors: str = "jax",
        **kwargs: Any,
    ) -> jax.Array | torch.Tensor:
        """Tokenize input text directly into device-ready token IDs using torchax_backend.

        Args:
            text: Input text string or list of text strings.
            return_tensors: "jax" (default) or "pt".
            **kwargs: Additional kwargs passed to the Hugging Face tokenizer.

        Returns:
            2D array/tensor of token IDs ready for forward execution.
        """
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer is not initialized. Initialize FrozenSubstrate with a model ID string or provide a tokenizer."
            )
        tokens = self._tokenizer(text, return_tensors="pt", **kwargs)
        ids_torch = to_torchax_device(tokens["input_ids"])
        if return_tensors == "jax":
            return to_jax_array(ids_torch)
        return ids_torch

    # ── forward execution ───────────────────────────────────────────────────

    def __call__(self, input_ids: jax.Array | torch.Tensor) -> ForwardResult:
        """Run the frozen substrate forward pass."""
        return self.run_with_interception(
            input_ids=input_ids,
            modify_fn=self._modify_hook,
            intercept_layers=self._intercept_layers,
        )

    def run_with_interception(
        self,
        input_ids: jax.Array | torch.Tensor,
        modify_fn: Callable[..., Any] | None = None,
        intercept_layers: Sequence[int] | None = None,
    ) -> ForwardResult:
        """Explicit interception API with custom hook and layers."""
        hook = modify_fn or self._modify_hook or identity_modify
        layers = (
            validate_interception_layers(
                intercept_layers, self._architecture.num_layers
            )
            if intercept_layers is not None
            else self._intercept_layers
        )

        # Handle legacy pure-JAX execution
        if self._legacy_mode:
            if not isinstance(input_ids, jax.Array):
                raise TypeError(
                    f"Legacy mode expects jax.Array input_ids, got {type(input_ids)}"
                )
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must be a 2D array of shape [batch, seq_len], "
                    f"got shape {tuple(input_ids.shape)}"
                )
            if input_ids.shape[1] < 1:
                raise ValueError("input_ids must contain at least one token position")
            params = jax.tree.map(jax.lax.stop_gradient, self._params)
            logits, intermediates = self._run_forward_legacy(
                params, self._architecture, layers, hook, input_ids
            )
            self._call_count += 1
            return ForwardResult(logits=logits, intermediates=intermediates)

        # Monolithic TorchAX execution
        if isinstance(input_ids, jax.Array):
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must be a 2D array of shape [batch, seq_len], "
                    f"got shape {tuple(input_ids.shape)}"
                )
            if input_ids.shape[1] < 1:
                raise ValueError("input_ids must contain at least one token position")
            input_ids_torch = from_jax_array(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must be a 2D tensor of shape [batch, seq_len], "
                    f"got shape {tuple(input_ids.shape)}"
                )
            if input_ids.shape[1] < 1:
                raise ValueError("input_ids must contain at least one token position")
            input_ids_torch = input_ids
        else:
            raise TypeError(
                f"input_ids must be jax.Array or torch.Tensor, got {type(input_ids)}"
            )

        if not is_on_torchax_device(input_ids_torch):
            input_ids_torch = to_torchax_device(input_ids_torch)

        output, intermediates = run_with_hooks(
            model=self._model,
            params=self._params,
            input_ids=input_ids_torch,
            arch=self._architecture,
            intercept_layers=layers,
            modify_fn=hook,
            to_jax=True,
        )

        logits_raw = output.logits if hasattr(output, "logits") else output
        logits_jax = to_jax_array(logits_raw)
        self._call_count += 1
        return ForwardResult(logits=logits_jax, intermediates=intermediates)

    # ── loss computation ────────────────────────────────────────────────────

    @staticmethod
    def compute_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Standard causal LM cross-entropy loss with shifted labels.

        Args:
            logits: Predicted unnormalized logits of shape [batch, seq_len, vocab_size].
            labels: Target token IDs of shape [batch, seq_len]. Tokens with label -100
                are ignored in the loss calculation.

        Returns:
            Scalar jax.Array representing the mean cross-entropy loss.
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        mask = shift_labels != -100
        safe_labels = jnp.where(mask, shift_labels, 0)

        gathered_log_probs = jnp.take_along_axis(
            log_probs, safe_labels[..., None], axis=-1
        ).squeeze(-1)

        loss = -jnp.sum(gathered_log_probs * mask) / jnp.maximum(1.0, jnp.sum(mask))
        return loss

    # ── freezing guarantees ─────────────────────────────────────────────────

    def params_unchanged(self) -> bool:
        """Verify that base model parameters theta_0 have not been modified."""
        if self._legacy_mode:
            identical = jax.tree.map(lambda a, b: a is b, self._pristine, self._params)
            return all(jax.tree.leaves(identical))

        for k, pristine_val in self._pristine.items():
            current_val = self._params.get(k)
            if current_val is None:
                return False
            if current_val.requires_grad:
                return False
            if not torch.equal(pristine_val, current_val):
                return False
        return True

    def verify_frozen(self) -> dict[str, Any]:
        """Run original-vs-wrapper parameter verification and return a report."""
        unchanged = self.params_unchanged()
        param_count = (
            len(jax.tree.leaves(self._params))
            if self._legacy_mode
            else len(self._params)
        )
        return {
            "params_unchanged": unchanged,
            "param_leaves": param_count,
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
        """Forward pass plus the memory headroom safety rule."""
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

    # ── forward internals (pure, JIT-safe legacy fallback) ──────────────────

    @staticmethod
    def _run_forward_legacy(
        params: Any,
        arch: Architecture,
        intercept_layers: tuple[int, ...],
        hook: Callable[[jax.Array, int], jax.Array],
        input_ids: jax.Array,
    ) -> tuple[jax.Array, dict[int, jax.Array]]:
        raise NotImplementedError(
            "Legacy pure-JAX forward execution via models.py was removed in favor of "
            "monolithic TorchAX execution. Use FrozenSubstrate with live models."
        )

    def __repr__(self) -> str:
        return (
            f"FrozenSubstrate(model_family={self._architecture.model_family!r}, "
            f"num_layers={self._architecture.num_layers}, "
            f"hidden_size={self._architecture.hidden_size}, "
            f"intercept_layers={list(self._intercept_layers)})"
        )


# Backward compatibility alias
Substrate = FrozenSubstrate
