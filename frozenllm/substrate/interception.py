"""Hidden-state interception for frozen LLM substrates.

Provides mechanisms for intercepting, caching, and modifying transformer
hidden states at designated layers during forward execution:

   Hook-based interception (InterceptionContext, run_with_hooks):
   Registers forward hooks on the model's transformer blocks so that
   torch.func.functional_call executes with pristine state caching and
   in-flight hidden state replacement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any

import jax
import torch
from torch.func import functional_call

from .architecture import (
    Architecture,
    detect_architecture,
    get_block_accessor,
    validate_interception_layers,
)
from .torchax_backend import from_jax_array, to_jax_array

# ModifyFn contract: (hidden_state, layer_idx) -> modified_hidden_state
ModifyFn = Callable[[Any, int], Any]


def identity_modify(hidden: Any, layer_idx: int) -> Any:
    return hidden


def _extract_hidden(output: Any) -> tuple[Any, bool, tuple[Any, ...]]:
    if isinstance(output, tuple):
        return output[0], True, output[1:]
    return output, False, ()


def _wrap_hidden(hidden: Any, is_tuple: bool, rest: tuple[Any, ...]) -> Any:
    if is_tuple:
        return (hidden, *rest)
    return hidden


class InterceptionContext(AbstractContextManager["InterceptionContext"]):
    def __init__(
        self,
        model: torch.nn.Module,
        arch: Architecture | None = None,
        intercept_layers: Sequence[int] | None = None,
        modify_fn: ModifyFn | None = None,
        clone_intermediates: bool = True,
        to_jax: bool = False,
    ):
        self.model = model
        if arch is None:
            if getattr(model, "config", None) is not None:
                arch = detect_architecture(model)
            else:
                raise ValueError(
                    "arch must be provided when model does not have a config attribute."
                )
        self.arch = arch
        self.intercept_layers = validate_interception_layers(
            intercept_layers, arch.num_layers
        )
        self.modify_fn = modify_fn or identity_modify
        self.clone_intermediates = clone_intermediates
        self.to_jax = to_jax
        self.intermediates: dict[int, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> InterceptionContext:
        self.intermediates.clear()
        self._handles.clear()

        if not self.intercept_layers:
            return self

        accessor = get_block_accessor(self.model, self.arch)

        for layer_idx in self.intercept_layers:
            block = accessor(layer_idx)
            handle = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int) -> Callable[..., Any]:
        def hook(module: Any, args: Any, output: Any) -> Any:
            h, is_tuple, rest = _extract_hidden(output)

            # 1. Cache pristine h_l^0 (pre-modification state)
            h_cached = (
                h.clone() if (self.clone_intermediates and hasattr(h, "clone")) else h
            )
            if self.to_jax:
                self.intermediates[layer_idx] = to_jax_array(h_cached)
                h_input = to_jax_array(h)
            else:
                self.intermediates[layer_idx] = h_cached
                h_input = h

            # 2. Apply modify_fn to pristine state
            h_mod = self.modify_fn(h_input, layer_idx)

            # Convert back to torch/torchax tensor if a JAX array was returned
            if hasattr(h_mod, "__class__") and "jax" in str(type(h_mod)).lower():
                h_modified = from_jax_array(h_mod)
            elif isinstance(h_mod, torch.Tensor):
                h_modified = h_mod
            else:
                try:
                    if isinstance(h_mod, jax.Array):
                        h_modified = from_jax_array(h_mod)
                    else:
                        h_modified = h_mod
                except Exception:
                    h_modified = h_mod

            # 3. Return modified state to flow into downstream blocks
            return _wrap_hidden(h_modified, is_tuple, rest)

        return hook


def run_with_hooks(
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    arch: Architecture | None = None,
    intercept_layers: Sequence[int] | None = None,
    modify_fn: ModifyFn | None = None,
    to_jax: bool = False,
) -> tuple[Any, dict[int, Any]]:
    with InterceptionContext(
        model=model,
        arch=arch,
        intercept_layers=intercept_layers,
        modify_fn=modify_fn,
        to_jax=to_jax,
    ) as ctx:
        output = functional_call(model, params, (input_ids,))
        intermediates = dict(ctx.intermediates)

    return output, intermediates


__all__ = [
    "InterceptionContext",
    "ModifyFn",
    "identity_modify",
    "run_with_hooks",
]
