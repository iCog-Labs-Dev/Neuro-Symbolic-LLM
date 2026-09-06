"""Device management, dispatch setup, and value-conversion utilities for
running PyTorch code on TorchAX's JAX-backed device.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import torch
import torchax
import torchax.interop as interop

T = TypeVar("T")

_torchax_enabled = False


def enable_torchax() -> None:
    """Enable TorchAX's global op interception.
    """
    global _torchax_enabled
    if not _torchax_enabled:
        torchax.enable_globally()
        _torchax_enabled = True


def is_torchax_enabled() -> bool:
    """Whether `enable_torchax()` has been called in this process yet."""
    return _torchax_enabled


def to_jax_array(value: Any) -> jax.Array:
    """Convert a `torchax.tensor.Tensor` (or plain torch.Tensor moved onto
    the 'jax' device) into a raw `jax.Array`.

    This is a value conversion, not an autograd boundary. the result is
    a jax.Array with no torch-autograd history attached.
    """
    enable_torchax()
    return interop.jax_view(value)


def from_jax_array(value: jax.Array) -> torch.Tensor:
    """Convert a raw `jax.Array` into a `torchax.tensor.Tensor`.
    """
    enable_torchax()
    return interop.torch_view(value)


def to_torchax_device(obj: T) -> T:
    """Move a `torch.nn.Module` or `torch.Tensor` onto TorchAX's 'jax'
    device, enabling global dispatch first if it hasn't been already.
    """
    enable_torchax()
    return obj.to("jax")  # type: ignore[no-any-return, attr-defined]


def is_on_torchax_device(tensor: torch.Tensor) -> bool:
    """Whether `tensor` is already on TorchAX's 'jax' device."""
    return str(tensor.device).startswith("jax")


def call_jax_differentiable(
    jax_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a genuine raw-JAX function so it can be called from torch-side
    code with correct autograd gradients.
    """
    enable_torchax()

    def torch_shell(*args: Any, **kwargs: Any) -> Any:
        return interop.call_jax(jax_fn, *args, **kwargs)

    return interop.j2t_autograd(torch_shell)  # type: ignore[no-any-return]