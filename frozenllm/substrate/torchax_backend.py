"""Device management, dispatch setup, and value-conversion utilities for
running PyTorch code on TorchAX's JAX-backed device.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = getattr(torch, "float8_e4m3fn", torch.uint8)

torchax: Any = None
interop: Any = None

T = TypeVar("T")

_torchax_enabled = False


def enable_torchax() -> None:
    global _torchax_enabled, torchax, interop
    if not _torchax_enabled:
        import torchax as _torchax
        import torchax.interop as _interop

        torchax = _torchax
        interop = _interop
        torchax.enable_globally()
        _torchax_enabled = True


def is_torchax_enabled() -> bool:
    return _torchax_enabled


def to_jax_array(value: Any) -> jax.Array:
    enable_torchax()
    return interop.jax_view(value)


def from_jax_array(value: jax.Array) -> torch.Tensor:
    enable_torchax()
    return interop.torch_view(value)


def to_torchax_device(obj: T) -> T:
    enable_torchax()
    return obj.to("jax")  # type: ignore[no-any-return, attr-defined]


def is_on_torchax_device(tensor: torch.Tensor) -> bool:
    return str(tensor.device).startswith("jax")


def call_jax_differentiable(
    jax_fn: Callable[..., Any],
) -> Callable[..., Any]:
    enable_torchax()

    def torch_shell(*args: Any, **kwargs: Any) -> Any:
        return interop.call_jax(jax_fn, *args, **kwargs)

    return interop.j2t_autograd(torch_shell)  # type: ignore[no-any-return]
