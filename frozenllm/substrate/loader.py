"""Model and tokenizer loading for frozen LLM substrates via TorchAX."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from frozenllm.substrate.torchax_backend import to_torchax_device

_SUPPORTED_MODEL_TYPES = {"gpt2", "gpt_neox"}


@contextmanager
def _native_torch_loading() -> Iterator[None]:
    """Load checkpoints on CPU, restoring any active default TorchAX modes."""
    import torchax
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    env = torchax.default_env()
    functions = _get_current_function_mode_stack()
    dispatches = _get_current_dispatch_mode_stack()
    function_depth = sum(mode is env._function_mode for mode in functions)
    dispatch_depth = sum(mode is env._dispatch_mode for mode in dispatches)
    if function_depth != dispatch_depth or function_depth > 1:
        raise RuntimeError("Native loading requires at most one default TorchAX scope")
    active = function_depth == 1
    if active and (
        functions[-1] is not env._function_mode
        or dispatches[-1] is not env._dispatch_mode
    ):
        raise RuntimeError("Native loading cannot suspend an overlaid TorchAX scope")
    if active:
        env.disable_torch_modes()
    try:
        with torch.device("cpu"):
            yield
    finally:
        if active:
            env.enable_torch_modes()


def load_torchax_model(
    model_id: str,
    revision: str | None = None,
    dtype: torch.dtype | str | None = None,
    attn_implementation: str | None = "eager",
    torch_dtype: torch.dtype | str | None = None,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    with _native_torch_loading():
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        if config.model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model architecture {config.model_type!r} for model "
                f"{model_id!r}. Supported: {sorted(_SUPPORTED_MODEL_TYPES)}."
            )

        effective_dtype = dtype or torch_dtype or torch.float32
        load_kwargs: dict[str, Any] = {"dtype": effective_dtype}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, **load_kwargs
            )
        except TypeError:
            load_kwargs.pop("dtype", None)
            load_kwargs["torch_dtype"] = effective_dtype
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=revision, **load_kwargs
                )
            except TypeError:
                load_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=revision, **load_kwargs
                )

        model.eval()
    model = to_torchax_device(model)

    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(False)

    return model, params


def load_tokenizer(model_id: str) -> Any:
    with _native_torch_loading():
        return AutoTokenizer.from_pretrained(model_id)
