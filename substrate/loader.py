from __future__ import annotations

from typing import Any

import torch


from substrate.torchax_backend import enable_torchax, to_torchax_device

_SUPPORTED_MODEL_TYPES = {"gpt2", "gpt_neox"}


def load_torchax_model(
    model_id: str,
    revision: str | None = None,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Load a real HF GPT-2 checkpoint, moved onto TorchAX's JAX-backed
    device, with its parameters frozen and exposed as an explicit dict.
    """
    enable_torchax()

    from transformers import AutoConfig, AutoModelForCausalLM  # local import: heavy dep


    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if config.model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model architecture {config.model_type!r} for model "
            f"{model_id!r}. Supported: {sorted(_SUPPORTED_MODEL_TYPES)}."
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.eval()
    model = to_torchax_device(model)

    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(False)

    return model, params

def load_tokenizer(model_id: str) -> Any:
    """Load the tokenizer for a given HF checkpoint.
    """
    from transformers import AutoTokenizer  # local import: heavy dep

    return AutoTokenizer.from_pretrained(model_id)