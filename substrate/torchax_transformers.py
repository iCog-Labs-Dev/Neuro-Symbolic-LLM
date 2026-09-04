from __future__ import annotations

from typing import Any

import torch
from torch.func import functional_call

from substrate.torchax_backend import enable_torchax, to_torchax_device

_SUPPORTED_MODEL_TYPES = {"gpt2", "gpt_neox"}


def load_torchax_gpt2(
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


def functional_gpt2(
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> Any:
    """ This is a thin wrapper around `torch.func.functional_call`.
    """
    return functional_call(model, params, (input_ids,))


def check_numerical_fidelity(
    model_id: str,
    revision: str | None = None,
    seq_len: int = 8,
    batch_size: int = 2,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare the TorchAX-dispatched forward pass against a plain,
    non-TorchAX PyTorch forward pass on the same loaded weights and the
    same random input.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    enable_torchax()

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model_plain = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model_plain.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        ref_logits = model_plain(input_ids=ids).logits

  
    model_jax = to_torchax_device(model_plain)
    params = dict(model_jax.named_parameters())
    for p in params.values():
        p.requires_grad_(False)

    out = functional_gpt2(model_jax, params, ids.to("jax"))
    jax_logits = out.logits.to("cpu")

    diff = (ref_logits - jax_logits).abs()
    return {
        "shapes_match": ref_logits.shape == jax_logits.shape,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(torch.allclose(ref_logits, jax_logits, atol=atol)),
        "atol": atol,
    }


def load_tokenizer(model_id: str) -> Any:
    """Load the tokenizer for a given HF checkpoint.
    """
    from transformers import AutoTokenizer  # local import: heavy dep

    return AutoTokenizer.from_pretrained(model_id)