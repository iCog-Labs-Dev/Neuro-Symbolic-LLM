"""Shared fixtures for substrate tests.

Reference "original frozen models" are real HuggingFace PyTorch models built
from small configs. Their weights are converted to JAX parameter PyTrees via
the same code path used for real checkpoints, and the wrapper output is
compared against the untouched torch forward pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch



GPT2_CFG: dict[str, Any] = {
    "n_layer": 12,
    "n_head": 4,
    "n_embd": 32,
    "n_positions": 32,
    "vocab_size": 64,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
}

NEOX_CFG: dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "rotary_pct": 1.0,
    "rope_theta": 10000.0,
    "layer_norm_eps": 1e-5,
    "use_parallel_residual": False,
    "attention_bias": True,
    "hidden_act": "gelu",
}

NUM_TOKENS = 64
BATCH = 2
SEQ = 9


def _torch_model(family: str, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if family == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        return GPT2LMHeadModel(GPT2Config(**GPT2_CFG))
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    return GPTNeoXForCausalLM(GPTNeoXConfig(**NEOX_CFG))


def _config(family: str):
    if family == "gpt2":
        from transformers import GPT2Config

        return GPT2Config(**GPT2_CFG)
    from transformers import GPTNeoXConfig

    return GPTNeoXConfig(**NEOX_CFG)





def torch_logits(model, ids):
    with torch.no_grad():
        return model(ids).logits.numpy()


def make_input_ids():
    return torch.randint(0, NUM_TOKENS, (BATCH, SEQ))


@pytest.fixture(scope="session")
def gpt2_reference():
    model = _torch_model("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="session")
def pythia_reference():
    model = _torch_model("neox")
    model.eval()
    return model
