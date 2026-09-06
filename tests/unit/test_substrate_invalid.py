"""Invalid-input and robustness tests for the torchax substrate."""

from __future__ import annotations

import pytest
import torch

from substrate.architecture import detect_architecture
from substrate.substrate import Substrate


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"n_layer": 1, "n_embd": 32})


def _dummy_gpt2_params():
    return {
        "transformer.wte.weight": torch.zeros((64, 32)),
        "transformer.wpe.weight": torch.zeros((32, 32)),
        "transformer.ln_f.weight": torch.ones((32,)),
        "transformer.ln_f.bias": torch.zeros((32,)),
        "transformer.h.0.ln_1.weight": torch.ones((32,)),
        "transformer.h.0.ln_1.bias": torch.zeros((32,)),
        "transformer.h.0.ln_2.weight": torch.ones((32,)),
        "transformer.h.0.ln_2.bias": torch.zeros((32,)),
        "transformer.h.0.attn.c_attn.weight": torch.zeros((32, 96)),
        "transformer.h.0.attn.c_attn.bias": torch.zeros((96,)),
        "transformer.h.0.attn.c_proj.weight": torch.zeros((32, 32)),
        "transformer.h.0.attn.c_proj.bias": torch.zeros((32,)),
        "transformer.h.0.mlp.c_fc.weight": torch.zeros((32, 128)),
        "transformer.h.0.mlp.c_fc.bias": torch.zeros((128,)),
        "transformer.h.0.mlp.c_proj.weight": torch.zeros((128, 32)),
        "transformer.h.0.mlp.c_proj.bias": torch.zeros((32,)),
        "lm_head.weight": torch.zeros((64, 32)),
    }


class TestUnsupportedArchitecture:
    def test_unknown_top_level_keys(self):
        params = {"bert.embeddings.weight": torch.zeros((1, 1))}
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            Substrate(DummyModel(), params)

    def test_detect_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            detect_architecture({"nonsense": torch.zeros((1,))})
