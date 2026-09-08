"""Unit tests for substrate.architecture module.

Tests architecture detection from HuggingFace config objects and parameter trees,
block accessors, module accessors, and interception layer validation using real
HuggingFace GPT-2 and Pythia/GPT-NeoX models and configurations.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PretrainedConfig,
)

from frozenllm.substrate.architecture import (
    detect_architecture,
    detect_architecture_from_config,
    discover_layers,
    discover_layers_from_config,
    get_block_accessor,
    get_embedding_module,
    get_head_modules,
    get_position_embedding_module,
    validate_interception_layers,
)


def _tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(n_layer=4, n_head=2, n_embd=32, vocab_size=100, n_positions=16)


def _tiny_neox_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        rotary_pct=1.0,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        use_parallel_residual=False,
        attention_bias=True,
        hidden_act="gelu",
    )


@pytest.fixture
def real_gpt2_config() -> GPT2Config:
    return _tiny_gpt2_config()


@pytest.fixture
def real_pythia_config() -> GPTNeoXConfig:
    return _tiny_neox_config()


@pytest.fixture
def real_gpt2_model(real_gpt2_config: GPT2Config) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(real_gpt2_config)
    model.eval()
    return model


@pytest.fixture
def real_pythia_model(real_pythia_config: GPTNeoXConfig) -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM(real_pythia_config)
    model.eval()
    return model


class TestDetectArchitectureFromConfig:
    def test_gpt2_config_object(self):
        cfg = GPT2Config(
            n_layer=12,
            n_embd=768,
            n_head=12,
            vocab_size=50257,
            n_positions=1024,
            layer_norm_epsilon=1e-5,
        )
        arch = detect_architecture_from_config(cfg)
        assert arch.model_family == "gpt2"
        assert arch.num_layers == 12
        assert arch.hidden_size == 768
        assert arch.num_heads == 12
        assert arch.head_dim == 64
        assert arch.vocab_size == 50257
        assert arch.max_position_embeddings == 1024
        assert arch.layer_norm_eps == 1e-5
        assert not arch.use_parallel_residual

    def test_gpt2_config_dict(self):
        cfg_dict = {
            "model_type": "gpt2",
            "n_layer": 6,
            "n_embd": 512,
            "n_head": 8,
            "vocab_size": 32000,
            "n_positions": 512,
        }
        arch = detect_architecture_from_config(cfg_dict)
        assert arch.model_family == "gpt2"
        assert arch.num_layers == 6
        assert arch.hidden_size == 512
        assert arch.num_heads == 8
        assert arch.head_dim == 64
        assert arch.vocab_size == 32000

    def test_neox_config_object(self):
        cfg = GPTNeoXConfig(
            num_hidden_layers=16,
            hidden_size=1024,
            num_attention_heads=16,
            vocab_size=50304,
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            rope_theta=10000.0,
            rotary_pct=0.25,
            use_parallel_residual=True,
        )
        arch = detect_architecture_from_config(cfg)
        assert arch.model_family == "neox"
        assert arch.num_layers == 16
        assert arch.hidden_size == 1024
        assert arch.num_heads == 16
        assert arch.head_dim == 64
        assert arch.vocab_size == 50304
        assert arch.max_position_embeddings == 2048
        assert arch.rotary_pct == 0.25
        assert arch.use_parallel_residual is True

    def test_neox_config_dict(self):
        cfg_dict = {
            "model_type": "gpt_neox",
            "num_hidden_layers": 8,
            "hidden_size": 512,
            "num_attention_heads": 8,
        }
        arch = detect_architecture_from_config(cfg_dict)
        assert arch.model_family == "neox"
        assert arch.num_layers == 8
        assert arch.hidden_size == 512
        assert arch.num_heads == 8
        assert arch.head_dim == 64
        assert arch.use_parallel_residual is True

    def test_pythia_detection_via_architectures(self):
        cfg = GPTNeoXConfig(
            architectures=["GPTNeoXForCausalLM"],
            num_hidden_layers=6,
            hidden_size=256,
            num_attention_heads=4,
        )
        arch = detect_architecture_from_config(cfg)
        assert arch.model_family == "neox"
        assert arch.num_layers == 6
        assert arch.hidden_size == 256
        assert arch.head_dim == 64

    def test_conftest_gpt2_config(self, real_gpt2_config: GPT2Config):
        arch = detect_architecture_from_config(real_gpt2_config)
        assert arch.model_family == "gpt2"
        assert arch.num_layers == real_gpt2_config.n_layer
        assert arch.hidden_size == real_gpt2_config.n_embd
        assert arch.num_heads == real_gpt2_config.n_head

    def test_conftest_pythia_config(self, real_pythia_config: GPTNeoXConfig):
        arch = detect_architecture_from_config(real_pythia_config)
        assert arch.model_family == "neox"
        assert arch.num_layers == real_pythia_config.num_hidden_layers
        assert arch.hidden_size == real_pythia_config.hidden_size
        assert arch.num_heads == real_pythia_config.num_attention_heads

    def test_unsupported_model_type_raises(self):
        class LlamaLikeConfig(PretrainedConfig):
            model_type = "llama"

        cfg = LlamaLikeConfig(num_hidden_layers=32)
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            detect_architecture_from_config(cfg)

    def test_indivisible_heads_raises(self):
        cfg = GPT2Config(
            n_layer=4,
            n_embd=100,
            n_head=3,  # 100 is not divisible by 3
        )
        with pytest.raises(ValueError, match="not divisible by num_heads"):
            detect_architecture_from_config(cfg)

    def test_invalid_num_layers_raises(self):
        cfg = GPT2Config(n_layer=0, n_embd=64, n_head=1)
        with pytest.raises(ValueError, match="Invalid num_layers"):
            detect_architecture_from_config(cfg)


class TestDiscoverLayersFromConfig:
    def test_discover_layers(self):
        cfg = GPT2Config(n_layer=24, n_embd=1024, n_head=16)
        assert discover_layers_from_config(cfg) == 24
        assert discover_layers(cfg) == 24

    def test_discover_layers_from_conftest_gpt2(self, real_gpt2_config: GPT2Config):
        assert discover_layers_from_config(real_gpt2_config) == real_gpt2_config.n_layer
        assert discover_layers(real_gpt2_config) == real_gpt2_config.n_layer

    def test_discover_layers_from_conftest_pythia(
        self, real_pythia_config: GPTNeoXConfig
    ):
        assert (
            discover_layers_from_config(real_pythia_config)
            == real_pythia_config.num_hidden_layers
        )
        assert (
            discover_layers(real_pythia_config) == real_pythia_config.num_hidden_layers
        )

    def test_unified_detect_architecture(self, real_gpt2_config: GPT2Config):
        arch = detect_architecture(real_gpt2_config)
        assert arch.model_family == "gpt2"
        assert arch.num_layers == real_gpt2_config.n_layer


# ── Accessor Tests with Real Models from conftest ────────────────────────────


class TestAccessors:
    def test_gpt2_accessors(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        arch = detect_architecture_from_config(model.config)

        # Block accessor
        accessor = get_block_accessor(model, arch)
        assert accessor(0) is model.transformer.h[0]
        assert accessor(3) is model.transformer.h[3]

        # Embedding accessors
        wte = get_embedding_module(model, arch)
        assert wte is model.transformer.wte
        wpe = get_position_embedding_module(model, arch)
        assert wpe is not None
        assert wpe is model.transformer.wpe

        # Head modules
        head_mods = get_head_modules(model, arch)
        assert len(head_mods) == 2
        assert head_mods[0] is model.transformer.ln_f
        assert head_mods[1] is model.lm_head

    def test_neox_accessors(self, real_pythia_model: GPTNeoXForCausalLM):
        model = real_pythia_model
        arch = detect_architecture_from_config(model.config)

        # Block accessor
        accessor = get_block_accessor(model, arch)
        assert accessor(0) is model.gpt_neox.layers[0]
        assert accessor(2) is model.gpt_neox.layers[2]

        # Embedding accessors
        embed_in = get_embedding_module(model, arch)
        assert embed_in is model.gpt_neox.embed_in
        assert get_position_embedding_module(model, arch) is None

        # Head modules
        head_mods = get_head_modules(model, arch)
        assert len(head_mods) == 2
        assert head_mods[0] is model.gpt_neox.final_layer_norm
        assert head_mods[1] is model.lm_head

    def test_wrapped_model_accessor(self, real_gpt2_model: GPT2LMHeadModel):
        inner = real_gpt2_model

        class WrappedModel(torch.nn.Module):
            def __init__(self, mod: torch.nn.Module):
                super().__init__()
                self.module = mod

        wrapped = WrappedModel(inner)
        arch = detect_architecture_from_config(inner.config)
        accessor = get_block_accessor(wrapped, arch)
        assert accessor(1) is inner.transformer.h[1]


# ── Interception Layer Validation Tests ───────────────────────────────────────


class TestValidateInterceptionLayers:
    def test_valid_layers(self):
        assert validate_interception_layers([3, 1, 0], 4) == (0, 1, 3)
        assert validate_interception_layers(None, 4) == ()
        assert validate_interception_layers([], 4) == ()

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid interception layer"):
            validate_interception_layers([4], 4)

    def test_negative_layer(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_interception_layers([-1], 4)

    def test_duplicate_layer(self):
        with pytest.raises(ValueError, match="Duplicate interception layer"):
            validate_interception_layers([1, 1], 4)