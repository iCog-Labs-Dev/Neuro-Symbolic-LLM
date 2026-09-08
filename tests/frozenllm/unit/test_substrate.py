"""Unit tests for substrate.substrate (FrozenSubstrate with TorchAX execution).

Verifies:
1. FrozenSubstrate initialization with live PyTorch models (GPT-2 and Pythia).
2. Architecture auto-detection and layer discovery.
3. Strict parameter immutability and theta_0 freezing invariant.
4. Forward pass execution with pure JAX logits and pristine intermediates.
5. In-flight hidden state interception and custom steering hooks.
6. Clean hook lifecycle (all hooks detached after execution).
7. Causal LM loss computation with label shifting and padding masking.
8. Memory status and input validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from frozenllm.substrate.architecture import detect_architecture
from frozenllm.substrate.substrate import ForwardResult, FrozenSubstrate
from frozenllm.substrate.torchax_backend import enable_torchax

enable_torchax()


def _tiny_model() -> GPT2LMHeadModel:
    cfg = GPT2Config(n_layer=4, n_head=2, n_embd=32, vocab_size=100, n_positions=16)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _tiny_neox_model() -> GPTNeoXForCausalLM:
    cfg = GPTNeoXConfig(
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
    model = GPTNeoXForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture
def real_gpt2_model() -> GPT2LMHeadModel:
    return _tiny_model()


@pytest.fixture
def real_pythia_model() -> GPTNeoXForCausalLM:
    return _tiny_neox_model()


class TestFrozenSubstrateInit:
    def test_initialization_with_gpt2_model(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model, intercept_layers=[1, 3])

        assert sub.architecture.model_family == "gpt2"
        assert sub.architecture.num_layers == model.config.n_layer
        assert sub.architecture.hidden_size == model.config.n_embd
        assert sub.intercept_layers == (1, 3)

    def test_initialization_with_pythia_model(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        model = real_pythia_model
        sub = FrozenSubstrate(model, intercept_layers=[0, 2])

        assert sub.architecture.model_family == "neox"
        assert sub.architecture.num_layers == model.config.num_hidden_layers
        assert sub.architecture.hidden_size == model.config.hidden_size
        assert sub.intercept_layers == (0, 2)

    def test_parameters_are_strictly_frozen(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model)

        # Invariant: every parameter leaf has requires_grad=False
        for name, param in sub.params.items():
            assert not param.requires_grad, f"Param {name} was not frozen!"

        assert sub.params_unchanged()
        report = sub.verify_frozen()
        assert report["params_unchanged"] is True
        assert report["architecture"]["num_layers"] == model.config.n_layer

    def test_invalid_interception_layers_rejected(
        self, real_gpt2_model: GPT2LMHeadModel
    ):
        model = real_gpt2_model
        n_layers = model.config.n_layer
        with pytest.raises(ValueError, match="non-negative"):
            FrozenSubstrate(model, intercept_layers=[-1])

        with pytest.raises(ValueError, match=f"only {n_layers} transformer layers"):
            FrozenSubstrate(model, intercept_layers=[n_layers])

    def test_detect_architecture_raises_on_invalid_keys(self):
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            detect_architecture({"nonsense": torch.zeros((1,))})


class TestFrozenSubstrateForward:
    def test_forward_produces_jax_logits_gpt2(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model, intercept_layers=[1])

        batch, seq_len = 2, 8
        input_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)

        result = sub(input_ids)

        assert isinstance(result, ForwardResult)
        assert isinstance(result.logits, jax.Array)
        assert result.logits.shape == (batch, seq_len, model.config.vocab_size)

        # Intermediates check
        assert result.layer_indices() == (1,)
        h1 = result.hidden_state(1)
        assert isinstance(h1, jax.Array)
        assert h1.shape == (batch, seq_len, model.config.n_embd)
        assert result.hidden_shapes() == {1: (batch, seq_len, model.config.n_embd)}

    def test_forward_produces_jax_logits_pythia(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        model = real_pythia_model
        sub = FrozenSubstrate(model, intercept_layers=[2])

        batch, seq_len = 2, 8
        input_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)

        result = sub(input_ids)

        assert isinstance(result, ForwardResult)
        assert isinstance(result.logits, jax.Array)
        assert result.logits.shape == (batch, seq_len, model.config.vocab_size)

        # Intermediates check
        assert result.layer_indices() == (2,)
        h2 = result.hidden_state(2)
        assert isinstance(h2, jax.Array)
        assert h2.shape == (batch, seq_len, model.config.hidden_size)
        assert result.hidden_shapes() == {2: (batch, seq_len, model.config.hidden_size)}

    def test_input_shape_validation(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model)

        with pytest.raises(ValueError, match="2D"):
            sub(jnp.zeros((4,), dtype=jnp.int32))

        with pytest.raises(ValueError, match="at least one"):
            sub(jnp.zeros((2, 0), dtype=jnp.int32))

    def test_tokenize_helper(self, real_gpt2_model: GPT2LMHeadModel):
        # Mock tokenizer returning PyTorch tensor
        class MockTokenizer:
            def __call__(self, text, return_tensors="pt", **kw):
                return {"input_ids": torch.tensor([[10, 20, 30]])}

        mock_tok = MockTokenizer()
        model = real_gpt2_model
        sub = FrozenSubstrate(model, tokenizer=mock_tok)

        assert sub.tokenizer is mock_tok
        ids_jax = sub.tokenize("test prompt", return_tensors="jax")
        assert isinstance(ids_jax, jax.Array)
        assert ids_jax.shape == (1, 3)

        ids_pt = sub.tokenize("test prompt", return_tensors="pt")
        assert isinstance(ids_pt, torch.Tensor)
        assert ids_pt.shape == (1, 3)

    def test_forward_execution_deterministic(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model, intercept_layers=[1])
        batch, seq_len = 2, 8
        input_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)

        res1 = sub(input_ids)
        res2 = sub(input_ids)

        assert jnp.all(res1.logits == res2.logits)
        assert sub.params_unchanged() is True


class TestInterceptionAndSteering:
    def test_steering_hook_modifies_output_gpt2(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model, intercept_layers=[2])

        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        # 1. Baseline unsteered run
        baseline = sub(input_ids)

        # 2. Steered run: add dimension-varying perturbation at layer 2
        def steer(h: jax.Array, layer_idx: int) -> jax.Array:
            delta = jnp.linspace(1.0, 5.0, h.shape[-1])
            return h + delta

        steered = sub.run_with_interception(
            input_ids=input_ids,
            modify_fn=steer,
            intercept_layers=[2],
        )

        # Both results return valid JAX arrays
        assert isinstance(steered.logits, jax.Array)

        # Logits must diverge due to downstream propagation
        diff = float(jnp.max(jnp.abs(steered.logits - baseline.logits)))
        assert diff > 0.0, "Steering did not affect downstream logits!"

        # Pristine intermediate at layer 2 should match between both runs
        pristine_diff = float(
            jnp.max(jnp.abs(steered.hidden_state(2) - baseline.hidden_state(2)))
        )
        assert pristine_diff < 1e-5, "Intermediates did not capture the pristine state!"

    def test_steering_hook_modifies_output_pythia(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        model = real_pythia_model
        sub = FrozenSubstrate(model, intercept_layers=[1])

        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        # 1. Baseline unsteered run
        baseline = sub(input_ids)

        # 2. Steered run: add dimension-varying perturbation at layer 1
        def steer(h: jax.Array, layer_idx: int) -> jax.Array:
            delta = jnp.linspace(1.0, 5.0, h.shape[-1])
            return h + delta

        steered = sub.run_with_interception(
            input_ids=input_ids,
            modify_fn=steer,
            intercept_layers=[1],
        )

        assert isinstance(steered.logits, jax.Array)
        diff = float(jnp.max(jnp.abs(steered.logits - baseline.logits)))
        assert diff > 0.0, "Steering did not affect downstream logits!"

        pristine_diff = float(
            jnp.max(jnp.abs(steered.hidden_state(1) - baseline.hidden_state(1)))
        )
        assert pristine_diff < 1e-5, "Intermediates did not capture the pristine state!"

    def test_clean_hook_lifecycle_gpt2(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model, intercept_layers=[1, 2])

        input_ids = jnp.zeros((1, 4), dtype=jnp.int32)
        _ = sub(input_ids)

        # Hooks must be cleaned up after forward pass
        for block in model.transformer.h:
            assert len(block._forward_hooks) == 0

    def test_clean_hook_lifecycle_pythia(self, real_pythia_model: GPTNeoXForCausalLM):
        model = real_pythia_model
        sub = FrozenSubstrate(model, intercept_layers=[0, 1])

        input_ids = jnp.zeros((1, 4), dtype=jnp.int32)
        _ = sub(input_ids)

        # Hooks must be cleaned up after forward pass
        for block in model.gpt_neox.layers:
            assert len(block._forward_hooks) == 0


class TestCausalLossComputation:
    def test_compute_loss_causal_shift(self):
        vocab_size = 64
        batch, seq_len = 2, 6
        # Deterministic logits
        logits = jnp.zeros((batch, seq_len, vocab_size))
        labels = jnp.zeros((batch, seq_len), dtype=jnp.int32)

        loss = FrozenSubstrate.compute_loss(logits, labels)

        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0
        # Uniform distribution over vocab_size tokens: -ln(1/V) = ln(V)
        expected = float(jnp.log(vocab_size))
        assert abs(float(loss) - expected) < 1e-4

    def test_compute_loss_ignores_padding(self):
        batch, seq_len, vocab_size = 1, 4, 10
        logits = jnp.zeros((batch, seq_len, vocab_size))
        # Mask out all target positions except one
        labels = jnp.array([[-100, -100, 0, -100]], dtype=jnp.int32)

        loss = FrozenSubstrate.compute_loss(logits, labels)
        expected = float(jnp.log(vocab_size))
        assert abs(float(loss) - expected) < 1e-4


class TestMemoryStatus:
    def test_memory_status_returns_valid_object(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        sub = FrozenSubstrate(model)

        status = sub.memory_status()
        assert status is not None
        assert hasattr(status, "allocated_bytes")
        assert hasattr(status, "bytes_in_use")
        assert hasattr(status, "available")
