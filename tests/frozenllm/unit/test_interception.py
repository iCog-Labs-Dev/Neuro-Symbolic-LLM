"""Unit tests for substrate.interception module using real models from conftest.

Tests hidden-state interception, hook registration, pristine state caching,
in-flight state modification, downstream propagation, and hook lifecycles
using real HuggingFace GPT-2 and Pythia/GPT-NeoX models.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from frozenllm.substrate.architecture import detect_architecture
from frozenllm.substrate.interception import (
    InterceptionContext,
    run_with_hooks,
)


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


class TestInterceptionContextLifecycle:
    def test_registers_and_cleans_up_hooks_gpt2(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        assert len(model.transformer.h[1]._forward_hooks) == 0
        assert len(model.transformer.h[3]._forward_hooks) == 0

        with InterceptionContext(model, intercept_layers=[1, 3]) as ctx:
            assert len(model.transformer.h[0]._forward_hooks) == 0
            assert len(model.transformer.h[1]._forward_hooks) == 1
            assert len(model.transformer.h[2]._forward_hooks) == 0
            assert len(model.transformer.h[3]._forward_hooks) == 1
            assert len(ctx._handles) == 2

        # Upon exit, all hooks are cleanly removed
        assert len(model.transformer.h[1]._forward_hooks) == 0
        assert len(model.transformer.h[3]._forward_hooks) == 0
        assert len(ctx._handles) == 0

    def test_registers_and_cleans_up_hooks_pythia(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        model = real_pythia_model
        assert len(model.gpt_neox.layers[0]._forward_hooks) == 0
        assert len(model.gpt_neox.layers[2]._forward_hooks) == 0

        with InterceptionContext(model, intercept_layers=[0, 2]) as ctx:
            assert len(model.gpt_neox.layers[0]._forward_hooks) == 1
            assert len(model.gpt_neox.layers[1]._forward_hooks) == 0
            assert len(model.gpt_neox.layers[2]._forward_hooks) == 1
            assert len(ctx._handles) == 2

        assert len(model.gpt_neox.layers[0]._forward_hooks) == 0
        assert len(model.gpt_neox.layers[2]._forward_hooks) == 0
        assert len(ctx._handles) == 0

    def test_cleans_up_hooks_on_exception(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        with (
            pytest.raises(RuntimeError, match="Simulated forward failure"),
            InterceptionContext(model, intercept_layers=[1]),
        ):
            assert len(model.transformer.h[1]._forward_hooks) == 1
            raise RuntimeError("Simulated forward failure")

        assert len(model.transformer.h[1]._forward_hooks) == 0

    def test_empty_interception_layers(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        with InterceptionContext(model, intercept_layers=[]) as ctx:
            assert len(ctx._handles) == 0
            assert ctx.intercept_layers == ()

    def test_invalid_interception_layers_raises(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        with pytest.raises(ValueError, match="Invalid interception layer"):
            InterceptionContext(model, intercept_layers=[99])

    def test_explicit_arch_passed(self, real_gpt2_model: GPT2LMHeadModel):
        model = real_gpt2_model
        arch = detect_architecture(model.config)
        with InterceptionContext(model, arch=arch, intercept_layers=[1]) as ctx:
            assert ctx.arch is arch
            assert len(ctx._handles) == 1
        assert len(model.transformer.h[1]._forward_hooks) == 0

    def test_no_config_and_no_arch_raises(self):
        dummy = torch.nn.Linear(10, 10)
        with pytest.raises(ValueError, match="arch must be provided"):
            InterceptionContext(dummy, intercept_layers=[0])


class TestInterceptionForwardExecution:
    def test_gpt2_intercepts_pristine_hidden_states(
        self, real_gpt2_model: GPT2LMHeadModel
    ):
        model = real_gpt2_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))

        with InterceptionContext(model, intercept_layers=[1, 3]) as ctx:
            out = model(input_ids=ids)
            assert out.logits.shape == (2, 8, model.config.vocab_size)
            assert sorted(ctx.intermediates.keys()) == [1, 3]

            # Hidden states have shape [batch, seq, hidden_size]
            for layer_idx in [1, 3]:
                h = ctx.intermediates[layer_idx]
                assert isinstance(h, torch.Tensor)
                assert h.shape == (2, 8, model.config.n_embd)

    def test_pythia_intercepts_pristine_hidden_states(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        model = real_pythia_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))

        with InterceptionContext(model, intercept_layers=[0, 2]) as ctx:
            out = model(input_ids=ids)
            assert out.logits.shape == (2, 8, model.config.vocab_size)
            assert sorted(ctx.intermediates.keys()) == [0, 2]

            for layer_idx in [0, 2]:
                h = ctx.intermediates[layer_idx]
                assert isinstance(h, torch.Tensor)
                assert h.shape == (2, 8, model.config.hidden_size)

    def test_pristine_caching_unaffected_by_modify_hook(
        self, real_gpt2_model: GPT2LMHeadModel
    ):
        model = real_gpt2_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))

        # 1. Run pristine baseline to record unmodified layer 1 output
        with (
            torch.no_grad(),
            InterceptionContext(model, intercept_layers=[1]) as ctx_base,
        ):
            base_out = model(input_ids=ids)
            base_h1 = ctx_base.intermediates[1].clone()

        # 2. Run with modification: perturb layer 1 hidden state with dimension-varying offset
        def perturb(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
            return h + 0.5 * torch.arange(h.shape[-1], device=h.device, dtype=h.dtype)

        with (
            torch.no_grad(),
            InterceptionContext(
                model, intercept_layers=[1], modify_fn=perturb
            ) as ctx_mod,
        ):
            mod_out = model(input_ids=ids)
            mod_cached_h1 = ctx_mod.intermediates[1]

        # Intermediates MUST capture pristine pre-modification state
        assert torch.allclose(base_h1, mod_cached_h1, atol=1e-6)

        # Output logits MUST change due to downstream perturbation
        assert not torch.allclose(base_out.logits, mod_out.logits, atol=1e-2)


class TestRunWithHooks:
    def test_run_with_hooks_matches_regular_forward(
        self, real_gpt2_model: GPT2LMHeadModel
    ):
        model = real_gpt2_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        params = dict(model.named_parameters())

        with torch.no_grad():
            ref_out = model(input_ids=ids)

        out, intermediates = run_with_hooks(
            model=model,
            params=params,
            input_ids=ids,
            intercept_layers=[1, 2],
        )

        assert torch.allclose(ref_out.logits, out.logits, atol=1e-5)
        assert sorted(intermediates.keys()) == [1, 2]
        assert intermediates[1].shape == (2, 8, model.config.n_embd)

    def test_run_with_hooks_pythia_modification(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        model = real_pythia_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        params = dict(model.named_parameters())

        # Baseline run
        base_out, base_inter = run_with_hooks(
            model=model,
            params=params,
            input_ids=ids,
            intercept_layers=[1],
        )

        # Perturbed run
        def zero_hook(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
            return h * 0.0

        mod_out, mod_inter = run_with_hooks(
            model=model,
            params=params,
            input_ids=ids,
            intercept_layers=[1],
            modify_fn=zero_hook,
        )

        # Pristine hidden states match
        assert torch.allclose(base_inter[1], mod_inter[1], atol=1e-6)

        # Logits differ due to modification
        assert not torch.allclose(base_out.logits, mod_out.logits, atol=1e-2)


class TestMultipleInterceptions:
    def test_multiple_interceptions_with_cascade_modifications(
        self, real_gpt2_model: GPT2LMHeadModel
    ):
        """Verify downstream propagation across multiple interception points."""
        model = real_gpt2_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))

        # 1. Baseline unmodified run
        with (
            torch.no_grad(),
            InterceptionContext(model, intercept_layers=[0, 2]) as ctx_base,
        ):
            base_out = model(input_ids=ids)
            base_h0 = ctx_base.intermediates[0].clone()
            base_h2 = ctx_base.intermediates[2].clone()

        # 2. Modify layer 0 only with dimension-varying perturbation
        def mod_layer_0_only(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if layer_idx == 0:
                return h + 0.5 * torch.arange(
                    h.shape[-1], device=h.device, dtype=h.dtype
                )
            return h

        with (
            torch.no_grad(),
            InterceptionContext(
                model, intercept_layers=[0, 2], modify_fn=mod_layer_0_only
            ) as ctx_l0,
        ):
            l0_out = model(input_ids=ids)
            l0_h0 = ctx_l0.intermediates[0]
            l0_h2 = ctx_l0.intermediates[2]

        # Pristine h0 MUST match baseline
        assert torch.allclose(base_h0, l0_h0, atol=1e-6)
        # But h2 in l0 run MUST differ from base_h2 because layer 0's modification cascaded through layer 1
        assert not torch.allclose(base_h2, l0_h2, atol=1e-3)

        # 3. Modify both layer 0 and layer 2 with distinct operations
        def mod_both(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if layer_idx == 0:
                return h + 0.5 * torch.arange(
                    h.shape[-1], device=h.device, dtype=h.dtype
                )
            elif layer_idx == 2:
                return h * 0.0
            return h

        with (
            torch.no_grad(),
            InterceptionContext(
                model, intercept_layers=[0, 2], modify_fn=mod_both
            ) as ctx_both,
        ):
            both_out = model(input_ids=ids)
            both_h0 = ctx_both.intermediates[0]
            both_h2 = ctx_both.intermediates[2]

        # In the combined run:
        # both_h0 is pristine layer 0 (matches base_h0)
        assert torch.allclose(base_h0, both_h0, atol=1e-6)
        # both_h2 is pristine layer 2 (matches l0_h2 before layer 2 was scaled by 0.5)
        assert torch.allclose(l0_h2, both_h2, atol=1e-6)
        # Logits of all three runs differ
        assert not torch.allclose(base_out.logits, l0_out.logits, atol=1e-2)
        assert not torch.allclose(l0_out.logits, both_out.logits, atol=1e-2)

    def test_out_of_order_intercept_layers(self, real_pythia_model: GPTNeoXForCausalLM):
        """Verify intercept_layers passed unsorted are handled in forward order."""
        model = real_pythia_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))

        with InterceptionContext(model, intercept_layers=[3, 0, 2]) as ctx:
            assert ctx.intercept_layers == (0, 2, 3)
            _out = model(input_ids=ids)
            assert sorted(ctx.intermediates.keys()) == [0, 2, 3]

    def test_all_layers_interception(self, real_gpt2_model: GPT2LMHeadModel):
        """Verify intercepting every layer captures all transformer blocks."""
        model = real_gpt2_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        all_layers = list(range(model.config.n_layer))

        with InterceptionContext(model, intercept_layers=all_layers) as ctx:
            _out = model(input_ids=ids)
            assert sorted(ctx.intermediates.keys()) == all_layers
            assert len(ctx.intermediates) == model.config.n_layer
            for idx in all_layers:
                assert ctx.intermediates[idx].shape == (2, 8, model.config.n_embd)

    def test_run_with_hooks_multiple_interceptions_pythia(
        self, real_pythia_model: GPTNeoXForCausalLM
    ):
        """Verify run_with_hooks across multiple interception points."""
        model = real_pythia_model
        torch.manual_seed(0)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        params = dict(model.named_parameters())

        called_layers = []

        def tracking_hook(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
            called_layers.append(layer_idx)
            return h + float(layer_idx + 1)

        out, intermediates = run_with_hooks(
            model=model,
            params=params,
            input_ids=ids,
            intercept_layers=[0, 1, 2, 3],
            modify_fn=tracking_hook,
        )

        assert called_layers == [0, 1, 2, 3]
        assert sorted(intermediates.keys()) == [0, 1, 2, 3]
        for idx in range(4):
            assert intermediates[idx].shape == (2, 8, model.config.hidden_size)
