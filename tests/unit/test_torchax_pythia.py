"""Tests for substrate.torchax_gpt2's Pythia/GPT-NeoX support -- run
entirely against a local random-init GPTNeoXConfig, no network needed.
"""

from __future__ import annotations

import torch
import torchax
from torch.func import functional_call
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from substrate.torchax_models import functional_model

torchax.enable_globally()


def _tiny_neox_model() -> GPTNeoXForCausalLM:
    # same scale as test_torchax_gpt2.py's _tiny_model(), same rotary/attention
    # settings as the project's existing NEOX_CFG fixture in conftest.py
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


class TestFunctionalCallCorrectness:
    def test_produces_correct_output_shape(self):
        model = _tiny_neox_model().to("jax")
        params = dict(model.named_parameters())
        ids = torch.randint(0, 100, (2, 6)).to("jax")
        out = functional_model(model, params, ids)
        assert out.logits.shape == (2, 6, 100)


class TestFidelity:
    def test_numerically_matches_plain_pytorch(self):
        model = _tiny_neox_model()
        torch.manual_seed(0)
        ids = torch.randint(0, 100, (2, 6))
        with torch.no_grad():
            ref_logits = model(input_ids=ids).logits

        model = model.to("jax")
        params = dict(model.named_parameters())
        out = functional_model(model, params, ids.to("jax"))
        jax_logits = out.logits.to("cpu")

        diff = (ref_logits - jax_logits).abs()
        print(
            f"\nNeoX/Pythia TorchAX fidelity: max_abs_diff={float(diff.max()):.3e}, "
            f"mean_abs_diff={float(diff.mean()):.3e}"
        )

        assert ref_logits.shape == jax_logits.shape
        assert torch.allclose(ref_logits, jax_logits, atol=1e-4)


class TestFreezing:
    def test_params_do_not_require_grad_after_explicit_freeze(self):
        model = _tiny_neox_model().to("jax")
        params = dict(model.named_parameters())
        for p in params.values():
            p.requires_grad_(False)
        assert all(not p.requires_grad for p in params.values())

    def test_frozen_params_receive_no_gradient_through_functional_call(self):
        model = _tiny_neox_model().to("jax")
        params = dict(model.named_parameters())
        for p in params.values():
            p.requires_grad_(False)

        ids = torch.randint(0, 100, (2, 6)).to("jax")
        targets = torch.randint(0, 100, (2, 6)).to("jax")
        out = functional_call(model, params, (ids,))
        loss = torch.nn.functional.cross_entropy(
            out.logits.view(-1, out.logits.shape[-1]), targets.view(-1)
        )
        assert not loss.requires_grad


class TestHookSurvivesFunctionalCall:
    def test_hook_fires_and_can_replace_output(self):
        model = _tiny_neox_model().to("jax")
        params = dict(model.named_parameters())

        calls = []

        def hook(module, inp, output):
            calls.append(type(output))
            return output

        handle = model.gpt_neox.layers[1].register_forward_hook(hook)
        try:
            ids = torch.randint(0, 100, (1, 4)).to("jax")
            out = functional_model(model, params, ids)
            assert len(calls) == 1
            assert out.logits.shape == (1, 4, 100)
        finally:
            handle.remove()

    def test_hook_replacement_actually_changes_output(self):
        model = _tiny_neox_model().to("jax")
        params = dict(model.named_parameters())

        def zero_hook(module, inp, output):
            return output * 0.0

        ids = torch.randint(0, 100, (1, 4)).to("jax")
        baseline = functional_model(model, params, ids).logits.to("cpu")

        handle = model.gpt_neox.layers[1].register_forward_hook(zero_hook)
        try:
            modified = functional_model(model, params, ids).logits.to("cpu")
        finally:
            handle.remove()

        assert not torch.allclose(baseline, modified)
