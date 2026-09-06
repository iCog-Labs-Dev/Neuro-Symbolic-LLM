"""Tests for substrate.torchax_transformers"""

from __future__ import annotations

import copy

import torch
import torchax
from substrate.torchax_models import functional_model
from torch.func import functional_call
from transformers import GPT2Config, GPT2LMHeadModel

torchax.enable_globally()


def _tiny_model() -> GPT2LMHeadModel:
    cfg = GPT2Config(n_layer=4, n_head=2, n_embd=32, vocab_size=100, n_positions=16)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


class TestFunctionalCallCorrectness:
    def test_produces_correct_output_shape(self):
        model = _tiny_model().to("jax")
        params = dict(model.named_parameters())
        ids = torch.randint(0, 100, (2, 6)).to("jax")
        out = functional_model(model, params, ids)
        assert out.logits.shape == (2, 6, 100)

    def test_numerically_matches_plain_pytorch(self):
        model_plain = _tiny_model()
        torch.manual_seed(0)
        ids = torch.randint(0, 100, (2, 6))
        with torch.no_grad():
            ref_logits = model_plain(input_ids=ids).logits

        model_jax = copy.deepcopy(model_plain).to("jax")
        params = dict(model_jax.named_parameters())
        out = functional_model(model_jax, params, ids.to("jax"))
        jax_logits = out.logits.to("cpu")

        assert torch.allclose(ref_logits, jax_logits, atol=1e-4)
        diff = (ref_logits - jax_logits).abs()

        assert float(diff.max()) < 1e-6


class TestFreezing:
    def test_params_do_not_require_grad_after_explicit_freeze(self):
        model = _tiny_model().to("jax")
        params = dict(model.named_parameters())
        for p in params.values():
            p.requires_grad_(False)
        assert all(not p.requires_grad for p in params.values())

    def test_frozen_params_receive_no_gradient_through_functional_call(self):
        model = _tiny_model().to("jax")
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
        model = _tiny_model().to("jax")
        params = dict(model.named_parameters())

        calls = []

        def hook(module, inp, output):
            calls.append(output.shape)
            return output

        handle = model.transformer.h[1].register_forward_hook(hook)
        try:
            ids = torch.randint(0, 100, (1, 4)).to("jax")
            out = functional_model(model, params, ids)
            assert len(calls) == 1
            assert out.logits.shape == (1, 4, 100)
        finally:
            handle.remove()

    def test_hook_replacement_actually_changes_output(self):
        model = _tiny_model().to("jax")
        params = dict(model.named_parameters())

        def zero_hook(module, inp, output):
            return output * 0.0

        ids = torch.randint(0, 100, (1, 4)).to("jax")
        baseline = functional_model(model, params, ids).logits.to("cpu")

        handle = model.transformer.h[1].register_forward_hook(zero_hook)
        try:
            modified = functional_model(model, params, ids).logits.to("cpu")
        finally:
            handle.remove()

        assert not torch.allclose(baseline, modified)
