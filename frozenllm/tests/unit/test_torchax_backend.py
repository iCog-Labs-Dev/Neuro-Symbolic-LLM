"""Tests for substrate.torchax_backend."""

from __future__ import annotations

import jax.numpy as jnp
import torch
from substrate.torchax_backend import (
    call_jax_differentiable,
    enable_torchax,
    from_jax_array,
    is_on_torchax_device,
    is_torchax_enabled,
    to_jax_array,
    to_torchax_device,
)


class TestEnableIdempotency:
    def test_enable_is_safe_to_call_repeatedly(self):
        enable_torchax()
        enable_torchax()
        enable_torchax()
        assert is_torchax_enabled() is True


class TestValueConversion:
    def test_to_jax_array_round_trips_values(self):
        t = to_torchax_device(torch.randn(4, 8))
        arr = to_jax_array(t)
        back = from_jax_array(arr)
        assert torch.allclose(t.to("cpu"), back.to("cpu"))

    def test_to_jax_array_produces_real_jax_array(self):
        t = to_torchax_device(torch.randn(3, 3))
        arr = to_jax_array(t)
        # genuine jax array math should work directly, no torch involved
        result = jnp.sum(arr)
        assert result.shape == ()

    def test_from_jax_array_produces_torchax_tensor_on_jax_device(self):
        arr = jnp.ones((2, 2))
        t = from_jax_array(arr)
        assert is_on_torchax_device(t)


class TestDeviceManagement:
    def test_to_torchax_device_moves_tensor(self):
        t = torch.randn(4)
        assert not is_on_torchax_device(t)
        t_jax = to_torchax_device(t)
        assert is_on_torchax_device(t_jax)

    def test_to_torchax_device_moves_module(self):
        module = torch.nn.Linear(4, 4)
        moved = to_torchax_device(module)
        assert all(is_on_torchax_device(p) for p in moved.parameters())


class TestCallJaxDifferentiable:
    def test_matches_pure_torch_ground_truth(self):
        torch.manual_seed(0)
        d = 5
        h0_vals = torch.randn(3, d)
        w_vals = torch.randn(d, d) * 0.02
        b_vals = torch.zeros(d)

        # ground truth: pure torch, no torchax involved at all
        h0g = h0_vals.clone().requires_grad_()
        w_g = w_vals.clone().requires_grad_()
        bg = b_vals.clone().requires_grad_()
        out_g = h0g + torch.tanh(h0g @ w_g + bg)
        (out_g**2).sum().backward()

        # real raw-JAX function -- no torch syntax anywhere, standing in
        # for a genuine FabricPC node computation
        def raw_jax_residual(h0, w, b):
            return h0 + jnp.tanh(jnp.matmul(h0, w) + b)

        bridge = call_jax_differentiable(raw_jax_residual)

        h0j = to_torchax_device(h0_vals.clone())
        w_j = to_torchax_device(w_vals.clone()).requires_grad_()
        bj = to_torchax_device(b_vals.clone()).requires_grad_()
        out_j = bridge(h0j, w_j, bj)
        (out_j**2).sum().backward()

        assert torch.allclose(out_g, out_j.to("cpu"), atol=1e-5)
        assert w_j.grad is not None
        assert bj.grad is not None
        assert torch.allclose(w_g.grad, w_j.grad.to("cpu"), atol=1e-5)
        assert torch.allclose(bg.grad, bj.grad.to("cpu"), atol=1e-5)

    def test_jax_fn_receives_no_torch_syntax_requirement(self):
        # explicitly confirms that jax_fn can
        # use jax.lax / jnp freely, no torch ops required inside it
        import jax.lax as lax

        def uses_lax(x, y):
            return lax.add(x, y)

        bridge = call_jax_differentiable(uses_lax)
        x = to_torchax_device(torch.ones(3)).requires_grad_()
        y = to_torchax_device(torch.ones(3) * 2)
        out = bridge(x, y)
        out.sum().backward()
        assert torch.allclose(out.to("cpu"), torch.tensor([3.0, 3.0, 3.0]))
        assert x.grad is not None

    def test_does_not_require_h0_to_require_grad(self):
        # matches the real use case: h0 is a frozen model's hidden state
        # (requires_grad=False), only the residual's own params need grad
        def raw_jax_residual(h0, W, b):
            return h0 + jnp.tanh(jnp.matmul(h0, W) + b)

        bridge = call_jax_differentiable(raw_jax_residual)
        h0 = to_torchax_device(torch.randn(3, 4))  # requires_grad=False
        w = to_torchax_device(torch.randn(4, 4) * 0.02).requires_grad_()
        b = to_torchax_device(torch.zeros(4)).requires_grad_()

        out = bridge(h0, w, b)
        out.sum().backward()
        assert w.grad is not None
        assert b.grad is not None
