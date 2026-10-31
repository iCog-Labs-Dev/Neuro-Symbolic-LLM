"""Regressions for the numerical edge cases reported in the PR #17 update."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residual.a2_losses import contrastive_future_loss, hidden_prediction_loss
from residual.cpc_heads import a_key, init_cpc_heads, q_key
from residual.local_predictor import PredictorConfig
from residual.stability import StabilityConfig, StabilityMonitor, residual_smoothness
from stages.stage_A.a2 import (
    A2Config,
    CPCConfig,
    HEPConfig,
    HiddenConfig,
    StabLossConfig,
)
from stages.stage_base import TrainingConfig


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "factory",
    [
        lambda x: A2Config(lambda_kl=x),
        lambda x: A2Config(lambda_wd=x),
        lambda x: CPCConfig(weight=x),
        lambda x: CPCConfig(tau=x),
        lambda x: CPCConfig(a_init_noise=x),
        lambda x: StabLossConfig(weight=x),
        lambda x: HEPConfig(alpha=x),
        lambda x: HiddenConfig(predictor_lr_mult=x),
        lambda x: PredictorConfig(out_init_std=x),
        lambda x: StabilityConfig(delta=x),
        lambda x: TrainingConfig(learning_rate=x),
    ],
)
def test_nonfinite_settings_rejected(factory, value):
    with pytest.raises(ValueError):
        factory(value)


def test_one_token_next_token_training_rejected():
    with pytest.raises(ValueError, match="seq_len must be >= 2"):
        TrainingConfig(seq_len=1)


def test_empty_hidden_prediction_is_finite_and_has_zero_gradient():
    x = jnp.ones((2, 1, 4))
    cfg = A2Config()
    out = hidden_prediction_loss(x, x, x, cfg)
    assert all(float(v) == 0.0 for v in out.values())
    grad = jax.grad(lambda p: hidden_prediction_loss(p, x, x, cfg)["loss"])(x)
    np.testing.assert_array_equal(grad, jnp.zeros_like(x))


@pytest.mark.parametrize("batch", [1, 2])
def test_cpc_without_negatives_has_zero_effective_anchors(batch):
    heads = init_cpc_heads(0, [1], 4, 3)
    x = jax.random.normal(jax.random.PRNGKey(1), (batch, 4, 4))
    for mask in [
        jnp.zeros((batch, 4), bool),
        jnp.ones((batch, 4), bool).at[1:].set(False),
    ]:

        def loss(p, mask=mask):
            return contrastive_future_loss(
                heads[q_key(1)], heads[a_key(1, 1)], p, x, mask=mask, backend="jax"
            )[0]

        value, metrics = contrastive_future_loss(
            heads[q_key(1)], heads[a_key(1, 1)], x, x, mask=mask, backend="jax"
        )
        assert (
            float(value)
            == float(metrics["cpc_valid"])
            == float(metrics["cpc_n_anchors"])
            == 0.0
        )
        np.testing.assert_array_equal(jax.grad(loss)(x), jnp.zeros_like(x))


def test_monitor_ignores_masked_positions_and_has_no_gradient():
    from types import SimpleNamespace

    h = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    out = h + 1
    mask = jnp.array([[True, True, False, False]])
    monitor = StabilityMonitor(StabilityConfig(mode="diagnostic"), [1])
    adapter = SimpleNamespace()  # corrected states supplied; no adapter call needed
    before = monitor.metrics(adapter, {}, {1: h}, {1: out}, mask=mask)
    after = monitor.metrics(
        adapter, {}, {1: h.at[:, 2:].set(123)}, {1: out.at[:, 2:].set(1000)}, mask=mask
    )
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    zeros = residual_smoothness(out, h, jnp.zeros_like(mask))
    assert all(float(v) == 0 for v in zeros.values())
    grad = jax.grad(
        lambda x: monitor.metrics(adapter, {}, {1: h}, {1: x}, mask=mask)["res_smooth"]
    )(out)
    np.testing.assert_array_equal(grad, jnp.zeros_like(out))
