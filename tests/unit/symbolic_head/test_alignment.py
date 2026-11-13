"""Alignment equations, fixed targets and independent predictor gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residual.symbolic_head.alignment import ValuePredictor, alignment_loss


def inputs():
    return (
        jnp.asarray([[1.0, 0.0], [0.0, 2.0]]),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        jnp.asarray([[0.0, 1.0], [2.0, 2.0], [4.0, 3.0]]),
        jnp.asarray([0, 2], dtype=jnp.int32),
    )


def test_losses_match_full_bank_equations_under_jit():
    q, pred, keys, values, labels = inputs()
    logits = np.asarray(q) @ np.asarray(keys).T
    expected_key = np.mean(
        np.log(np.exp(logits).sum(axis=-1)) - logits[np.arange(2), labels]
    )
    expected_value = np.mean(
        np.sum((np.asarray(pred) - np.asarray(values)[labels]) ** 2, axis=-1)
    )
    actual = jax.jit(alignment_loss)(q, pred, keys, values, labels)
    np.testing.assert_allclose(actual.key, expected_key, rtol=1e-6)
    np.testing.assert_allclose(actual.value, expected_value, rtol=1e-6)
    np.testing.assert_allclose(actual.total, expected_key + expected_value, rtol=1e-6)


def test_query_and_value_gradients_match_analytic_derivatives():
    q, pred, keys, values, labels = inputs()
    gradients = jax.jit(
        jax.grad(lambda *args: alignment_loss(*args).total, argnums=(0, 1, 2, 3))
    )(q, pred, keys, values, labels)
    probabilities = jax.nn.softmax(q @ keys.T, axis=-1)
    expected_q = (probabilities @ keys - keys[labels]) / q.shape[0]
    expected_value = 2 * (pred - values[labels]) / q.shape[0]
    np.testing.assert_allclose(gradients[0], expected_q, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(gradients[1], expected_value, rtol=1e-6)
    np.testing.assert_array_equal(gradients[2], jnp.zeros_like(keys))
    np.testing.assert_array_equal(gradients[3], jnp.zeros_like(values))


@pytest.mark.parametrize("label", [-1, 3])
def test_invalid_label_is_not_silently_clipped_or_wrapped(label):
    q, pred, keys, values, labels = inputs()
    labels = labels.at[0].set(label)
    actual = jax.jit(alignment_loss)(q, pred, keys, values, labels)
    assert all(np.isnan(float(term)) for term in actual)


def test_float_labels_are_rejected():
    q, pred, keys, values, labels = inputs()
    with pytest.raises(ValueError, match="integer target"):
        alignment_loss(q, pred, keys, values, labels.astype(jnp.float32))


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4])
def test_incompatible_shapes_are_rejected(position):
    arrays = list(inputs())
    arrays[position] = arrays[position][:-1]
    with pytest.raises(ValueError):
        alignment_loss(*arrays)


def test_predictor_has_independent_parameters_and_nonzero_gradient():
    predictor = ValuePredictor(3, 2)
    params = predictor.init_params(0)
    hidden = jnp.asarray([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(
        predictor.apply(params, hidden), hidden @ params["V_pred"]
    )
    gradient = jax.grad(lambda p: jnp.sum(jnp.square(predictor.apply(p, hidden))))(
        params
    )
    assert set(params) == {"V_pred"}
    assert params["V_pred"].shape == (3, 2)
    assert np.isfinite(gradient["V_pred"]).all()
    assert np.linalg.norm(gradient["V_pred"]) > 0
    np.testing.assert_array_equal(params["V_pred"], predictor.init_params(0)["V_pred"])


def test_predictor_rejects_writeback_shaped_parameters():
    predictor = ValuePredictor(3, 2)
    with pytest.raises(ValueError, match="parameter keys or shapes"):
        predictor.apply({"V_pred": jnp.zeros((2, 3))}, jnp.ones((1, 3)))


@pytest.mark.parametrize("dimensions", [(0, 2), (3, -1), (True, 2)])
def test_predictor_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="positive integers"):
        ValuePredictor(*dimensions)
