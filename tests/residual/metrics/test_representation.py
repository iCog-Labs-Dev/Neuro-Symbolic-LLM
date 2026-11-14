"""Unit tests for residual compressibility. No model needed."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from metrics.representation import (
    canonical_low_rank_factors,
    canonical_low_rank_svd,
    canonicalize_adapter_params,
    compressibility_curve,
    gradient_conflict_summary,
    loss_improvement,
    operator_commutator,
    operator_commutator_matrix,
    parameter_commutator,
    residual_compressibility,
)


def test_loss_improvement_sign() -> None:
    assert float(loss_improvement(3.0, 2.5)) == pytest.approx(0.5)
    assert float(loss_improvement(3.0, 3.2)) == pytest.approx(-0.2)


def test_residual_compressibility() -> None:
    assert float(residual_compressibility(0.4, 0.5)) == pytest.approx(0.8)
    assert float(residual_compressibility(0.0, 0.0, eps=1e-3)) == 0.0


def test_negative_eps_raises() -> None:
    with pytest.raises(ValueError, match="eps"):
        residual_compressibility(0.1, 0.2, eps=-1.0)


def test_compressibility_curve_is_relative_to_largest_rank() -> None:
    curve = compressibility_curve({64: 0.50, 1: 0.10, 16: 0.45}, eps=0.0)
    assert list(curve) == [1, 16, 64]
    assert float(curve[1]) == pytest.approx(0.2)
    assert float(curve[16]) == pytest.approx(0.9)
    assert float(curve[64]) == pytest.approx(1.0)


@pytest.mark.parametrize("deltas", [{}, {0: 0.1, 4: 0.2}])
def test_compressibility_curve_rejects_bad_input(deltas: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        compressibility_curve(deltas)


def test_guarded_compressibility_ok() -> None:
    from metrics.representation import guarded_compressibility

    out = guarded_compressibility({4: 0.2, 16: 0.4})
    assert out["status"] == "ok"
    assert out["reference_rank"] == 16
    assert out["curve"][4] == pytest.approx(0.5)
    assert out["curve"][16] == pytest.approx(1.0)


@pytest.mark.parametrize("ref", [0.0, -0.3, 0.001])
def test_guarded_compressibility_rejects_nonpositive_or_noise_reference(ref) -> None:
    """A3 worse than base at the largest rank must not yield a huge/sign-flipped ratio."""
    from metrics.representation import guarded_compressibility

    out = guarded_compressibility({4: 0.2, 16: ref})
    assert out["status"] == "unreliable_reference"
    assert out["curve"] is None
    assert out["reference_improvement"] == ref


def test_guarded_compressibility_needs_two_ranks_and_flags_negative_ranks() -> None:
    from metrics.representation import guarded_compressibility

    assert guarded_compressibility({16: 0.4})["status"] == "insufficient_ranks"
    out = guarded_compressibility({4: -0.1, 16: 0.4})
    assert out["negative_ranks"] == [4]
    assert out["curve"][4] == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        guarded_compressibility({4: 0.1, 8: 0.2}, min_reference=0.0)
    assert (
        guarded_compressibility({4: float("nan"), 16: 0.4})["status"]
        == "insufficient_ranks"
    )


def test_canonical_low_rank_factors_are_gauge_invariant() -> None:
    a = jnp.array(
        [
            [1.0, 0.2],
            [0.0, 2.0],
            [1.5, -0.5],
            [-1.0, 1.0],
        ]
    )
    b = jnp.array(
        [
            [2.0, -1.0, 0.5, 1.0],
            [0.25, 1.5, -2.0, 0.0],
        ]
    )
    gauge = jnp.array([[2.0, 0.5], [-0.25, 1.5]])
    a_gauge = a @ gauge
    b_gauge = jnp.linalg.solve(gauge, b)

    canonical_a, canonical_b = canonical_low_rank_factors(a, b)
    gauge_a, gauge_b = canonical_low_rank_factors(a_gauge, b_gauge)

    assert canonical_a.shape == a.shape
    assert canonical_b.shape == b.shape
    assert jnp.allclose(canonical_a @ canonical_b, a @ b, atol=2e-5)
    assert jnp.allclose(gauge_a, canonical_a, atol=2e-5)
    assert jnp.allclose(gauge_b, canonical_b, atol=2e-5)


def test_canonical_factors_fix_repeated_singular_subspace_gauge() -> None:
    a = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gauge = jnp.array([[2.0, 1.0], [0.3, 1.7]])

    canonical_a, canonical_b = canonical_low_rank_factors(a, b)
    gauge_a, gauge_b = canonical_low_rank_factors(a @ gauge, jnp.linalg.solve(gauge, b))

    assert jnp.allclose(gauge_a, canonical_a, atol=2e-5)
    assert jnp.allclose(gauge_b, canonical_b, atol=2e-5)


def test_canonical_low_rank_svd_and_param_tree() -> None:
    a = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    b = jnp.array([[2.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    u, singular, vh = canonical_low_rank_svd(a, b)
    assert u.shape == (3, 2)
    assert singular.shape == (2,)
    assert vh.shape == (2, 3)
    assert jnp.allclose((u * singular) @ vh, a @ b, atol=1e-5)

    params = {
        "layer_1": {"A": a, "B": b, "tag": "kept"},
        "predictor": jnp.array([3.0]),
    }
    canonical = canonicalize_adapter_params(params)
    assert canonical["layer_1"]["tag"] == "kept"
    assert canonical["predictor"] is params["predictor"]
    assert jnp.allclose(
        canonical["layer_1"]["A"] @ canonical["layer_1"]["B"],
        a @ b,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="identity"):
        canonicalize_adapter_params(params, activation="relu")


def test_canonical_zero_adapter_has_zero_factors() -> None:
    a = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    b = jnp.zeros((3, 4), dtype=jnp.float32)
    canonical_a, canonical_b = canonical_low_rank_factors(a, b)
    assert jnp.array_equal(canonical_a, jnp.zeros_like(a))
    assert jnp.array_equal(canonical_b, jnp.zeros_like(b))


def test_gradient_conflict_summary_exact_and_ignores_zero_gradient() -> None:
    summary = gradient_conflict_summary(
        {
            "a": {"w": jnp.array([1.0, 0.0])},
            "b": {"w": jnp.array([-1.0, 0.0])},
            "c": {"w": jnp.array([0.0, 1.0])},
            "zero": {"w": jnp.zeros(2)},
        }
    )
    assert summary["status"] == "ok"
    assert int(summary["pair_count"]) == 3
    assert float(summary["mean"]) == pytest.approx(1 / 3)
    assert float(summary["max"]) == pytest.approx(1.0)
    assert float(summary["fraction_conflicting"]) == pytest.approx(1 / 3)
    assert bool(jnp.isnan(summary["conflict"][0, 3]))


def test_parameter_and_operator_commutators_exact() -> None:
    parameter_value = parameter_commutator(
        {"w": jnp.array([2.0, 0.0])},
        {"w": jnp.array([0.0, 2.0])},
        {"w": jnp.array([1.0, 1.0])},
        eps=0.0,
    )
    assert float(parameter_value) == pytest.approx(2.0)

    operator_i = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    operator_j = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    expected = jnp.array([[0.0, -2.0], [2.0, 0.0]])
    assert jnp.array_equal(operator_commutator_matrix(operator_i, operator_j), expected)
    assert float(operator_commutator(operator_i, operator_j, eps=0.0)) == pytest.approx(
        2**0.5
    )
    assert float(operator_commutator(jnp.eye(2), operator_j)) == pytest.approx(0.0)
