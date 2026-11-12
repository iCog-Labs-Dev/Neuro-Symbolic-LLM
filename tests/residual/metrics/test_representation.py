"""Unit tests for residual compressibility. No model needed."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from metrics.representation import (
    ablation_fidelity,
    canonical_low_rank_factors,
    canonical_low_rank_svd,
    canonicalize_adapter_params,
    category_naturality_error,
    causal_fingerprint_separability,
    compressibility_curve,
    factor_importance,
    gradient_conflict_summary,
    infer_support,
    loss_improvement,
    mixed_curvature_coupling,
    off_support_leakage,
    operator_commutator,
    operator_commutator_matrix,
    parameter_commutator,
    residual_compressibility,
    semantic_factor_metrics,
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


def test_semantic_factor_metrics_use_positive_signed_effects() -> None:
    effects = jnp.array(
        [
            [3.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [-4.0, 1.0, -2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    report = semantic_factor_metrics(effects)

    assert report["status"] == "partial"
    assert report["factor_status"] == ("ok", "ok", "unavailable")
    assert jnp.array_equal(factor_importance(effects), jnp.maximum(effects, 0.0))
    assert jnp.allclose(report["positive_mass"], jnp.array([4.0, 2.0, 0.0]))
    expected_effective = jnp.exp(-0.75 * jnp.log(0.75) - 0.25 * jnp.log(0.25))
    assert float(report["fracture"][0]) == pytest.approx(float(expected_effective / 4))
    assert float(report["fracture"][1]) == pytest.approx(0.5)
    assert float(report["overlap"][0, 1]) == pytest.approx(0.25)
    assert bool(jnp.isnan(report["fracture"][2]))
    assert bool(jnp.isnan(report["overlap"][0, 2]))


def test_semantic_factor_zero_positive_mass_is_unavailable() -> None:
    report = semantic_factor_metrics(jnp.array([[0.0, -1.0], [-2.0, 0.0]]))
    assert report["status"] == "unavailable"
    assert report["factor_status"] == ("unavailable", "unavailable")
    assert bool(jnp.all(jnp.isnan(report["fracture"])))
    assert bool(jnp.all(jnp.isnan(report["overlap"])))


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


def test_mixed_curvature_coupling_exact_and_scale_invariant() -> None:
    gradients = jnp.array([1.0, 1.0])
    orthogonal_samples = jnp.array([1.0, -1.0])
    assert float(mixed_curvature_coupling(gradients, 7 * gradients)) == pytest.approx(
        1.0
    )
    assert float(
        mixed_curvature_coupling(gradients, orthogonal_samples)
    ) == pytest.approx(0.0, abs=1e-7)
    assert float(
        mixed_curvature_coupling(jnp.array([1.0, 0.0]), gradients)
    ) == pytest.approx(2**-0.5)


def test_inferred_support_off_support_leakage() -> None:
    gradients = jnp.array([[1.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
    scores = jnp.array([1.0, -1.0, 0.0])
    assert jnp.array_equal(infer_support(scores), jnp.array([True, False, False]))
    assert float(off_support_leakage(gradients, scores)) == pytest.approx(0.75)
    assert float(off_support_leakage(gradients, jnp.zeros(3))) == pytest.approx(1.0)
    assert float(
        off_support_leakage(gradients, support=jnp.array([True, True, False]))
    ) == pytest.approx(0.5)


def test_causal_fingerprint_separability_exact_and_statuses() -> None:
    fingerprints = jnp.array([[0.0], [2.0], [10.0], [12.0]])
    labels = jnp.array([0, 0, 1, 1])
    labeled = causal_fingerprint_separability(fingerprints, labels, eps=0.0)
    assert labeled["status"] == "ok"
    assert labeled["labels_independent"] is True
    assert float(labeled["within_variance"]) == pytest.approx(1.0)
    assert float(labeled["between_variance"]) == pytest.approx(25.0)
    assert float(labeled["score"]) == pytest.approx(25.0)

    unavailable = causal_fingerprint_separability(fingerprints)
    assert unavailable["status"] == "unavailable"
    exploratory = causal_fingerprint_separability(fingerprints, n_clusters=2)
    assert exploratory["status"] == "exploratory"
    assert exploratory["labels_independent"] is False
    one_label = causal_fingerprint_separability(
        fingerprints, jnp.zeros(4, dtype=jnp.int32)
    )
    assert one_label["status"] == "unavailable"
    assert bool(jnp.isnan(one_label["score"]))


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


def test_category_naturality_error_exact() -> None:
    states = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    swap = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    identity = jnp.eye(2)
    assert float(
        category_naturality_error(states, swap, identity, swap)
    ) == pytest.approx(0.0)
    assert float(
        category_naturality_error(states, swap, identity, identity)
    ) == pytest.approx(2.0)
    assert float(
        category_naturality_error(jnp.array([[1.0, 2.0]]), jnp.zeros((1, 2)))
    ) == pytest.approx(5.0)


def test_ablation_fidelity_perfect_with_ties() -> None:
    effects = jnp.array([1.0, 1.0, -1.0, 0.0])
    report = ablation_fidelity(effects, effects, top_k=2)
    assert report["status"] == "ok"
    assert float(report["spearman"]) == pytest.approx(1.0)
    assert float(report["pearson"]) == pytest.approx(1.0)
    assert float(report["nrmse"]) == pytest.approx(0.0)
    assert float(report["sign_accuracy"]) == pytest.approx(1.0)
    assert float(report["top_k_overlap"]) == pytest.approx(1.0)


def test_ablation_fidelity_tie_aware_exact_example() -> None:
    report = ablation_fidelity(
        jnp.array([1.0, 1.0, 2.0]),
        jnp.array([1.0, 2.0, 2.0]),
        top_k=2,
    )
    assert float(report["spearman"]) == pytest.approx(0.5)
    assert float(report["pearson"]) == pytest.approx(0.5)
    assert float(report["nrmse"]) == pytest.approx((1 / 3) ** 0.5)
    assert float(report["sign_accuracy"]) == pytest.approx(1.0)
    assert float(report["top_k_overlap"]) == pytest.approx(0.5)
