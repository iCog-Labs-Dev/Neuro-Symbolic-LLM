"""Analytic audit checks; these fixtures are not trained benchmark results."""

import jax.numpy as jnp
import numpy as np
import pytest

from metrics.adapter_audits import (
    measure_adapter_ablations,
    measure_cross_adapter_curvature,
)


def parameters():
    return {
        "layer_1": {"A": jnp.ones(2), "B": jnp.array([1.0, 2.0])},
        "layer_2": {"A": jnp.ones(2), "B": jnp.array([3.0, 4.0])},
    }


def test_ablations_record_signed_effects_without_mutating_checkpoint():
    params = parameters()
    audit = measure_adapter_ablations(
        {
            "helpful": lambda p: 10 - p["layer_1"]["B"].sum(),
            "harmful": lambda p: p["layer_2"]["B"].sum(),
        },
        params,
    )
    assert audit["effects"] == {
        "layer_1": {"helpful": 3.0, "harmful": 0.0},
        "layer_2": {"helpful": 0.0, "harmful": -7.0},
    }
    assert audit["mean_loss_increase"] == -1
    assert audit["fingerprint_rms"] == pytest.approx(np.sqrt(58 / 4))
    np.testing.assert_array_equal(params["layer_1"]["B"], [1, 2])
    np.testing.assert_array_equal(params["layer_2"]["B"], [3, 4])


def test_cross_adapter_norm_covers_both_factors_and_excludes_diagonal_blocks():
    def loss(p):
        a, b = p["layer_1"], p["layer_2"]
        return (
            3 * jnp.vdot(a["A"], b["B"])
            + 2 * jnp.vdot(a["B"], b["A"])
            + 100 * jnp.vdot(a["A"], a["B"])
        )

    audit = measure_cross_adapter_curvature(loss, parameters(), samples=3)
    assert audit["value"] == pytest.approx(np.sqrt(26), rel=0.002)
    assert list(audit["pairs"]) == ["layer_1:layer_2"]
    independent = measure_cross_adapter_curvature(
        lambda p: sum(jnp.sum(b["B"] ** 2) for b in p.values()), parameters()
    )
    assert independent["value"] == 0


def test_audits_reject_undefined_or_nonfinite_measurements():
    with pytest.raises(ValueError, match="at least two"):
        measure_cross_adapter_curvature(
            lambda p: 0, {"layer_1": parameters()["layer_1"]}
        )
    with pytest.raises(ValueError, match="positive integer"):
        measure_cross_adapter_curvature(lambda p: 0, parameters(), samples=0)
    with pytest.raises(ValueError, match="non-finite"):
        measure_adapter_ablations({"bad": lambda p: float("nan")}, parameters())
