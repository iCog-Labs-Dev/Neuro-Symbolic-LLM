"""Unit tests for the HEP highway matrices (``residual/hep.py``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residual.cpc_heads import init_cpc_heads
from residual.hep import (
    apply_highway,
    hep_key,
    highway_for,
    init_hep,
    split_hep,
)

DY, DI, R = 40, 12, 4


def test_dense_params_keys_and_shapes() -> None:
    p = init_hep("dense", 0, [5, 6], DY, DI)
    assert set(p) == {hep_key(5), hep_key(6)} and hep_key(5) == "hep_5"
    assert p["hep_5"]["V"].shape == (DY, DI) and p["hep_5"]["V"].dtype == jnp.float32
    assert not np.array_equal(p["hep_5"]["V"], p["hep_6"]["V"])


def test_lowrank_params_shapes() -> None:
    p = init_hep("lowrank", 0, [5], DY, DI, rank=R)
    assert set(p["hep_5"]) == {"U", "W"}
    assert p["hep_5"]["U"].shape == (DY, R) and p["hep_5"]["W"].shape == (R, DI)
    # Two-level layout required by RunWriter.save_params.
    assert all(isinstance(v, dict) for v in p.values())


@pytest.mark.parametrize("kind", ["fixed_random", "tied"])
def test_fixed_kinds_have_no_parameters(kind: str) -> None:
    # Fixed matrices must not be in the tree, or Adam would update them.
    assert init_hep(kind, 0, [5, 6], DY, DI) == {}


def test_deterministic_in_seed_and_independent_of_layer_order() -> None:
    a = init_hep("dense", 0, [6, 5, 5], DY, DI)
    b = init_hep("dense", 0, [5, 6], DY, DI)
    c = init_hep("dense", 1, [5, 6], DY, DI)
    np.testing.assert_array_equal(a["hep_5"]["V"], b["hep_5"]["V"])
    assert not np.array_equal(a["hep_5"]["V"], c["hep_5"]["V"])


def test_scale_is_one_over_fan_in() -> None:
    v = np.asarray(init_hep("dense", 0, [5], 400, 64)["hep_5"]["V"])
    assert v.std() == pytest.approx(1 / np.sqrt(400), rel=0.1)


def test_streams_are_disjoint_from_the_cpc_heads() -> None:
    cpc = init_cpc_heads(0, [5], DI, DI)
    hep = init_hep("dense", 0, [5], DI, DI)
    assert not np.array_equal(hep["hep_5"]["V"], cpc["cpcq_5"]["W"])
    assert not np.array_equal(hep["hep_5"]["V"], cpc["cpca_5_1"]["M"])


def test_highway_for_each_kind() -> None:
    dense = init_hep("dense", 0, [5], DY, DI)
    low = init_hep("lowrank", 0, [5], DY, DI, rank=R)
    tied = jnp.ones((DY, DI))
    assert (
        highway_for("dense", 5, params=dense, d_out=DY, d_hidden=DI, rank=R)
        is dense["hep_5"]["V"]
    )
    u, w = highway_for("lowrank", 5, params=low, d_out=DY, d_hidden=DI, rank=R)
    assert u is low["hep_5"]["U"] and w is low["hep_5"]["W"]
    assert highway_for("tied", 5, tied=tied, d_out=DY, d_hidden=DI, rank=R) is tied
    fu, fw = highway_for("fixed_random", 5, seed=3, d_out=DY, d_hidden=DI, rank=R)
    assert fu.shape == (DY, R) and fw.shape == (R, DI)
    fu2, _ = highway_for("fixed_random", 5, seed=3, d_out=DY, d_hidden=DI, rank=R)
    fu3, _ = highway_for("fixed_random", 6, seed=3, d_out=DY, d_hidden=DI, rank=R)
    np.testing.assert_array_equal(fu, fu2)  # reproducible
    assert not np.array_equal(fu, fu3)  # per layer


def test_fixed_random_equals_the_trainable_lowrank_init() -> None:
    # Same generator: a fixed-random highway is the lowrank init frozen.
    low = init_hep("lowrank", 7, [5], DY, DI, rank=R)
    fu, fw = highway_for("fixed_random", 5, seed=7, d_out=DY, d_hidden=DI, rank=R)
    np.testing.assert_array_equal(fu, low["hep_5"]["U"])
    np.testing.assert_array_equal(fw, low["hep_5"]["W"])


def test_highway_for_errors() -> None:
    with pytest.raises(KeyError, match="hep_5"):
        highway_for("dense", 5, params={}, d_out=DY, d_hidden=DI)
    with pytest.raises(KeyError):
        highway_for("lowrank", 5, params=None, d_out=DY, d_hidden=DI)
    with pytest.raises(ValueError, match="tied"):
        highway_for("tied", 5, d_out=DY, d_hidden=DI)
    with pytest.raises(ValueError, match="shape"):
        highway_for("tied", 5, tied=jnp.ones((DY, DI + 1)), d_out=DY, d_hidden=DI)
    with pytest.raises(ValueError, match="kind"):
        highway_for("sparse", 5, d_out=DY, d_hidden=DI)


def test_apply_highway_dense_and_lowrank_agree() -> None:
    eps = jax.random.normal(jax.random.PRNGKey(0), (2, 3, DY))
    u = jax.random.normal(jax.random.PRNGKey(1), (DY, R))
    w = jax.random.normal(jax.random.PRNGKey(2), (R, DI))
    out = apply_highway((u, w), eps)
    assert out.shape == (2, 3, DI) and out.dtype == jnp.float32
    np.testing.assert_allclose(out, apply_highway(u @ w, eps), rtol=1e-4, atol=1e-4)
    assert apply_highway(u @ w, eps.astype(jnp.bfloat16)).dtype == jnp.float32


@pytest.mark.parametrize(
    "kwargs",
    [{"rank": 0}, {"rank": True}, {"d_out": 0}, {"d_hidden": 0}, {"kind": "x"}],
)
def test_init_validation(kwargs: dict) -> None:
    args = {"kind": "lowrank", "seed": 0, "layers": [5], "d_out": DY, "d_hidden": DI}
    with pytest.raises(ValueError):
        init_hep(**(args | kwargs))


def test_split_hep_selects_only_hep_entries() -> None:
    tree = {
        "layer_5": {"A": jnp.ones((2, 2))},
        "cpcq_5": {"W": jnp.ones((2, 2))},
        **init_hep("dense", 0, [5], DY, DI),
    }
    assert set(split_hep(tree)) == {"hep_5"}
