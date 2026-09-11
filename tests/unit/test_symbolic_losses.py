import numpy as np

from Residual.symbolic_head.losses import (
    combined_symbolic_loss,
    key_space_alignment_loss,
    value_space_regression_loss,
)


def test_value_space_regression_zero_when_equal():
    vec = np.ones(8, dtype=np.float32)
    assert value_space_regression_loss(vec, vec) == 0.0


def test_value_space_regression_is_squared_l2():
    """||(1,1)-(0,0)||^2 == 2."""
    pred = np.array([1.0, 1.0], dtype=np.float32)
    target = np.zeros(2, dtype=np.float32)
    assert value_space_regression_loss(pred, target) == 2.0


def test_key_space_loss_lower_when_query_matches_target():
    rng = np.random.default_rng(0)
    keys = rng.normal(0, 1, (5, 8)).astype(np.float32)
    target = keys[0]
    q_good = target.copy()
    q_bad = keys[4]
    loss_good = key_space_alignment_loss(q_good, target, keys, temperature=0.1)
    loss_bad = key_space_alignment_loss(q_bad, target, keys, temperature=0.1)
    assert loss_good < loss_bad
    assert loss_good >= 0.0


def test_combined_symbolic_loss_sums_weighted_terms():
    q = np.array([1.0, 0.0], dtype=np.float32)
    keys = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = combined_symbolic_loss(
        q_sym=q,
        target_key=keys[0],
        all_keys=keys,
        u_satt=np.ones(4, dtype=np.float32),
        target_value=np.ones(4, dtype=np.float32),
        lambda_key=2.0,
        lambda_value=3.0,
        temperature=1.0,
    )
    expected = 2.0 * out["l_sym_key"] + 3.0 * out["l_sym_value"]
    assert out["l_sym_value"] == 0.0
    assert abs(out["l_sym_total"] - expected) < 1e-6
