"""Unit tests for the contrastive future loss (``residual/aux_losses.py``)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from residual.a2_losses import anchor_positions, contrastive_future_loss
from residual.cpc_heads import a_key, init_cpc_heads, predict_future, project, q_key

D = M = 16


def heads(seed: int = 0, d: int = D, m: int = M, k: int = 1):
    p = init_cpc_heads(seed, [3], d, m, offsets=(k,), a_noise_std=0.05)
    return p[q_key(3)], p[a_key(3, k)]


def randn(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


_jit_loss = jax.jit(
    contrastive_future_loss,
    static_argnames=("offset", "tau", "max_anchors_per_seq"),
)


def loss_fn(q, a, anchors, targets, **kw):
    return _jit_loss(q, a, anchors, targets, **kw)


# --- an independent, loop-based reference of the definition -----------------------------


def reference_loss(q, a, anchors, targets, *, offset, tau, max_anchors=None):
    """Literal per-anchor implementation; positive branch under stop_gradient."""
    *lead, batch, seq_len, d = anchors.shape
    anc = anchors.reshape(-1, batch, seq_len, d)
    tgt = targets.reshape(-1, batch, seq_len, d)
    n = seq_len - offset
    stride = 1 if max_anchors is None else math.ceil(n / max_anchors)
    items = []  # (sequence index, predicted code, positive code)
    for s in range(anc.shape[0]):
        for b in range(batch):
            for t in range(0, n, stride):
                z = predict_future(a, project(q, anc[s, b, t]))
                y = jax.lax.stop_gradient(project(q, tgt[s, b, t + offset]))
                z = z / jnp.sqrt(jnp.sum(z * z) + 1e-8)
                y = y / jnp.sqrt(jnp.sum(y * y) + 1e-8)
                items.append((b, z, y))
    losses = []
    for bi, zi, yi in items:
        logits = [zi @ yi / tau]
        logits += [zi @ yj / tau for j, (bj, _, yj) in enumerate(items) if bj != bi]
        losses.append(jax.nn.logsumexp(jnp.stack(logits)) - logits[0])
    return jnp.mean(jnp.stack(losses))


@pytest.mark.parametrize(
    "shape, offset, max_anchors",
    [
        ((3, 5, D), 1, None),
        ((3, 6, D), 3, None),
        ((3, 9, D), 1, 3),  # stride > 1
        ((2, 3, 4, D), 2, None),  # leading step axis S = 2
    ],
)
def test_matches_reference_value_and_gradients(shape, offset, max_anchors) -> None:
    q, a = heads(k=offset)
    anchors, targets = randn(*shape, seed=1), randn(*shape, seed=2)
    kw = {"offset": offset, "tau": 0.2}

    def mine(q, a, anchors):
        return loss_fn(q, a, anchors, targets, max_anchors_per_seq=max_anchors, **kw)[0]

    def ref(q, a, anchors):
        return reference_loss(q, a, anchors, targets, max_anchors=max_anchors, **kw)

    v1, g1 = jax.jit(jax.value_and_grad(mine, argnums=(0, 1, 2)))(q, a, anchors)
    v2, g2 = jax.jit(jax.value_and_grad(ref, argnums=(0, 1, 2)))(q, a, anchors)
    np.testing.assert_allclose(v1, v2, rtol=1e-5)
    # Equal gradients also prove the positive branch is under stop_gradient.
    for x, y in zip(
        jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2), strict=False
    ):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_aligned_positives_give_near_zero_loss_and_perfect_retrieval() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    seq = randn(4, 6, D, seed=3)
    offset = 1
    # The anchor at t has exactly the content of the target at t + offset.
    anchors = jnp.concatenate([seq[:, offset:], seq[:, :offset]], axis=1)
    loss, m = loss_fn(q, a, anchors, seq, offset=offset, tau=0.05)
    assert float(loss) < 1e-2
    assert float(m["cpc_top1"]) == 1.0
    assert float(m["cpc_pos_sim"]) == pytest.approx(1.0, abs=1e-4)
    assert float(m["cpc_neg_sim"]) < 0.9


@pytest.mark.parametrize("batch, steps", [(3, 1), (4, 1), (3, 2)])
def test_uninformative_codes_give_log_n_loss(batch, steps) -> None:
    q, a = heads()
    # Constant states: LayerNorm maps them to 0, so every similarity is 0.
    shape = (steps, batch, 5, D) if steps > 1 else (batch, 5, D)
    x = jnp.full(shape, 2.0)
    loss, m = loss_fn(q, a, x, x, offset=1, tau=0.1)
    n_pos = 4
    expected = math.log(1 + (batch - 1) * n_pos * steps)
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert float(m["cpc_valid"]) == 1.0


def test_other_positions_of_the_same_sequence_are_not_negatives() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    batch, seq_len = 4, 9
    per_seq = randn(batch, 1, D, seed=4)
    x = jnp.broadcast_to(per_seq, (batch, seq_len, D))  # identical at every position
    loss, m = loss_fn(q, a, x, x, offset=1, tau=0.05)
    n_pos = seq_len - 1
    # If same-sequence entries counted as negatives, ties with the positive would force
    # loss >= log(n_pos). Masked, only the (random) other sequences compete.
    assert float(loss) < 0.25 * math.log(n_pos)
    assert float(m["cpc_top1"]) == 1.0


@pytest.mark.parametrize("offset", [1, 3])
def test_positions_beyond_the_sequence_are_ignored(offset) -> None:
    q, a = heads(k=offset)
    anchors, targets = randn(3, 10, D, seed=5), randn(3, 10, D, seed=6)
    base, _ = loss_fn(q, a, anchors, targets, offset=offset)
    # Last `offset` anchor positions have no future: must not matter.
    a2 = anchors.at[:, -offset:].set(randn(3, offset, D, seed=7))
    np.testing.assert_array_equal(base, loss_fn(q, a, a2, targets, offset=offset)[0])
    # First `offset` target positions are nobody's future: must not matter.
    t2 = targets.at[:, :offset].set(randn(3, offset, D, seed=8))
    np.testing.assert_array_equal(base, loss_fn(q, a, anchors, t2, offset=offset)[0])


def test_single_sequence_has_no_negatives() -> None:
    q, a = heads()
    for shape in ((1, 6, D), (3, 1, 6, D)):
        anchors, targets = randn(*shape, seed=9), randn(*shape, seed=10)

        def f(q, a, anchors, targets=targets):
            return loss_fn(q, a, anchors, targets, offset=1)

        (loss, m), grads = jax.value_and_grad(f, argnums=(0, 1, 2), has_aux=True)(
            q, a, anchors
        )
        assert float(loss) == 0.0
        assert float(m["cpc_valid"]) == 0.0
        assert all(
            np.isfinite(g).all() and not np.any(g)
            for g in jax.tree_util.tree_leaves(grads)
        )


def test_same_sequence_at_other_leading_steps_is_not_a_negative() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    steps, batch, seq_len = 3, 4, 6
    per_seq = randn(1, batch, 1, D, seed=11)
    x = jnp.broadcast_to(
        per_seq, (steps, batch, seq_len, D)
    )  # same at every step/position
    loss, m = loss_fn(q, a, x, x, offset=1, tau=0.05)
    # Counted as negatives, the steps * (seq_len - 1) - 1 copies of the sequence would tie
    # with the positive and push the loss to >= log(steps * n_pos).
    assert float(loss) < 0.25 * math.log(steps * (seq_len - 1))
    assert float(m["cpc_top1"]) == 1.0
    assert float(m["cpc_n_anchors"]) == steps * batch * (seq_len - 1)


def test_extra_negative_copies_raise_the_loss() -> None:
    # Duplicating the batch along a leading axis adds copies of *other* sequences'
    # positives, i.e. more negatives: the loss must grow (so the mask is not too greedy).
    q, a = heads()
    anchors, targets = randn(4, 7, D, seed=11), randn(4, 7, D, seed=12)
    base, _ = loss_fn(q, a, anchors, targets, offset=1, tau=0.2)
    rep = lambda x: jnp.stack([x, x, x])  # noqa: E731
    stacked, _ = loss_fn(q, a, rep(anchors), rep(targets), offset=1, tau=0.2)
    assert float(stacked) > float(base)


def test_gradient_audit() -> None:
    q, a = heads()
    anchors, targets = randn(4, 8, D, seed=13), randn(4, 8, D, seed=14)

    def f(q, a, anchors, targets):
        return loss_fn(q, a, anchors, targets, offset=1, tau=0.1)[0]

    gq, ga, g_anchor, g_target = jax.grad(f, argnums=(0, 1, 2, 3))(
        q, a, anchors, targets
    )
    assert not np.any(g_target), "the positive must not receive gradient"
    assert np.abs(gq["W"]).max() > 0 and np.abs(ga["M"]).max() > 0
    assert np.abs(g_anchor).max() > 0
    # No gradient into anchor positions that have no future.
    assert not np.any(g_anchor[:, -1])
    for g in (gq["W"], ga["M"], g_anchor):
        assert np.isfinite(g).all()


def test_anchor_positions() -> None:
    assert anchor_positions(5, None) == (0, 1, 2, 3, 4)
    assert anchor_positions(127, 256) == tuple(range(127))
    pos = anchor_positions(127, 50)  # stride 3
    assert pos[0] == 0 and pos[1] - pos[0] == 3 and len(pos) <= 50
    assert pos == anchor_positions(127, 50)  # deterministic
    assert len(anchor_positions(1000, 256)) <= 256
    with pytest.raises(ValueError):
        anchor_positions(0, None)
    with pytest.raises(ValueError):
        anchor_positions(5, 0)


def test_max_anchors_limits_the_anchor_count() -> None:
    q, a = heads()
    x = randn(4, 40, D, seed=15)
    _, m_all = loss_fn(q, a, x, x, offset=1)
    _, m_cap = loss_fn(q, a, x, x, offset=1, max_anchors_per_seq=10)
    assert float(m_all["cpc_n_anchors"]) == 4 * 39
    assert float(m_cap["cpc_n_anchors"]) <= 4 * 10


def test_numerically_stable_for_tiny_temperature_and_low_precision() -> None:
    q, a = heads()
    anchors, targets = randn(4, 8, D, seed=16), randn(4, 8, D, seed=17)

    def f(q, a, anchors):
        return loss_fn(q, a, anchors, targets, offset=1, tau=1e-3)[0]

    v, g = jax.value_and_grad(f, argnums=(0, 1, 2))(q, a, anchors)
    assert np.isfinite(v)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(g))

    f32, _ = loss_fn(q, a, anchors, targets, offset=1, tau=0.2)
    bf16, _ = loss_fn(
        q,
        a,
        anchors.astype(jnp.bfloat16),
        targets.astype(jnp.bfloat16),
        offset=1,
        tau=0.2,
    )
    assert bf16.dtype == jnp.float32 and np.isfinite(bf16)
    np.testing.assert_allclose(bf16, f32, rtol=5e-2)


def test_jit_matches_eager() -> None:
    q, a = heads()
    anchors, targets = randn(4, 8, D, seed=18), randn(4, 8, D, seed=19)

    def f(q, a, anchors, targets):
        return loss_fn(q, a, anchors, targets, offset=2, tau=0.1, max_anchors_per_seq=4)

    (e, em), eg = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(
        q, a, anchors, targets
    )
    (j, jm), jg = jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=True))(
        q, a, anchors, targets
    )
    np.testing.assert_allclose(e, j, rtol=1e-5)
    for x, y in zip(
        jax.tree_util.tree_leaves(eg), jax.tree_util.tree_leaves(jg), strict=False
    ):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)
    assert set(em) == set(jm)


@pytest.mark.parametrize(
    "kwargs, shape, match",
    [
        ({"offset": 0}, (3, 6, D), "offset"),
        ({"offset": -1}, (3, 6, D), "offset"),
        ({"offset": True}, (3, 6, D), "offset"),
        ({"offset": 1.0}, (3, 6, D), "offset"),
        ({"offset": 6}, (3, 6, D), "seq_len"),
        ({"tau": 0.0}, (3, 6, D), "tau"),
        ({"tau": -0.1}, (3, 6, D), "tau"),
        ({"max_anchors_per_seq": 0}, (3, 6, D), "max_anchors"),
        ({}, (6, D), "expected"),
    ],
)
def test_validation(kwargs, shape, match) -> None:
    q, a = heads()
    x = randn(*shape)
    with pytest.raises(ValueError, match=match):
        loss_fn(q, a, x, x, **kwargs)


def test_shape_mismatch_is_rejected() -> None:
    q, a = heads()
    with pytest.raises(ValueError, match="equal shapes"):
        loss_fn(q, a, randn(3, 6, D), randn(3, 7, D))


def test_heads_learn_to_predict_the_future_of_a_linear_process() -> None:
    """The loss is actually minimisable: trained heads retrieve the true future."""
    d, m, batch, seq_len = 16, 8, 16, 12
    rng = np.random.default_rng(0)
    rot = np.linalg.qr(rng.normal(size=(d, d)))[0]  # h_{t+1} = R h_t + noise
    h = np.zeros((batch, seq_len, d), np.float32)
    h[:, 0] = rng.normal(size=(batch, d))
    for t in range(1, seq_len):
        h[:, t] = h[:, t - 1] @ rot.T + 0.05 * rng.normal(size=(batch, d))
    x = jnp.asarray(h)

    params = init_cpc_heads(0, [3], d, m)
    params = {"q": params[q_key(3)], "a": params[a_key(3, 1)]}

    def f(p):
        return loss_fn(p["q"], p["a"], x, x, offset=1, tau=0.1)

    opt = optax.adam(2e-2)
    state = opt.init(params)
    step = jax.jit(jax.value_and_grad(f, has_aux=True))
    (first, m0), _ = step(params)
    for _ in range(300):
        (_, _), grads = step(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    (last, m1), _ = step(params)

    chance = 1 / (1 + (batch - 1) * (seq_len - 1))
    assert float(last) < 0.5 * float(first)
    assert float(m1["cpc_top1"]) > max(10 * chance, 2 * float(m0["cpc_top1"]))