"""Tests for the contrastive future loss (``residual/a2_losses.py``).

it teststhe loss against an independent loop-based reference (value, gradient,
        subsampling, jit).
It tests padding mask and head backend. The paper's note: "Mask padding positions and
        positions where t + k lies outside the sequence." A position is usable only if
        it is a real token; masked positions are never an anchor, a positive or a
        negative.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from residual.a2_losses import (
    anchor_positions,
    contrastive_future_loss,
)
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


# ----------------------------------------------------------------------
# Padding mask and head backend
# ----------------------------------------------------------------------

TAU = 0.2


def _heads(k: int = 1, seed: int = 0):
    p = init_cpc_heads(seed, [3], D, M, offsets=(k,), a_noise_std=0.1)
    return p[q_key(3)], p[a_key(3, k)]


def _x(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


_loss = jax.jit(
    contrastive_future_loss,
    static_argnames=("offset", "tau", "max_anchors_per_seq", "backend"),
)


def _right_padded(lengths, seq_len: int) -> jax.Array:
    return jnp.arange(seq_len)[None, :] < jnp.asarray(lengths)[:, None]


def reference_masked(q, a, anchors, targets, mask, *, offset, tau):
    """Literal loop: valid anchors need t and t+k real; negatives are the positives of
    the valid anchors of *other* sequences; anchors without any negative are skipped."""
    batch, seq_len, _ = anchors.shape
    items = []
    for b in range(batch):
        for t in range(seq_len - offset):
            if bool(mask[b, t]) and bool(mask[b, t + offset]):
                z = predict_future(a, project(q, anchors[b, t], "jax"), "jax")
                y = jax.lax.stop_gradient(project(q, targets[b, t + offset], "jax"))
                z = z / jnp.sqrt(jnp.sum(z * z) + 1e-8)
                y = y / jnp.sqrt(jnp.sum(y * y) + 1e-8)
                items.append((b, z, y))
    losses = []
    for bi, zi, yi in items:
        neg = [zi @ yj / tau for bj, _, yj in items if bj != bi]
        if not neg:
            continue
        logits = jnp.stack([zi @ yi / tau, *neg])
        losses.append(jax.nn.logsumexp(logits) - logits[0])
    return jnp.mean(jnp.stack(losses)) if losses else jnp.zeros(())


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


# --- mask semantics --------------------------------------------------------------------------


def test_all_true_mask_equals_no_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=1), _x(3, 7, D, seed=2)
    ones = jnp.ones((3, 7), dtype=bool)

    def f(mask):
        return jax.value_and_grad(
            lambda q, a, anc: _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0],
            argnums=(0, 1, 2),
        )(q, a, anc)

    (v0, g0), (v1, g1) = f(None), f(ones)
    np.testing.assert_allclose(v0, v1, rtol=1e-5, atol=1e-7)
    for x, y in zip(_leaves(g0), _leaves(g1), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-7)
    m0 = _loss(q, a, anc, tgt, offset=1, tau=TAU)[1]
    m1 = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=ones)[1]
    for k in m0:
        np.testing.assert_allclose(m0[k], m1[k], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "mask",
    [
        _right_padded([9, 5, 7], 9),  # right padding
        jnp.asarray(  # a hole in the middle of a sequence, plus a short one
            [[1, 1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1] * 9], bool
        ),
    ],
)
@pytest.mark.parametrize("offset", [1, 2])
def test_matches_a_literal_reference_with_padding(mask, offset) -> None:
    q, a = _heads(k=offset)
    anc, tgt = _x(3, 9, D, seed=3), _x(3, 9, D, seed=4)

    def mine(q, a, anc):
        return _loss(q, a, anc, tgt, offset=offset, tau=TAU, mask=mask)[0]

    def ref(q, a, anc):
        return reference_masked(q, a, anc, tgt, mask, offset=offset, tau=TAU)

    v1, g1 = jax.value_and_grad(mine, argnums=(0, 1, 2))(q, a, anc)
    v2, g2 = jax.value_and_grad(ref, argnums=(0, 1, 2))(q, a, anc)
    np.testing.assert_allclose(v1, v2, rtol=1e-5)
    for x, y in zip(_leaves(g1), _leaves(g2), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_masked_positions_have_no_influence_and_no_gradient() -> None:
    q, a = _heads()
    mask = _right_padded([9, 5, 7], 9)
    anc, tgt = _x(3, 9, D, seed=5), _x(3, 9, D, seed=6)
    # Different random content on the padding only.
    junk = jnp.where(mask[..., None], 0.0, _x(3, 9, D, seed=32) * 100.0)

    def f(anc, tgt):
        return _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]

    v0 = f(anc, tgt)
    v1 = f(anc + junk, tgt + junk)
    np.testing.assert_allclose(v0, v1, rtol=1e-5, atol=1e-7)
    g_anc, g_tgt = jax.grad(f, argnums=(0, 1))(anc, tgt)
    assert not np.any(g_anc[~mask])  # padding receives no gradient
    assert bool(jnp.all(jnp.isfinite(g_anc)))
    assert not np.any(g_tgt)  # positives are stop-gradient


def test_a_position_only_counts_when_both_t_and_t_plus_k_are_real() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 8, D, seed=7), _x(3, 8, D, seed=8)
    # Only token 4 of sequence 0 is masked: anchors (0, 4) and (0, 3) disappear.
    mask = jnp.ones((3, 8), bool).at[0, 4].set(False)

    def f(anc, tgt, mask):
        return _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)

    _, m = f(anc, tgt, mask)
    assert float(m["cpc_n_anchors"]) == 3 * 7 - 2
    # Changing the content of the masked token changes nothing.
    # (random content: a constant shift would be removed by the head's LayerNorm)
    anc2 = anc.at[0, 4].set(_x(D, seed=30) * 5.0)
    tgt2 = tgt.at[0, 4].set(_x(D, seed=31) * 5.0)
    np.testing.assert_allclose(
        f(anc, tgt, mask)[0], f(anc2, tgt2, mask)[0], rtol=1e-5, atol=1e-7
    )
    # ... but without the mask it does.
    assert abs(float(f(anc, tgt, None)[0]) - float(f(anc2, tgt2, None)[0])) > 1e-3


def test_a_fully_masked_sequence_equals_dropping_it() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=9), _x(3, 7, D, seed=10)
    mask = jnp.asarray([[1] * 7, [1] * 7, [0] * 7], bool)
    masked, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)
    dropped, md = _loss(q, a, anc[:2], tgt[:2], offset=1, tau=TAU)
    np.testing.assert_allclose(masked, dropped, rtol=1e-5, atol=1e-7)
    for k in ("cpc_top1", "cpc_pos_sim", "cpc_neg_sim", "cpc_n_anchors"):
        np.testing.assert_allclose(m[k], md[k], rtol=1e-5, atol=1e-7)


def test_padded_negatives_are_not_used() -> None:
    """Moving a padded sequence's content must not change the other sequences' loss."""
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=11), _x(3, 7, D, seed=12)
    mask = _right_padded([7, 7, 3], 7)
    base = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    tgt_p = tgt.at[2, 3:].set(_x(4, D, seed=13) * 10.0)  # only the padded tail
    np.testing.assert_allclose(
        base,
        _loss(q, a, anc, tgt_p, offset=1, tau=TAU, mask=mask)[0],
        rtol=1e-5,
        atol=1e-7,
    )


def test_without_any_valid_negative_the_loss_is_zero_and_flagged_invalid() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 6, D, seed=14), _x(3, 6, D, seed=15)
    only_first = jnp.asarray([[1] * 6, [0] * 6, [0] * 6], bool)
    nothing = jnp.zeros((3, 6), bool)
    for mask in (only_first, nothing):
        loss, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)
        assert float(loss) == 0.0
        assert float(m["cpc_valid"]) == 0.0 and float(m["cpc_n_anchors"]) == 0.0
        assert all(bool(jnp.isfinite(v)) for v in m.values())

        def loss_of(q, a, anc, mask=mask):
            return _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]

        g = jax.grad(loss_of, argnums=(0, 1, 2))(q, a, anc)
        assert all(not np.any(x) and bool(jnp.all(jnp.isfinite(x))) for x in _leaves(g))


def test_metrics_only_count_valid_anchors() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    seq = _x(4, 6, D, seed=16)
    anchors = jnp.concatenate([seq[:, 1:], seq[:, :1]], axis=1)  # aligned for k=1
    mask = _right_padded([6, 6, 4, 6], 6)
    # Garble the padded tail so that, were it counted, retrieval would fail.
    anchors = jnp.where(mask[..., None], anchors, _x(4, 6, D, seed=17))
    loss, m = _loss(q, a, anchors, seq, offset=1, tau=0.05, mask=mask)
    assert float(m["cpc_top1"]) == 1.0
    assert float(m["cpc_pos_sim"]) == pytest.approx(1.0, abs=1e-4)
    assert float(loss) < 1e-2


def test_step_axis_accepts_a_shared_or_a_per_step_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(2, 3, 7, D, seed=18), _x(2, 3, 7, D, seed=19)
    mask = _right_padded([7, 4, 6], 7)
    shared = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    per_step = _loss(
        q, a, anc, tgt, offset=1, tau=TAU, mask=jnp.broadcast_to(mask, (2, 3, 7))
    )[0]
    np.testing.assert_allclose(shared, per_step, rtol=1e-5, atol=1e-7)


def test_integer_masks_are_accepted() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=20), _x(3, 7, D, seed=21)
    mask = _right_padded([7, 5, 6], 7)
    np.testing.assert_allclose(
        _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0],
        _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask.astype(jnp.int32))[0],
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("shape", [(3,), (7, 3), (2, 7), (3, 8), (2, 3, 7)])
def test_wrongly_shaped_mask_is_rejected(shape) -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=22), _x(3, 7, D, seed=23)
    with pytest.raises(ValueError, match="mask"):
        contrastive_future_loss(
            q, a, anc, tgt, offset=1, tau=TAU, mask=jnp.ones(shape, bool)
        )


def test_jit_with_a_traced_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=24), _x(3, 7, D, seed=25)
    mask = _right_padded([7, 4, 6], 7)
    eager = contrastive_future_loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    jitted = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    np.testing.assert_allclose(eager, jitted, rtol=1e-5)


def test_max_anchors_stride_still_applies_with_a_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 13, D, seed=26), _x(3, 13, D, seed=27)
    mask = _right_padded([13, 13, 13], 13)
    _, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask, max_anchors_per_seq=4)
    assert float(m["cpc_n_anchors"]) == 3 * 4  # 12 positions, stride 3 -> 4 each


# --- backends --------------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [1, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_fabricpc_and_jax_heads_give_the_same_loss_and_gradients(
    offset, masked
) -> None:
    q, a = _heads(k=offset)
    anc, tgt = _x(3, 9, D, seed=28), _x(3, 9, D, seed=29)
    mask = _right_padded([9, 6, 8], 9) if masked else None

    def run(backend):
        def f(q, a, anc):
            return _loss(
                q, a, anc, tgt, offset=offset, tau=TAU, mask=mask, backend=backend
            )[0]

        return jax.value_and_grad(f, argnums=(0, 1, 2))(q, a, anc)

    (v_f, g_f), (v_j, g_j) = run("fabricpc"), run("jax")
    np.testing.assert_allclose(v_f, v_j, rtol=1e-5, atol=1e-7)
    for x, y in zip(_leaves(g_f), _leaves(g_j), strict=True):
        assert float(jnp.abs(y).max()) > 0
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def test_unknown_backend_is_rejected() -> None:
    q, a = _heads()
    with pytest.raises(ValueError, match="backend"):
        contrastive_future_loss(
            q, a, _x(3, 7, D), _x(3, 7, D, seed=1), offset=1, tau=TAU, backend="torch"
        )


@pytest.mark.parametrize("backend, expect_calls", [("fabricpc", True), ("jax", False)])
def test_the_backend_choice_reaches_the_heads(
    monkeypatch, backend, expect_calls
) -> None:
    from residual import cpc_heads

    calls = []
    real = cpc_heads._run_head

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(cpc_heads, "_run_head", spy)
    q, a = _heads()
    contrastive_future_loss(
        q,
        a,
        _x(3, 7, D, seed=40),
        _x(3, 7, D, seed=41),
        offset=1,
        tau=TAU,
        backend=backend,
    )
    assert bool(calls) is expect_calls
