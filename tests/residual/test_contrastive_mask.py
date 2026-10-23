"""Padding mask and head backend of the contrastive future loss.

The paper's note: "Mask padding positions and positions where t + k lies outside
the sequence." A position is usable only if it is a real token; masked positions
are never an anchor, a positive or a negative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residual.a2_losses import contrastive_future_loss
from residual.cpc_heads import a_key, init_cpc_heads, predict_future, project, q_key

D = M = 12
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
    np.testing.assert_allclose(v0, v1, rtol=1e-6)
    for x, y in zip(_leaves(g0), _leaves(g1), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-7)
    m0 = _loss(q, a, anc, tgt, offset=1, tau=TAU)[1]
    m1 = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=ones)[1]
    for k in m0:
        np.testing.assert_allclose(m0[k], m1[k], rtol=1e-6)


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
    np.testing.assert_allclose(v0, v1, rtol=1e-6)
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
    np.testing.assert_allclose(f(anc, tgt, mask)[0], f(anc2, tgt2, mask)[0], rtol=1e-6)
    # ... but without the mask it does.
    assert abs(float(f(anc, tgt, None)[0]) - float(f(anc2, tgt2, None)[0])) > 1e-3


def test_a_fully_masked_sequence_equals_dropping_it() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=9), _x(3, 7, D, seed=10)
    mask = jnp.asarray([[1] * 7, [1] * 7, [0] * 7], bool)
    masked, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)
    dropped, md = _loss(q, a, anc[:2], tgt[:2], offset=1, tau=TAU)
    np.testing.assert_allclose(masked, dropped, rtol=1e-6)
    for k in ("cpc_top1", "cpc_pos_sim", "cpc_neg_sim", "cpc_n_anchors"):
        np.testing.assert_allclose(m[k], md[k], rtol=1e-6)


def test_padded_negatives_are_not_used() -> None:
    """Moving a padded sequence's content must not change the other sequences' loss."""
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=11), _x(3, 7, D, seed=12)
    mask = _right_padded([7, 7, 3], 7)
    base = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    tgt_p = tgt.at[2, 3:].set(_x(4, D, seed=13) * 10.0)  # only the padded tail
    np.testing.assert_allclose(
        base, _loss(q, a, anc, tgt_p, offset=1, tau=TAU, mask=mask)[0], rtol=1e-6
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
    np.testing.assert_allclose(shared, per_step, rtol=1e-6)


def test_integer_masks_are_accepted() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=20), _x(3, 7, D, seed=21)
    mask = _right_padded([7, 5, 6], 7)
    np.testing.assert_allclose(
        _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0],
        _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask.astype(jnp.int32))[0],
        rtol=1e-6,
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
    np.testing.assert_allclose(v_f, v_j, rtol=1e-6)
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
    monkeypatch.setattr(
        cpc_heads, "_run_head", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
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