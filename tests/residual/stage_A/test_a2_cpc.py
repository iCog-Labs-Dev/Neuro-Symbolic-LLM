"""Stage A2 contrastive-future term: config, reduction over sites, and the full path
through the frozen substrate and the A1 adapter (GPT-2 and Pythia fixtures)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import AdapterParams, ResidualAdapter, ResidualConfig
from residual.a2_losses import contrastive_future_loss
from residual.cpc_heads import a_key, init_cpc_heads, project, q_key
from stages.stage_A.a2 import CPCConfig, cpc_term, init_cpc_params, token_mask
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

FAMILIES = ("gpt2", "neox")
SEQ = 12
LAYERS = [5, 6]


# --- configuration ------------------------------------------------------------------------


def test_defaults_are_off() -> None:
    cfg = CPCConfig()
    assert not cfg.enabled and not cfg.needs_base_states
    assert (cfg.offsets, cfg.dim, cfg.tau, cfg.positive) == ((1,), 32, 0.1, "base")
    assert cfg.heads == "fabricpc"


def test_flags() -> None:
    assert CPCConfig(weight=0.1).enabled
    assert CPCConfig(weight=0.1).needs_base_states
    assert not CPCConfig(weight=0.1, positive="corrected").needs_base_states
    assert not CPCConfig(weight=0.0, positive="base").needs_base_states


def test_from_dict_roundtrip_and_unknown_key() -> None:
    cfg = CPCConfig.from_dict({"weight": 0.2, "offset": 2, "max_anchors_per_seq": None})
    assert cfg.weight == 0.2 and cfg.offsets == (2,) and cfg.max_anchors_per_seq is None
    assert CPCConfig.from_dict({}) == CPCConfig()
    with pytest.raises(ValueError, match="Unknown CPCConfig keys"):
        CPCConfig.from_dict({"wieght": 0.2})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": -0.1},
        {"weight": "1"},
        {"offsets": (0,)},
        {"offsets": (True,)},
        {"offsets": ()},
        {"offsets": (1, 1)},
        {"offsets": ("1",)},
        {"heads": "torch"},
        {"dim": 0},
        {"tau": 0},
        {"tau": -1.0},
        {"positive": "future"},
        {"max_anchors_per_seq": 0},
        {"a_init_noise": -0.1},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CPCConfig(**kwargs)


# --- initialisation ---------------------------------------------------------------------------


def test_disabled_term_has_no_parameters() -> None:
    assert init_cpc_params(CPCConfig(), 0, (5, 6), 32) == {}


def test_enabled_term_initialises_one_head_pair_per_site() -> None:
    cfg = CPCConfig(weight=0.1, offsets=(2,), dim=8)
    params = init_cpc_params(cfg, 0, (5, 6), 32)
    assert set(params) == {q_key(5), q_key(6), a_key(5, 2), a_key(6, 2)}
    assert params[q_key(5)]["W"].shape == (32, 8)


# --- reduction over sites ------------------------------------------------------------------------


def _states(seed: int, layers=(5, 6), d: int = 16):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return {
        layer: jax.random.normal(k, (3, 8, d))
        for layer, k in zip(layers, keys, strict=True)
    }


def test_term_is_the_mean_over_sites_of_the_pure_loss() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(2,), dim=8, tau=0.2, positive="base")
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)
    loss, metrics = cpc_term(params, config=cfg, corrected=corrected, base=base)

    per_site = [
        contrastive_future_loss(
            params[q_key(layer)],
            params[a_key(layer, 2)],
            corrected[layer],
            base[layer],
            offset=2,
            tau=0.2,
            max_anchors_per_seq=cfg.max_anchors_per_seq,
        )
        for layer in (5, 6)
    ]
    np.testing.assert_allclose(
        loss, np.mean([float(p[0]) for p in per_site]), rtol=1e-6
    )
    np.testing.assert_allclose(
        metrics["cpc_top1"],
        np.mean([float(p[1]["cpc_top1"]) for p in per_site]),
        rtol=1e-6,
    )
    assert set(metrics) == set(per_site[0][1])


def test_positive_base_reads_targets_from_base_and_corrected_from_anchors() -> None:
    corrected, base = _states(1), _states(2)
    cfg_b = CPCConfig(weight=1.0, dim=8, positive="base")
    cfg_c = CPCConfig(weight=1.0, dim=8, positive="corrected")
    params = init_cpc_params(cfg_b, 0, (5, 6), 16)

    l_base, _ = cpc_term(params, config=cfg_b, corrected=corrected, base=base)
    l_corr, _ = cpc_term(params, config=cfg_c, corrected=corrected, base=base)
    l_corr_nobase, _ = cpc_term(params, config=cfg_c, corrected=corrected)
    assert float(l_base) != float(l_corr)
    np.testing.assert_array_equal(l_corr, l_corr_nobase)  # base ignored when unused

    # With base == corrected, both modes coincide.
    same, _ = cpc_term(params, config=cfg_b, corrected=corrected, base=corrected)
    np.testing.assert_allclose(same, l_corr, rtol=1e-6)


def test_term_input_errors() -> None:
    cfg = CPCConfig(weight=1.0, dim=8)
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected = _states(1)
    with pytest.raises(ValueError, match="at least one site"):
        cpc_term(params, config=cfg, corrected={}, base={})
    with pytest.raises(ValueError, match="requires the base"):
        cpc_term(params, config=cfg, corrected=corrected)
    with pytest.raises(KeyError, match="layer 6"):
        cpc_term(params, config=cfg, corrected=corrected, base={5: corrected[5]})
    with pytest.raises(KeyError):  # head missing for a site
        cpc_term(
            {k: v for k, v in params.items() if k != q_key(6)},
            config=cfg,
            corrected=corrected,
            base=corrected,
        )


# --- through the frozen substrate and the A1 adapter ---------------------------------------------------


@pytest.fixture(scope="module", params=FAMILIES)
def substrate(request: pytest.FixtureRequest) -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate(request.param)[1]
    return sub


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(
        ResidualConfig(rank=4, layers=LAYERS), substrate.architecture
    )


@pytest.fixture(scope="module")
def input_ids() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)


def _with_random_b(params: AdapterParams, scale: float = 0.05) -> AdapterParams:
    rng = np.random.default_rng(1)
    return {
        k: {
            "A": v["A"],
            "B": jnp.asarray(scale * rng.normal(size=v["B"].shape), jnp.float32),
        }
        for k, v in params.items()
    }


def _loss(substrate, adapter, cfg, input_ids, adapter_params, heads, mask=None):
    run = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(adapter_params),
        intercept_layers=adapter.layers,
    )
    corrected = {
        layer: adapter.apply(adapter_params, run.hidden_state(layer), layer)
        for layer in adapter.layers
    }
    base = None
    if cfg.needs_base_states:
        base_run = substrate.run_with_interception(
            input_ids, modify_fn=identity_modify, intercept_layers=adapter.layers
        )
        base = {
            layer: jax.lax.stop_gradient(base_run.hidden_state(layer))
            for layer in adapter.layers
        }
    return cpc_term(heads, config=cfg, corrected=corrected, base=base, mask=mask)


def _setup(substrate, adapter, **cfg_kwargs):
    cfg = CPCConfig(weight=0.5, dim=8, **cfg_kwargs)
    heads = init_cpc_params(cfg, 0, adapter.layers, substrate.architecture.hidden_size)
    return cfg, heads


@pytest.mark.parametrize("positive", ["base", "corrected"])
def test_term_is_finite_and_leaves_the_base_frozen(
    substrate, adapter, input_ids, positive
) -> None:
    cfg, heads = _setup(substrate, adapter, positive=positive)
    loss, metrics = _loss(
        substrate, adapter, cfg, input_ids, _with_random_b(adapter.init_params()), heads
    )
    assert np.isfinite(loss) and float(loss) > 0
    assert float(metrics["cpc_valid"]) == 1.0
    assert substrate.params_unchanged()


def test_gradient_reaches_heads_and_adapter_only(substrate, adapter, input_ids) -> None:
    cfg, heads = _setup(substrate, adapter)
    ap = _with_random_b(adapter.init_params())

    def f(ap, heads):
        return _loss(substrate, adapter, cfg, input_ids, ap, heads)[0]

    g_ap, g_heads = jax.grad(f, argnums=(0, 1))(ap, heads)
    for name, g in g_heads.items():
        assert all(
            np.isfinite(x).all() and np.abs(x).max() > 0 for x in g.values()
        ), name
    for name, g in g_ap.items():
        assert np.abs(g["A"]).max() > 0 and np.abs(g["B"]).max() > 0, name
    assert substrate.params_unchanged()


def test_term_can_train_the_adapter_from_the_zero_initialised_residual(
    substrate, adapter, input_ids
) -> None:
    """At init B = 0 (identity). The anchor side must still send gradient into B."""
    cfg, heads = _setup(substrate, adapter)
    ap = adapter.init_params()

    def f(ap):
        return _loss(substrate, adapter, cfg, input_ids, ap, heads)[0]

    g = jax.grad(f)(ap)
    for name in ap:
        assert np.abs(g[name]["B"]).max() > 0, name
        assert not np.any(g[name]["A"]), name  # dL/dA = 0 while B = 0, as in A1


def test_corrected_target_matches_base_when_the_residual_is_zero(
    substrate, adapter, input_ids
) -> None:
    ap = adapter.init_params()
    cfg_b, heads = _setup(substrate, adapter, positive="base")
    cfg_c, _ = _setup(substrate, adapter, positive="corrected")
    lb, _ = _loss(substrate, adapter, cfg_b, input_ids, ap, heads)
    lc, _ = _loss(substrate, adapter, cfg_c, input_ids, ap, heads)
    np.testing.assert_allclose(lb, lc, rtol=1e-5)


def test_anchor_codes_are_causal(substrate, adapter, input_ids) -> None:
    """Changing tokens after t must not change the anchor code at t."""
    ap = _with_random_b(adapter.init_params())
    heads = init_cpc_heads(0, adapter.layers, substrate.architecture.hidden_size, 8)
    t0 = 7
    changed = input_ids.at[:, t0:].set((input_ids[:, t0:] + 1) % NUM_TOKENS)

    def codes(ids):
        run = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
        )
        return {
            layer: project(
                heads[q_key(layer)], adapter.apply(ap, run.hidden_state(layer), layer)
            )
            for layer in adapter.layers
        }

    a, b = codes(input_ids), codes(changed)
    for layer in adapter.layers:
        np.testing.assert_allclose(a[layer][:, :t0], b[layer][:, :t0], atol=1e-5)
        assert float(jnp.abs(a[layer][:, t0:] - b[layer][:, t0:]).max()) > 1e-4


def test_jit_matches_eager(substrate, adapter, input_ids) -> None:
    cfg, heads = _setup(substrate, adapter, offsets=(2,))
    ap = _with_random_b(adapter.init_params())

    def f(ap, heads):
        return _loss(substrate, adapter, cfg, input_ids, ap, heads)[0]

    e, ge = jax.value_and_grad(f, argnums=(0, 1))(ap, heads)
    j, gj = jax.jit(jax.value_and_grad(f, argnums=(0, 1)))(ap, heads)
    np.testing.assert_allclose(e, j, rtol=1e-4)
    for x, y in zip(
        jax.tree_util.tree_leaves(ge), jax.tree_util.tree_leaves(gj), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-3, atol=1e-6)


def test_heads_and_adapter_jointly_decrease_the_loss(
    substrate, adapter, input_ids
) -> None:
    cfg, heads = _setup(substrate, adapter)
    params = {"adapter": adapter.init_params(), "heads": heads}

    def f(p):
        return _loss(substrate, adapter, cfg, input_ids, p["adapter"], p["heads"])[0]

    step = jax.jit(jax.value_and_grad(f))
    opt = optax.adam(1e-2)
    state = opt.init(params)
    first, _ = step(params)
    for _ in range(40):
        _, g = step(params)
        upd, state = opt.update(g, state, params)
        params = optax.apply_updates(params, upd)
    last, _ = step(params)
    assert float(last) < float(first)
    assert substrate.params_unchanged()


# --- several offsets, the padding mask and the head backend (paper: k in {1, 2, 4}) ------------


def test_offsets_spellings() -> None:
    assert CPCConfig(offsets=2).offsets == (2,)
    assert CPCConfig(offsets=[1, 2, 4]).offsets == (1, 2, 4)
    assert CPCConfig.from_dict({"offsets": [1, 2, 4]}).offsets == (1, 2, 4)
    assert CPCConfig.from_dict({"offset": 2}).offsets == (2,)
    with pytest.raises(ValueError, match="either"):
        CPCConfig.from_dict({"offset": 2, "offsets": [1]})


def test_one_predictor_per_offset_each_with_its_own_initialisation() -> None:
    cfg = CPCConfig(weight=0.1, offsets=(1, 2, 4), dim=8)
    params = init_cpc_params(cfg, 0, (5, 6), 32)
    assert set(params) == {q_key(5), q_key(6)} | {
        a_key(layer, k) for layer in (5, 6) for k in (1, 2, 4)
    }
    assert not np.allclose(params[a_key(5, 1)]["M"], params[a_key(5, 2)]["M"])
    assert not np.allclose(params[a_key(5, 1)]["M"], params[a_key(6, 1)]["M"])


def test_term_is_the_mean_over_sites_and_offsets() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 2), dim=8, tau=0.2, positive="base")
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)
    loss, metrics = cpc_term(params, config=cfg, corrected=corrected, base=base)
    per_site = [
        contrastive_future_loss(
            params[q_key(layer)],
            params[a_key(layer, k)],
            corrected[layer],
            base[layer],
            offset=k,
            tau=0.2,
            max_anchors_per_seq=cfg.max_anchors_per_seq,
        )
        for layer in (5, 6)
        for k in (1, 2)
    ]
    assert len(per_site) == 4
    np.testing.assert_allclose(
        loss, np.mean([float(p[0]) for p in per_site]), rtol=1e-6
    )
    np.testing.assert_allclose(
        metrics["cpc_top1"],
        np.mean([float(p[1]["cpc_top1"]) for p in per_site]),
        rtol=1e-6,
    )
    # The offsets really differ: a longer horizon is a different problem.
    assert float(per_site[0][0]) != float(per_site[1][0])


def test_a_head_missing_for_one_offset_is_an_error() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 2), dim=8)
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    del params[a_key(6, 2)]
    with pytest.raises(KeyError):
        cpc_term(params, config=cfg, corrected=_states(1), base=_states(2))


def test_offset_longer_than_the_sequence_is_an_error() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 8), dim=8)  # SEQ in _states is 8
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    with pytest.raises(ValueError, match="offset"):
        cpc_term(params, config=cfg, corrected=_states(1), base=_states(2))


def test_fabricpc_and_jax_heads_give_the_same_term() -> None:
    kw = {"weight": 1.0, "offsets": (1, 2), "dim": 8, "tau": 0.2}
    cfg_f, cfg_j = CPCConfig(heads="fabricpc", **kw), CPCConfig(heads="jax", **kw)
    params = init_cpc_params(cfg_f, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)

    def f(cfg):
        return jax.value_and_grad(
            lambda p, c: cpc_term(p, config=cfg, corrected=c, base=base)[0],
            argnums=(0, 1),
        )(params, corrected)

    (v_f, g_f), (v_j, g_j) = f(cfg_f), f(cfg_j)
    np.testing.assert_allclose(v_f, v_j, rtol=1e-6)
    for x, y in zip(
        jax.tree_util.tree_leaves(g_f), jax.tree_util.tree_leaves(g_j), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def test_token_mask_from_labels() -> None:
    labels = jnp.asarray([[3, 4, -100, -100], [1, -100, 2, 5]])
    np.testing.assert_array_equal(
        token_mask(labels), [[True, True, False, False], [True, False, True, True]]
    )
    assert token_mask(None) is None
    np.testing.assert_array_equal(
        token_mask(jnp.asarray([[0, 7]]), ignore_index=7), [[True, False]]
    )


def test_term_ignores_padded_positions_at_every_site() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 2), dim=8, tau=0.2, positive="base")
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)
    mask = jnp.arange(8)[None, :] < jnp.asarray([8, 5, 6])[:, None]
    junk = jnp.where(
        mask[..., None],
        0.0,
        500.0 * jax.random.normal(jax.random.PRNGKey(9), (3, 8, 16)),
    )
    clean, _ = cpc_term(params, config=cfg, corrected=corrected, base=base, mask=mask)
    dirty, _ = cpc_term(
        params,
        config=cfg,
        corrected={k: v + junk for k, v in corrected.items()},
        base={k: v + junk for k, v in base.items()},
        mask=mask,
    )
    np.testing.assert_allclose(clean, dirty, rtol=1e-5)
    unmasked, _ = cpc_term(params, config=cfg, corrected=corrected, base=base)
    assert float(unmasked) != float(clean)


def _padded_labels(input_ids: jax.Array, length: int) -> jax.Array:
    return jnp.where(jnp.arange(input_ids.shape[1])[None, :] < length, input_ids, -100)


def test_right_padding_is_equivalent_to_truncating_the_batch(
    substrate, adapter, input_ids
) -> None:
    """A causal model's states at real positions do not see the padding after them."""
    cfg, heads = _setup(substrate, adapter, offsets=(1, 2), max_anchors_per_seq=None)
    ap = _with_random_b(adapter.init_params())
    length = 8
    mask = token_mask(_padded_labels(input_ids, length))
    padded, _ = _loss(substrate, adapter, cfg, input_ids, ap, heads, mask=mask)
    truncated, _ = _loss(substrate, adapter, cfg, input_ids[:, :length], ap, heads)
    np.testing.assert_allclose(padded, truncated, rtol=1e-4)
    full, _ = _loss(substrate, adapter, cfg, input_ids, ap, heads)
    assert abs(float(full) - float(padded)) > 1e-4


def test_every_offset_head_gets_a_gradient_through_the_model(
    substrate, adapter, input_ids
) -> None:
    cfg, heads = _setup(substrate, adapter, offsets=(1, 2))
    ap = _with_random_b(adapter.init_params())
    mask = token_mask(_padded_labels(input_ids, 10))

    def f(ap, heads):
        return _loss(substrate, adapter, cfg, input_ids, ap, heads, mask=mask)[0]

    g_ap, g_heads = jax.grad(f, argnums=(0, 1))(ap, heads)
    assert set(g_heads) == set(heads)
    for name, g in g_heads.items():
        assert all(
            np.abs(x).max() > 0 and np.isfinite(x).all() for x in g.values()
        ), name
    assert all(np.abs(g["B"]).max() > 0 for g in g_ap.values())
    assert substrate.params_unchanged()


def test_backends_agree_through_the_model_and_jit_matches_eager(
    substrate, adapter, input_ids
) -> None:
    ap = _with_random_b(adapter.init_params())
    mask = token_mask(_padded_labels(input_ids, 9))
    out = {}
    for backend in ("fabricpc", "jax"):
        cfg, heads = _setup(substrate, adapter, offsets=(1, 2), heads=backend)

        def f(ap, heads, cfg=cfg):
            return _loss(substrate, adapter, cfg, input_ids, ap, heads, mask=mask)[0]

        out[backend] = (
            jax.value_and_grad(f, argnums=(0, 1))(ap, heads),
            jax.jit(jax.value_and_grad(f, argnums=(0, 1)))(ap, heads),
        )
    (v_f, g_f), (vj_f, gj_f) = out["fabricpc"]
    (v_j, g_j), _ = out["jax"]
    np.testing.assert_allclose(v_f, v_j, rtol=1e-4)
    np.testing.assert_allclose(v_f, vj_f, rtol=1e-4)
    for x, y, z in zip(
        jax.tree_util.tree_leaves(g_f),
        jax.tree_util.tree_leaves(g_j),
        jax.tree_util.tree_leaves(gj_f),
        strict=True,
    ):
        np.testing.assert_allclose(x, y, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(x, z, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "heads_backend, expect_calls", [("fabricpc", True), ("jax", False)]
)
def test_the_heads_setting_selects_the_backend(
    monkeypatch, heads_backend, expect_calls
) -> None:
    from residual import cpc_heads

    calls = []
    real = cpc_heads._run_head
    monkeypatch.setattr(
        cpc_heads, "_run_head", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    cfg = CPCConfig(weight=1.0, dim=8, heads=heads_backend)
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    cpc_term(params, config=cfg, corrected=_states(1), base=_states(2))
    assert bool(calls) is expect_calls