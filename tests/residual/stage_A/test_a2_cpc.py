"""Stage A2 contrastive-future term: config, reduction over sites, and the full path
through the frozen substrate and the A1 adapter (GPT-2 and Pythia fixtures)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate, identity_modify
from losses.a2_loss import contrastive_future_loss
from residual import AdapterParams, ResidualAdapter, ResidualConfig
from residual.cpc_heads import a_key, init_cpc_heads, project, q_key
from stages.stage_A.a2 import CPCConfig, cpc_term, init_cpc_params, token_mask
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

FAMILIES = ("gpt2", "neox")
SEQ = 12
LAYERS = [5, 6]


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
    return cpc_term(
        heads,
        config=cfg,
        corrected=corrected,
        base=base,
        mask=mask,
        cpc_detach_query=False,
    )


def _setup(substrate, adapter, **cfg_kwargs):
    cfg = CPCConfig(weight=0.5, dim=8, **cfg_kwargs)
    heads = init_cpc_params(cfg, 0, adapter.layers, substrate.architecture.hidden_size)
    return cfg, heads


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


def test_fabricpc_and_jax_heads_give_the_same_term() -> None:
    cfg_f = CPCConfig(weight=1.0, offsets=(1, 2), dim=8, tau=0.2, heads="fabricpc")
    cfg_j = CPCConfig(weight=1.0, offsets=(1, 2), dim=8, tau=0.2, heads="jax")
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
