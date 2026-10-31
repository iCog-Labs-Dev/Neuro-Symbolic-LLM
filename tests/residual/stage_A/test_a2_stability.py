"""Stage A2 stability term (anchor loss + HEP nudge): config, reduction over sites and the
full path through the frozen substrate and the A1 adapter (GPT-2 and Pythia fixtures)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import AdapterParams, ResidualAdapter, ResidualConfig
from residual.a2_losses import anchor_loss
from residual.hep import hep_key
from stages.stage_A.a2 import (
    HEPConfig,
    StabLossConfig,
    init_hep_params,
    stability_term,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

FAMILIES = ("gpt2", "neox")
SEQ = 12
LAYERS = [5, 6]


def cfg_of(hep_kind=None, **kw) -> StabLossConfig:
    hep = HEPConfig(kind=hep_kind, rank=4, alpha=1.0)
    return StabLossConfig(weight=1.0, hep=hep, **kw)


# --- configuration --------------------------------------------------------------------------


def test_defaults_are_off() -> None:
    cfg = StabLossConfig()
    assert not cfg.enabled and not cfg.needs_base_states and not cfg.needs_output_error
    assert cfg.hep.kind is None and cfg.w_anch == 1.0 and cfg.w_hep == 0.0


def test_flags() -> None:
    assert cfg_of().enabled and cfg_of().needs_base_states
    assert not cfg_of().needs_output_error
    assert cfg_of("tied").needs_output_error
    assert not cfg_of(w_anch=0.0).needs_base_states
    assert not StabLossConfig(
        weight=0.0, hep=HEPConfig(kind="dense")
    ).needs_output_error


def test_from_dict_nested_and_unknown_keys() -> None:
    cfg = StabLossConfig.from_dict(
        {
            "weight": 0.3,
            "w_hep": 0.1,
            "hep": {"kind": "lowrank", "rank": 8, "alpha": 0.5},
        }
    )
    assert cfg.hep == HEPConfig(kind="lowrank", rank=8, alpha=0.5) and cfg.w_hep == 0.1
    assert StabLossConfig.from_dict({}) == StabLossConfig()
    assert StabLossConfig.from_dict({"hep": None}) == StabLossConfig()
    with pytest.raises(ValueError, match="Unknown StabLossConfig keys"):
        StabLossConfig.from_dict({"w_stepp": 1.0})
    with pytest.raises(ValueError, match="Unknown HEPConfig keys"):
        StabLossConfig.from_dict({"hep": {"gain": 1.0}})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": -1.0},
        {"w_anch": -0.1},
        {"w_hep": -0.1},
        {"weight": True},
        {"normalize": "l2"},
        {"w_hep": 0.1},  # no hep.kind
    ],
)
def test_invalid_stability_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        StabLossConfig(**kwargs)


@pytest.mark.parametrize("kind", ["tied", "fixed_random"])
def test_hep_loss_needs_a_trainable_highway(kind: str) -> None:
    with pytest.raises(ValueError, match="no gradient"):
        StabLossConfig(w_hep=0.1, hep=HEPConfig(kind=kind))
    StabLossConfig(w_hep=0.0, hep=HEPConfig(kind=kind))  # metric only: fine


@pytest.mark.parametrize("kind", ["dense", "lowrank"])
def test_hep_loss_allowed_with_trainable_highway(kind: str) -> None:
    assert StabLossConfig(w_hep=0.1, hep=HEPConfig(kind=kind)).w_hep == 0.1


@pytest.mark.parametrize(
    "kwargs", [{"kind": "sparse"}, {"rank": 0}, {"rank": True}, {"alpha": -1.0}]
)
def test_invalid_hep_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HEPConfig(**kwargs)


# --- initialisation --------------------------------------------------------------------------


def test_init_hep_params_only_for_enabled_trainable_highways() -> None:
    assert init_hep_params(StabLossConfig(), 0, (5, 6), 64, 32) == {}
    assert init_hep_params(cfg_of("tied"), 0, (5, 6), 64, 32) == {}
    assert init_hep_params(cfg_of("fixed_random"), 0, (5, 6), 64, 32) == {}
    off = StabLossConfig(weight=0.0, hep=HEPConfig(kind="dense"))
    assert init_hep_params(off, 0, (5, 6), 64, 32) == {}
    dense = init_hep_params(cfg_of("dense"), 0, (5, 6), 64, 32)
    assert set(dense) == {hep_key(5), hep_key(6)} and dense["hep_5"]["V"].shape == (
        64,
        32,
    )
    low = init_hep_params(cfg_of("lowrank"), 0, (5,), 64, 32)
    assert low["hep_5"]["U"].shape == (64, 4)


# --- reduction over sites ----------------------------------------------------------------------


def _states(seed: int, layers=(5, 6), d: int = 16, shape=(3, 8)):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return {
        layer: jax.random.normal(k, (*shape, d))
        for layer, k in zip(layers, keys, strict=True)
    }


def test_anchor_term_is_the_weighted_mean_over_sites() -> None:
    cfg = cfg_of(w_anch=2.0)
    corrected, base = _states(1), _states(2)
    loss, m = stability_term({}, config=cfg, corrected=corrected, base=base)
    per = [float(anchor_loss(corrected[layer], base[layer])[0]) for layer in (5, 6)]
    np.testing.assert_allclose(loss, 2.0 * np.mean(per), rtol=1e-6)
    np.testing.assert_allclose(m["stab_anch"], np.mean(per), rtol=1e-6)
    assert {"anch_dist", "anch_ratio"} <= set(m) and "stab_hep" not in m


def test_w_anch_zero_skips_the_anchor_and_needs_no_base() -> None:
    loss, m = stability_term({}, config=cfg_of(w_anch=0.0), corrected=_states(1))
    assert float(loss) == 0.0 and "stab_anch" not in m


def test_hep_metric_is_logged_but_not_added_when_w_hep_is_zero() -> None:
    corrected, base = _states(1), _states(2)
    logits = jax.random.normal(jax.random.PRNGKey(3), (3, 8, 20))
    labels = jax.random.randint(jax.random.PRNGKey(4), (3, 8), 0, 20)
    tied = jax.random.normal(jax.random.PRNGKey(5), (20, 16))
    kw = {
        "corrected": corrected,
        "base": base,
        "logits": logits,
        "labels": labels,
        "tied": tied,
    }
    with_hep, m = stability_term({}, config=cfg_of("tied"), **kw)
    anchor_only, _ = stability_term({}, config=cfg_of(), corrected=corrected, base=base)
    np.testing.assert_allclose(with_hep, anchor_only, rtol=1e-6)
    assert float(m["hep_nudge_norm"]) > 0 and "stab_hep" in m and "hep_nudge_ratio" in m


def test_term_input_errors() -> None:
    corrected = _states(1)
    logits = jnp.zeros((3, 8, 20))
    labels = jnp.zeros((3, 8), jnp.int32)
    with pytest.raises(ValueError, match="at least one site"):
        stability_term({}, config=cfg_of(), corrected={}, base={})
    with pytest.raises(ValueError, match="requires the base"):
        stability_term({}, config=cfg_of(), corrected=corrected)
    with pytest.raises(KeyError, match="layer 6"):
        stability_term({}, config=cfg_of(), corrected=corrected, base={5: corrected[5]})
    with pytest.raises(ValueError, match="logits and the labels"):
        stability_term({}, config=cfg_of("tied"), corrected=corrected, base=corrected)
    with pytest.raises(ValueError, match="tied"):
        stability_term(
            {},
            config=cfg_of("tied"),
            corrected=corrected,
            base=corrected,
            logits=logits,
            labels=labels,
        )
    with pytest.raises(KeyError, match="hep_5"):  # trainable kind but no parameters
        stability_term(
            {},
            config=cfg_of("dense"),
            corrected=corrected,
            base=corrected,
            logits=logits,
            labels=labels,
        )


# --- through the frozen substrate and the A1 adapter ----------------------------------------------


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


def _term(
    substrate,
    adapter,
    cfg,
    input_ids,
    ap,
    hep_params,
    *,
    labels=None,
    tied=None,
    seed=0,
    mask=None,
):
    run = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
    )
    corrected = {
        layer: adapter.apply(ap, run.hidden_state(layer), layer)
        for layer in adapter.layers
    }
    base_run = substrate.run_with_interception(
        input_ids, modify_fn=identity_modify, intercept_layers=adapter.layers
    )
    base = {
        layer: jax.lax.stop_gradient(base_run.hidden_state(layer))
        for layer in adapter.layers
    }
    return stability_term(
        hep_params,
        config=cfg,
        corrected=corrected,
        base=base,
        logits=run.logits,
        labels=input_ids if labels is None else labels,
        seed=seed,
        tied=tied,
        mask=mask,
    )


def _setup(substrate, adapter, kind, **kw):
    cfg = cfg_of(kind, **kw)
    d = substrate.architecture.hidden_size
    return cfg, init_hep_params(cfg, 0, adapter.layers, NUM_TOKENS, d)


def _tied(substrate) -> jax.Array:
    d = substrate.architecture.hidden_size
    return jax.random.normal(jax.random.PRNGKey(9), (NUM_TOKENS, d)) / np.sqrt(
        NUM_TOKENS
    )


def test_anchor_is_zero_at_init_and_positive_after_the_residual_moves(
    substrate, adapter, input_ids
) -> None:
    cfg, hp = _setup(substrate, adapter, None)
    at_init, m0 = _term(substrate, adapter, cfg, input_ids, adapter.init_params(), hp)
    assert float(at_init) < 1e-10 and float(m0["anch_ratio"]) < 1e-3
    moved, m1 = _term(
        substrate, adapter, cfg, input_ids, _with_random_b(adapter.init_params()), hp
    )
    assert float(moved) > 1e-6 and float(m1["anch_ratio"]) > float(m0["anch_ratio"])
    assert substrate.params_unchanged()


def test_anchor_gradient_pulls_the_residual_back(substrate, adapter, input_ids) -> None:
    cfg, hp = _setup(substrate, adapter, None)
    ap = _with_random_b(adapter.init_params())

    def f(ap):
        return _term(substrate, adapter, cfg, input_ids, ap, hp)[0]

    g = jax.grad(f)(ap)
    for name in ap:
        assert np.abs(g[name]["A"]).max() > 0 and np.abs(g[name]["B"]).max() > 0, name

    step = jax.jit(jax.value_and_grad(f))
    opt = optax.adam(1e-2)
    state, p = opt.init(ap), ap
    first, _ = step(p)
    for _ in range(40):
        _, grads = step(p)
        upd, state = opt.update(grads, state, p)
        p = optax.apply_updates(p, upd)
    last, _ = step(p)
    assert float(last) < 0.5 * float(first)
    assert substrate.params_unchanged()


@pytest.mark.parametrize("kind", ["dense", "lowrank"])
def test_hep_term_trains_the_highway_and_never_the_adapter(
    substrate, adapter, input_ids, kind
) -> None:
    cfg, hp = _setup(substrate, adapter, kind, w_anch=0.0, w_hep=1.0)
    ap = _with_random_b(adapter.init_params())

    def f(ap, hp):
        return _term(substrate, adapter, cfg, input_ids, ap, hp)[0]

    loss = f(ap, hp)
    assert np.isfinite(loss) and float(loss) > 0
    g_ap, g_hp = jax.grad(f, argnums=(0, 1))(ap, hp)
    for name, g in g_ap.items():  # eps and the reference are stop-gradient
        assert not np.any(g["A"]) and not np.any(g["B"]), name
    for name, g in g_hp.items():
        assert all(
            np.abs(x).max() > 0 and np.isfinite(x).all() for x in g.values()
        ), name


def test_adding_the_hep_term_leaves_the_adapter_gradient_unchanged(
    substrate, adapter, input_ids
) -> None:
    ap = _with_random_b(adapter.init_params())
    cfg_a, hp_a = _setup(substrate, adapter, None)
    cfg_b, hp_b = _setup(substrate, adapter, "dense", w_hep=1.0)

    ga = jax.grad(lambda ap: _term(substrate, adapter, cfg_a, input_ids, ap, hp_a)[0])(
        ap
    )
    gb = jax.grad(lambda ap: _term(substrate, adapter, cfg_b, input_ids, ap, hp_b)[0])(
        ap
    )
    for x, y in zip(
        jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-8)


def test_hep_loss_shrinks_the_highway_by_gradient_descent(
    substrate, adapter, input_ids
) -> None:
    cfg, hp = _setup(substrate, adapter, "dense", w_anch=0.0, w_hep=1.0)
    ap = adapter.init_params()
    f = jax.jit(
        jax.value_and_grad(
            lambda hp: _term(substrate, adapter, cfg, input_ids, ap, hp)[0]
        )
    )
    first, _ = f(hp)
    for _ in range(30):
        _, g = f(hp)
        hp = jax.tree_util.tree_map(lambda p, d: p - 0.5 * d, hp, g)
    last, _ = f(hp)
    assert float(last) < 0.2 * float(first)


@pytest.mark.parametrize("kind", ["lowrank", "fixed_random", "tied", "dense"])
def test_every_highway_kind_gives_finite_values_and_metrics(
    substrate, adapter, input_ids, kind
) -> None:
    cfg, hp = _setup(substrate, adapter, kind)
    tied = _tied(substrate) if kind == "tied" else None
    loss, m = _term(
        substrate,
        adapter,
        cfg,
        input_ids,
        _with_random_b(adapter.init_params()),
        hp,
        tied=tied,
    )
    assert np.isfinite(loss)
    assert float(m["hep_nudge_norm"]) > 0 and float(m["hep_nudge_ratio"]) > 0
    assert (len(hp) > 0) == (
        kind in ("dense", "lowrank")
    )  # fixed highways are not parameters
    assert substrate.params_unchanged()


def test_fixed_random_highway_depends_on_the_seed(
    substrate, adapter, input_ids
) -> None:
    cfg, hp = _setup(substrate, adapter, "fixed_random")
    ap = adapter.init_params()
    a = _term(substrate, adapter, cfg, input_ids, ap, hp, seed=0)[1]["hep_nudge_norm"]
    a2 = _term(substrate, adapter, cfg, input_ids, ap, hp, seed=0)[1]["hep_nudge_norm"]
    b = _term(substrate, adapter, cfg, input_ids, ap, hp, seed=1)[1]["hep_nudge_norm"]
    assert float(a) == float(a2) and float(a) != float(b)


def test_ignored_labels_are_excluded_from_the_hep_term(
    substrate, adapter, input_ids
) -> None:
    cfg, hp = _setup(substrate, adapter, "tied")
    tied, ap = _tied(substrate), adapter.init_params()
    labels = input_ids.at[:, 6:].set(
        -100
    )  # the model is not asked to predict tokens 6+
    full = _term(substrate, adapter, cfg, input_ids, ap, hp, tied=tied)[1][
        "hep_nudge_norm"
    ]
    masked = _term(
        substrate, adapter, cfg, input_ids, ap, hp, labels=labels, tied=tied
    )[1]["hep_nudge_norm"]
    assert float(full) != float(masked) and np.isfinite(masked) and float(masked) > 0
    # Fully masked: no prediction errors at all.
    none = _term(
        substrate,
        adapter,
        cfg,
        input_ids,
        ap,
        hp,
        labels=jnp.full_like(input_ids, -100),
        tied=tied,
    )[1]
    assert float(none["hep_nudge_norm"]) < 1e-3


def test_jit_matches_eager(substrate, adapter, input_ids) -> None:
    cfg, hp = _setup(substrate, adapter, "dense", w_hep=0.5)
    ap = _with_random_b(adapter.init_params())

    def f(ap, hp):
        return _term(substrate, adapter, cfg, input_ids, ap, hp)[0]

    e, ge = jax.value_and_grad(f, argnums=(0, 1))(ap, hp)
    j, gj = jax.jit(jax.value_and_grad(f, argnums=(0, 1)))(ap, hp)
    np.testing.assert_allclose(e, j, rtol=1e-4)
    for x, y in zip(
        jax.tree_util.tree_leaves(ge), jax.tree_util.tree_leaves(gj), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-3, atol=1e-6)


# --- backend and padding mask --------------------------------------------------------------


def test_backend_setting_is_validated_and_defaults_to_fabricpc() -> None:
    assert StabLossConfig().backend == "fabricpc"
    assert StabLossConfig.from_dict({"backend": "jax"}).backend == "jax"
    for bad in ("torch", "", None):
        with pytest.raises(ValueError, match="backend"):
            StabLossConfig(backend=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", [None, "dense", "lowrank", "fixed_random", "tied"])
def test_term_agrees_across_backends(kind) -> None:
    d, v = 16, 24
    corrected, base = _states(1), _states(2)
    logits = jax.random.normal(jax.random.PRNGKey(3), (3, 8, v))
    labels = jax.random.randint(jax.random.PRNGKey(4), (3, 8), 0, v).at[1, 5:].set(-100)
    mask = jnp.arange(8)[None, :] < jnp.asarray([8, 5, 6])[:, None]
    tied = jax.random.normal(jax.random.PRNGKey(5), (v, d)) / np.sqrt(v)
    out = {}
    for backend in ("fabricpc", "jax"):
        cfg = StabLossConfig(
            weight=1.0,
            w_hep=0.5 if kind in ("dense", "lowrank") else 0.0,
            backend=backend,
            hep=HEPConfig(kind=kind, rank=4, alpha=0.7),
        )
        hp = init_hep_params(cfg, 0, (5, 6), v, d)

        def f(c, hp, cfg=cfg):
            return stability_term(
                hp,
                config=cfg,
                corrected=c,
                base=base,
                logits=logits,
                labels=labels,
                tied=tied,
                mask=mask,
            )

        (val, m), g = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(corrected, hp)
        out[backend] = (val, m, g)
    (v_f, m_f, g_f), (v_j, m_j, g_j) = out["fabricpc"], out["jax"]
    np.testing.assert_allclose(v_f, v_j, rtol=1e-5)
    assert set(m_f) == set(m_j)
    for k in m_j:
        np.testing.assert_allclose(m_f[k], m_j[k], rtol=1e-4, atol=1e-6)
    for x, y in zip(
        jax.tree_util.tree_leaves(g_f), jax.tree_util.tree_leaves(g_j), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_padding_is_ignored_by_the_anchor_term() -> None:
    cfg = StabLossConfig(weight=1.0)
    corrected, base = _states(1), _states(2)
    mask = jnp.arange(8)[None, :] < jnp.asarray([8, 5, 6])[:, None]
    junk = jnp.where(
        mask[..., None],
        0.0,
        100.0 * jax.random.normal(jax.random.PRNGKey(9), (3, 8, 16)),
    )
    clean, _ = stability_term({}, config=cfg, corrected=corrected, base=base, mask=mask)
    dirty, _ = stability_term(
        {},
        config=cfg,
        corrected={k: v + junk for k, v in corrected.items()},
        base=base,
        mask=mask,
    )
    np.testing.assert_allclose(clean, dirty, rtol=1e-5)
    unmasked, _ = stability_term({}, config=cfg, corrected=corrected, base=base)
    assert float(unmasked) != float(clean)


def test_backends_agree_through_the_model_and_jit_matches_eager(
    substrate, adapter, input_ids
) -> None:
    ap = _with_random_b(adapter.init_params())
    labels = input_ids.at[:, 9:].set(-100)
    mask = labels != -100
    out = {}
    for backend in ("fabricpc", "jax"):
        cfg, hp = _setup(substrate, adapter, "dense", w_hep=0.5, backend=backend)

        def f(ap, hp, cfg=cfg):
            return _term(
                substrate, adapter, cfg, input_ids, ap, hp, labels=labels, mask=mask
            )[0]

        out[backend] = (
            jax.value_and_grad(f, argnums=(0, 1))(ap, hp),
            jax.jit(jax.value_and_grad(f, argnums=(0, 1)))(ap, hp),
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
    assert substrate.params_unchanged()


@pytest.mark.parametrize("backend, expect", [("fabricpc", True), ("jax", False)])
def test_the_config_backend_reaches_the_energies(monkeypatch, backend, expect) -> None:
    from residual import fpc_energy

    calls = []
    real = fpc_energy._per_token_energy

    def spy(*a, **k):
        calls.append(a[-1])
        return real(*a, **k)

    monkeypatch.setattr(fpc_energy, "_per_token_energy", spy)
    cfg = StabLossConfig(weight=1.0, backend=backend, hep=HEPConfig(kind="dense"))
    d, v = 16, 24
    hp = init_hep_params(cfg, 0, (5, 6), v, d)
    stability_term(
        hp,
        config=cfg,
        corrected=_states(1),
        base=_states(2),
        logits=jax.random.normal(jax.random.PRNGKey(3), (3, 8, v)),
        labels=jax.random.randint(jax.random.PRNGKey(4), (3, 8), 0, v),
    )
    assert bool(calls) is expect
    assert (set(calls) == {"anch", "hep"}) is expect
