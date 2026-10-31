"""The four stability losses of A2: anchor, HEP nudge, residual smoothness and step loss."""

from __future__ import annotations

from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from residual import ResidualAdapter, ResidualConfig
from stages.stage_A.a2 import (
    A2Config,
    HEPConfig,
    StabLossConfig,
    a2_audit,
    a2_objective,
    adapter_params,
    apply_arm,
    audit_ok,
    init_a2_params,
    residual_smoothness_loss,
    stability_term,
    step_loss,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

SEQ = 12
LAYERS = [5, 6]
BACKENDS = ("fabricpc", "jax")


def rand(*shape: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32)


# --- residual_smoothness_loss ---------------------------------------------------------------


def reference_smoothness(r: np.ndarray, h: np.ndarray, mask=None):
    def adj(x):
        return np.sum((x[:, 1:] - x[:, :-1]) ** 2, axis=-1)

    w = np.ones(adj(r).shape) if mask is None else (mask[:, 1:] & mask[:, :-1])
    res = (adj(r) * w).sum() / w.sum()
    inp = (adj(h) * w).sum() / w.sum()
    return res, inp


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_mask", [False, True])
def test_smoothness_matches_a_numpy_reference(backend: str, use_mask: bool) -> None:
    r, h = rand(2, 7, 8, seed=1), rand(2, 7, 8, seed=2)
    mask = np.array([[1, 1, 1, 1, 0, 0, 0], [1] * 7], bool) if use_mask else None
    res, inp = reference_smoothness(np.asarray(r), np.asarray(h), mask)
    jmask = None if mask is None else jnp.asarray(mask)
    none, m = residual_smoothness_loss(
        r, h, normalize="none", mask=jmask, backend=backend
    )
    rel, _ = residual_smoothness_loss(
        r, h, normalize="relative", mask=jmask, backend=backend
    )
    assert float(none) == pytest.approx(res, rel=1e-5)
    assert float(rel) == pytest.approx(res / inp, rel=1e-4)
    assert float(m["smooth_res"]) == pytest.approx(res, rel=1e-5)
    assert float(m["smooth_ratio"]) == pytest.approx(res / inp, rel=1e-4)


def test_smoothness_backends_agree_in_value_and_gradient() -> None:
    r, h = rand(2, 6, 8, seed=3), rand(2, 6, 8, seed=4)
    mask = jnp.asarray([[1, 1, 1, 0, 0, 0], [1] * 6], bool)
    out = {}
    for backend in BACKENDS:

        def f(x, backend=backend):
            return residual_smoothness_loss(x, h, mask=mask, backend=backend)[0]

        out[backend] = (f(r), jax.grad(f)(r))
    assert float(out["fabricpc"][0]) == pytest.approx(float(out["jax"][0]), rel=1e-5)
    np.testing.assert_allclose(out["fabricpc"][1], out["jax"][1], rtol=1e-4, atol=1e-6)


def test_a_constant_correction_is_perfectly_smooth() -> None:
    r = jnp.broadcast_to(rand(2, 1, 8), (2, 6, 8))
    loss, _ = residual_smoothness_loss(r, rand(2, 6, 8, seed=5))
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


def test_smoothness_is_scale_free_when_relative() -> None:
    r, h = rand(2, 6, 8, seed=6), rand(2, 6, 8, seed=7)
    a = residual_smoothness_loss(r, h)[0]
    b = residual_smoothness_loss(3.0 * r, 3.0 * h)[0]
    assert float(a) == pytest.approx(float(b), rel=1e-4)
    c = residual_smoothness_loss(r, h, normalize="none")[0]
    d = residual_smoothness_loss(3.0 * r, 3.0 * h, normalize="none")[0]
    assert float(d) == pytest.approx(9.0 * float(c), rel=1e-4)


def test_the_input_jitter_is_a_constant_for_the_gradient() -> None:
    r, h = rand(2, 6, 8, seed=8), rand(2, 6, 8, seed=9)
    g = jax.grad(lambda x: residual_smoothness_loss(r, x)[0])(h)
    assert float(jnp.max(jnp.abs(g))) == 0.0
    assert (
        float(
            jnp.max(jnp.abs(jax.grad(lambda x: residual_smoothness_loss(x, h)[0])(r)))
        )
        > 0
    )


def test_masked_out_tokens_have_no_influence() -> None:
    r, h = rand(2, 6, 8, seed=10), rand(2, 6, 8, seed=11)
    mask = jnp.asarray([[1, 1, 1, 0, 0, 0], [1] * 6], bool)
    g = jax.grad(lambda x: residual_smoothness_loss(x, h, mask=mask)[0])(r)
    assert float(jnp.max(jnp.abs(g[0, 4:]))) == 0.0
    assert float(jnp.max(jnp.abs(g[0, :3]))) > 0.0


def test_a_single_token_has_no_pairs() -> None:
    r = rand(2, 1, 8)
    loss, m = residual_smoothness_loss(r, rand(2, 1, 8, seed=1))
    assert float(loss) == 0.0 and float(m["smooth_ratio"]) == 0.0
    g = jax.grad(lambda x: residual_smoothness_loss(x, r)[0])(r)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_smoothness_input_errors() -> None:
    with pytest.raises(ValueError, match="normalize"):
        residual_smoothness_loss(rand(2, 4, 8), rand(2, 4, 8), normalize="mean")
    with pytest.raises(ValueError, match="share a shape"):
        residual_smoothness_loss(rand(2, 4, 8), rand(2, 5, 8))
    with pytest.raises(ValueError, match="backend"):
        residual_smoothness_loss(rand(2, 4, 8), rand(2, 4, 8), backend="torch")


# --- step_loss ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_step_loss_matches_a_numpy_reference(backend: str) -> None:
    z = rand(4, 2, 5, 8, seed=12)
    zn = np.asarray(z)
    raw = (np.sum((zn[1:] - zn[:-1]) ** 2, axis=-1).sum(axis=0)).mean()
    scale = np.mean(np.sum(zn[0] ** 2, axis=-1))
    none, m = step_loss(z, normalize="none", backend=backend)
    rel, _ = step_loss(z, normalize="relative", backend=backend)
    assert float(none) == pytest.approx(raw, rel=1e-5)
    assert float(rel) == pytest.approx(raw / scale, rel=1e-4)
    assert float(m["step_sq"]) == pytest.approx(raw, rel=1e-5)


def test_step_loss_backends_agree_in_value_and_gradient() -> None:
    z = rand(3, 2, 5, 8, seed=13)
    mask = jnp.asarray([[1, 1, 1, 0, 0], [1] * 5], bool)
    out = {}
    for backend in BACKENDS:

        def f(x, backend=backend):
            return step_loss(x, mask=mask, backend=backend)[0]

        out[backend] = (f(z), jax.grad(f)(z))
    assert float(out["fabricpc"][0]) == pytest.approx(float(out["jax"][0]), rel=1e-5)
    np.testing.assert_allclose(out["fabricpc"][1], out["jax"][1], rtol=1e-4, atol=1e-6)


def test_step_loss_is_exactly_zero_without_settling() -> None:
    z = rand(1, 2, 5, 8)
    loss, m = step_loss(z)
    assert float(loss) == 0.0 and float(m["step_sq"]) == 0.0
    g = jax.grad(lambda x: step_loss(x)[0])(z)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_step_loss_is_zero_for_a_fixed_point_and_grows_with_the_steps() -> None:
    z0 = rand(1, 2, 5, 8, seed=14)
    fixed = jnp.concatenate([z0, z0, z0])
    assert float(step_loss(fixed)[0]) == 0.0
    moving = jnp.concatenate([z0, z0 + 0.1, z0 + 0.2])
    assert float(step_loss(moving, normalize="none")[0]) == pytest.approx(
        2 * 8 * 0.01, rel=1e-4
    )


def test_step_loss_input_errors() -> None:
    with pytest.raises(ValueError, match="trajectory"):
        step_loss(rand(2, 5, 8))
    with pytest.raises(ValueError, match="normalize"):
        step_loss(rand(2, 2, 5, 8), normalize="x")


# --- configuration ------------------------------------------------------------------------------


def test_the_new_weights_default_to_off_and_are_validated() -> None:
    cfg = StabLossConfig()
    assert (cfg.w_smooth, cfg.w_step) == (0.0, 0.0)
    with pytest.raises(ValueError, match="w_smooth"):
        StabLossConfig(w_smooth=-0.1)
    with pytest.raises(ValueError, match="w_smooth"):
        StabLossConfig(w_smooth=True)
    assert StabLossConfig(w_smooth=0.5).w_smooth == 0.5

    with pytest.raises(ValueError, match="w_step"):
        StabLossConfig(w_step=-0.1)
    with pytest.raises(ValueError, match="w_step"):
        StabLossConfig(w_step=True)
    assert StabLossConfig(w_step=0.5).w_step == 0.5


def test_from_dict_reads_the_new_weights_and_round_trips() -> None:
    cfg = A2Config.from_dict({"stab": {"weight": 0.2, "w_smooth": 0.3, "w_step": 1.5}})
    assert (cfg.stab.w_smooth, cfg.stab.w_step) == (0.3, 1.5)
    assert A2Config.from_dict(asdict(cfg)) == cfg


# --- the stage term ---------------------------------------------------------------------------------


def term_inputs(seed: int = 0):
    sites = {5: rand(2, 6, 8, seed=seed), 6: rand(2, 6, 8, seed=seed + 1)}
    corrected = {
        k: v + 0.1 * rand(2, 6, 8, seed=seed + 5 + k) for k, v in sites.items()
    }
    base = dict(sites)
    return sites, corrected, base


def test_the_term_is_the_weighted_sum_of_all_four_parts() -> None:
    sites, corrected, base = term_inputs()
    cfg = StabLossConfig(weight=1.0, w_anch=0.7, w_smooth=0.4, w_step=2.0)
    total, m = stability_term(
        {}, config=cfg, corrected=corrected, base=base, sites=sites
    )
    parts = {
        "anch": np.mean([float(jnp.sum((corrected[k] - base[k]) ** 2) / jnp.sum(base[k] ** 2)) for k in corrected]),
        "smooth": np.mean([float(residual_smoothness_loss(corrected[k] - sites[k], sites[k])[0]) for k in corrected]),
    }  # fmt: skip
    assert float(m["stab_smooth"]) == pytest.approx(parts["smooth"], rel=1e-4)
    assert float(m["stab_step"]) == 0.0
    want = 0.7 * float(m["stab_anch"]) + 0.4 * float(m["stab_smooth"])
    assert float(total) == pytest.approx(want, rel=1e-5)
    assert float(m["stab_anch"]) == pytest.approx(parts["anch"], rel=1e-3)


def test_w_smooth_needs_the_adapter_input_states() -> None:
    _, corrected, base = term_inputs()
    cfg = StabLossConfig(weight=1.0, w_smooth=0.5)
    with pytest.raises(ValueError, match="sites"):
        stability_term({}, config=cfg, corrected=corrected, base=base)
    with pytest.raises(KeyError, match="adapter input"):
        stability_term(
            {}, config=cfg, corrected=corrected, base=base, sites={5: corrected[5]}
        )


def test_smoothness_alone_needs_no_base_states() -> None:
    sites, corrected, _ = term_inputs()
    cfg = StabLossConfig(weight=1.0, w_anch=0.0, w_smooth=1.0)
    total, m = stability_term({}, config=cfg, corrected=corrected, sites=sites)
    assert "stab_anch" not in m and float(total) == pytest.approx(
        float(m["stab_smooth"])
    )


def test_the_step_term_uses_a_real_trajectory_when_given_one() -> None:
    sites, corrected, _ = term_inputs()
    cfg = StabLossConfig(weight=1.0, w_anch=0.0, w_step=1.0)
    single, _ = stability_term({}, config=cfg, corrected=corrected, sites=sites)
    assert float(single) == 0.0
    traj = {k: jnp.stack([v, v + 0.3, v + 0.9]) for k, v in corrected.items()}
    total, m = stability_term(
        {}, config=cfg, corrected=corrected, sites=sites, trajectories=traj
    )
    assert float(total) > 0.0 and float(m["stab_step"]) == pytest.approx(float(total))


def test_the_loss_terms_vanish_when_their_weights_are_zero() -> None:
    sites, corrected, base = term_inputs()
    cfg = StabLossConfig(weight=1.0, w_anch=0.0)
    total, m = stability_term(
        {}, config=cfg, corrected=corrected, sites=sites, base=base
    )
    assert float(total) == 0.0 and not {"stab_smooth", "stab_step", "stab_anch"} & set(
        m
    )


# --- full path through the substrate ----------------------------------------------------------------


@pytest.fixture(scope="module")
def substrate():
    return make_substrate("gpt2")[1]


@pytest.fixture(scope="module")
def adapter(substrate) -> ResidualAdapter:
    return ResidualAdapter(
        ResidualConfig(rank=4, layers=LAYERS), substrate.architecture
    )


@pytest.fixture(scope="module")
def input_ids() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)


def cfg_all(**stab) -> A2Config:
    kw = {"weight": 1.0, "w_anch": 1.0, "w_smooth": 0.5, "w_step": 1.0, **stab}
    return A2Config(stab=StabLossConfig(**kw))


def probe(cfg: A2Config, adapter: ResidualAdapter, scale: float = 0.05) -> dict:
    params = init_a2_params(cfg, adapter)
    rng = np.random.default_rng(1)
    for k in adapter_params(params):
        params[k]["B"] = jnp.asarray(
            scale * rng.normal(size=params[k]["B"].shape), jnp.float32
        )
    return params


def run(substrate, adapter, cfg, params, ids, **kw):
    return a2_objective(
        params, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids, **kw
    )


def test_both_new_terms_are_logged_and_enter_the_total(
    substrate, adapter, input_ids
) -> None:
    cfg = cfg_all()
    params = probe(cfg, adapter)
    total, t = run(substrate, adapter, cfg, params, input_ids)
    assert {"stab", "stab_anch", "stab_smooth", "stab_step"} <= set(t)
    assert {"smooth_res", "smooth_ratio", "step_sq"} <= set(t)
    assert float(t["stab_smooth"]) > 0.0 and float(t["stab_step"]) == 0.0
    want = 1.0 * float(t["stab_anch"]) + 0.5 * float(t["stab_smooth"])
    assert float(t["stab"]) == pytest.approx(want, rel=1e-5)
    assert float(total) == pytest.approx(float(t["task"]) + float(t["stab"]), rel=1e-5)


def test_the_smoothness_loss_is_zero_at_init(substrate, adapter, input_ids) -> None:
    cfg = cfg_all()
    _, t = run(substrate, adapter, cfg, init_a2_params(cfg, adapter), input_ids)
    assert float(t["stab_smooth"]) == pytest.approx(0.0, abs=1e-8)
    assert float(t["stab_anch"]) == pytest.approx(0.0, abs=1e-6)


def test_smoothness_matches_a_direct_computation(substrate, adapter, input_ids) -> None:
    cfg = cfg_all(w_anch=0.0)
    params = probe(cfg, adapter)
    ap = adapter_params(params)
    run_ = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
    )
    vals = []
    for i in adapter.layers:
        h = run_.hidden_state(i)
        vals.append(float(residual_smoothness_loss(adapter.apply(ap, h, i) - h, h)[0]))
    _, t = run(substrate, adapter, cfg, params, input_ids)
    assert float(t["stab_smooth"]) == pytest.approx(np.mean(vals), rel=2e-3)


def test_smoothness_reaches_only_the_adapter_and_the_step_loss_nothing(
    substrate, adapter, input_ids
) -> None:
    cfg = cfg_all()
    params = probe(cfg, adapter)

    def norm(tree) -> float:
        return sum(float(jnp.sum(x**2)) for x in jax.tree_util.tree_leaves(tree))

    g_smooth = jax.grad(
        lambda p: run(substrate, adapter, cfg, p, input_ids)[1]["stab_smooth"]
    )(params)
    g_step = jax.grad(lambda p: run(substrate, adapter, cfg, p, input_ids)[1]["stab_step"])(params)  # fmt: skip
    assert norm(adapter_params(g_smooth)) > 0.0
    assert (
        norm({k: v for k, v in g_smooth.items() if k not in adapter_params(g_smooth)})
        == 0.0
    )
    assert norm(g_step) == 0.0


def test_the_gradient_is_linear_in_w_smooth(substrate, adapter, input_ids) -> None:
    base = cfg_all(w_anch=0.0, w_step=0.0, w_smooth=1.0)
    params = probe(base, adapter)

    def grad(cfg):
        return jax.grad(lambda p: run(substrate, adapter, cfg, p, input_ids)[0])(params)

    task_only = grad(replace(base, stab=replace(base.stab, weight=0.0)))
    g1 = grad(base)
    g3 = grad(replace(base, stab=replace(base.stab, w_smooth=3.0)))
    jax.tree_util.tree_map(
        lambda t, a, b: np.testing.assert_allclose(
            np.asarray(b) - np.asarray(t), 3 * (np.asarray(a) - np.asarray(t)), rtol=2e-3, atol=1e-5
        ),
        task_only, g1, g3,
    )  # fmt: skip


def test_descending_the_smoothness_loss_smooths_the_correction(
    substrate, adapter, input_ids
) -> None:
    cfg = cfg_all(w_anch=0.0, w_step=0.0, w_smooth=1.0)
    params = probe(cfg, adapter, scale=0.2)
    opt = optax.adam(1e-2)
    state = opt.init(params)

    def loss(p):
        return run(substrate, adapter, cfg, p, input_ids)[1]["stab_smooth"]

    first = float(loss(params))
    for _ in range(25):
        g = jax.grad(loss)(params)
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
    assert float(loss(params)) < 0.8 * first


def test_ignored_labels_keep_the_new_terms_finite(
    substrate, adapter, input_ids
) -> None:
    cfg = cfg_all()
    labels = input_ids.at[:, SEQ // 2 :].set(-100)
    _, t = run(substrate, adapter, cfg, probe(cfg, adapter), input_ids, labels=labels)
    assert all(bool(jnp.isfinite(v)) for v in t.values())


def test_jit_matches_eager(substrate, adapter, input_ids) -> None:
    cfg = cfg_all()
    params = probe(cfg, adapter)
    eager = float(run(substrate, adapter, cfg, params, input_ids)[0])
    jitted = float(
        jax.jit(lambda p: run(substrate, adapter, cfg, p, input_ids)[0])(params)
    )
    assert jitted == pytest.approx(eager, rel=1e-4)


# --- arms -------------------------------------------------------------------------------------------


def test_the_stability_arms_switch_on_the_right_parts() -> None:
    cfg = A2Config()
    smooth = apply_arm(cfg, "stab_smooth")
    assert smooth.stab.enabled and smooth.stab.w_anch == 1.0
    assert (
        smooth.stab.w_smooth > 0 and smooth.stab.w_step == 0 and smooth.stab.w_hep == 0
    )
    allfour = apply_arm(cfg, "stab_all")
    s = allfour.stab
    assert (
        s.enabled and s.w_anch > 0 and s.w_hep > 0 and s.w_smooth > 0 and s.w_step > 0
    )
    assert s.hep.trainable
    for arm in ("a1_mode", "cpc_base", "hid_coupled", "stab_anch", "stab_hep"):
        out = apply_arm(replace(cfg, stab=allfour.stab), arm)
        assert (out.stab.w_smooth, out.stab.w_step) == (
            (0.0, 0.0) if arm != "stab_all" else (s.w_smooth, s.w_step)
        ), arm
    assert A2Config.from_dict(asdict(allfour)) == allfour


def test_arms_keep_configured_weights() -> None:
    cfg = A2Config(
        stab=StabLossConfig(
            weight=0.3, w_smooth=0.9, w_step=4.0, hep=HEPConfig("dense", 4)
        )
    )
    out = apply_arm(cfg, "stab_all").stab
    assert (out.weight, out.w_smooth, out.w_step) == (0.3, 0.9, 4.0)


# --- audit with every stability loss on -------------------------------------------------------------


def test_the_audit_passes_with_all_four_stability_losses(
    substrate, adapter, input_ids
) -> None:
    cfg = apply_arm(A2Config(), "stab_all")
    checks = a2_audit(substrate, adapter, cfg, input_ids)
    names = {c.name for c in checks}
    assert {
        "grad_routing_stab_smooth",
        "grad_routing_stab_step",
        "grad_routing_stab_hep",
    } <= names
    failed = [c for c in checks if c.ok is False]
    assert audit_ok(checks), failed
