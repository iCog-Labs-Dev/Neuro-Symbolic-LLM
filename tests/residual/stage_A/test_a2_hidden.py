"""Stage A2 hidden-prediction loss and stability monitor: config, parameters, the term, the
full path through the frozen substrate and the A1 adapter, gradient routing and the audit
(GPT-2 and Pythia fixtures)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig, layer_key
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    predictor_key,
)
from stages.results import RunWriter, load_params
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_A.a2 import (
    A2Config,
    CPCConfig,
    HiddenConfig,
    MonitorConfig,
    StabLossConfig,
    a2_audit,
    a2_diagnostics,
    a2_objective,
    adapter_params,
    audit_ok,
    hidden_prediction_loss,
    init_a2_params,
    make_a2_optimizer,
    predictor_bank,
    predictor_params,
    token_mask,
)
from stages.stage_base import (
    Domain,
    TrainingConfig,
    make_eval_set,
    run_sequential,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

FAMILIES = ("gpt2", "neox")
SEQ = 12
LAYERS = [5, 6]
FF = 16


def make_cfg(
    *,
    hid: float = 0.0,
    detach: bool = False,
    monitor: str = "off",
    cpc: float = 0.0,
    stab: float = 0.0,
    kl: float = 0.0,
    wd: float = 0.0,
    seed: int = 0,
    **hid_kw,
) -> A2Config:
    return A2Config(
        lambda_kl=kl,
        lambda_wd=wd,
        seed=seed,
        hid_detach_input=detach,
        cpc=CPCConfig(weight=cpc, dim=8),
        stab=StabLossConfig(weight=stab),
        hid=HiddenConfig(
            weight=hid,
            predictor=PredictorConfig(ff_dim=FF, detach_input=detach),
            **hid_kw,
        ),
        monitor=MonitorConfig(mode=monitor),
    )


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


@pytest.fixture(scope="module")
def gpt2() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


def _bank(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, **kw: object
) -> LocalPredictorBank:
    """A raw LocalPredictorBank for the isolation tests."""
    cfg = PredictorConfig(**{"ff_dim": FF, **kw})  # type: ignore[arg-type]
    return LocalPredictorBank(cfg, substrate.architecture, SEQ, adapter.layers)


def _u(d: int, seed: int = 1) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (BATCH, SEQ, d))


def with_random_b(params: dict, scale: float = 0.05) -> dict:
    rng = np.random.default_rng(1)
    out = {k: dict(v) for k, v in params.items()}
    for k in adapter_params(params):
        out[k]["B"] = jnp.asarray(
            scale * rng.normal(size=out[k]["B"].shape), jnp.float32
        )
    return out


def with_random_predictors(params: dict, scale: float = 0.1) -> dict:
    """Non-trivial predictors (the output bias starts at 0)."""
    rng = np.random.default_rng(2)
    out = {k: dict(v) for k, v in params.items()}
    for k in predictor_params(params):
        out[k] = {
            n: v + jnp.asarray(scale * rng.normal(size=v.shape), jnp.float32)
            for n, v in params[k].items()
        }
    return out


def run_objective(substrate, adapter, cfg, params, ids, **kw):
    return a2_objective(
        params, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids, **kw
    )


def grads_of(substrate, adapter, cfg, params, ids, **kw):
    return jax.grad(lambda p: run_objective(substrate, adapter, cfg, p, ids, **kw)[0])(
        params
    )


def max_abs(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return max(float(jnp.max(jnp.abs(x))) for x in leaves) if leaves else 0.0


def test_bank_graph_is_feedforward_without_inference(substrate, adapter) -> None:
    """The compute graph is Identity → LnMlp1 → Linear, no inference node."""
    bank = _bank(substrate, adapter)
    assert bank.structure.config["inference"] is None
    assert list(bank.structure.node_order) == [
        "pred_input",
        "pred_hidden",
        "pred_output",
    ]
    kinds = {n: v.node_info.node_type for n, v in bank.structure.nodes.items()}
    assert kinds == {
        "pred_input": "IdentityNode",
        "pred_hidden": "LnMlp1Node",
        "pred_output": "Linear",
    }


def test_bank_gradient_routing_detach_input(substrate, adapter) -> None:
    """detach_input=True blocks gradients w.r.t. the input tensor u."""
    d = substrate.architecture.hidden_size
    layer = LAYERS[0]
    u = _u(d)
    for detach in (True, False):
        bank = _bank(substrate, adapter, detach_input=detach)
        params = bank.init_params()
        g_params, g_u = jax.grad(
            lambda p, x, bank=bank: jnp.sum(bank.predict(p, layer, x) ** 2),
            argnums=(0, 1),
        )(params, u)
        assert (
            float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(g_params)))
            > 0
        )
        assert (float(jnp.abs(g_u).sum()) == 0.0) is detach


def test_predictors_are_added_per_site_only_when_enabled(substrate, adapter) -> None:
    off = init_a2_params(make_cfg(), adapter)
    assert set(off) == set(adapter.init_params())
    on = init_a2_params(make_cfg(hid=0.1), adapter)
    assert set(on) - set(off) == {predictor_key(layer) for layer in LAYERS}
    assert set(predictor_params(on)) == {predictor_key(layer) for layer in LAYERS}
    assert set(adapter_params(on)) == {layer_key(layer) for layer in LAYERS}


def test_with_hid_off_the_objective_is_a1(substrate, adapter, input_ids) -> None:
    params = with_random_b(init_a2_params(make_cfg(kl=0.1, wd=0.1), adapter))
    total, terms = run_objective(
        substrate, adapter, make_cfg(kl=0.1, wd=0.1), params, input_ids
    )
    a1_total, _ = a1_objective(
        adapter_params(params),
        substrate=substrate,
        adapter=adapter,
        config=A1Config(lambda_kl=0.1, lambda_wd=0.1),
        input_ids=input_ids,
    )
    assert float(total) == pytest.approx(float(a1_total), rel=1e-6)
    assert not any(k.startswith("hid") for k in terms)


def test_the_target_is_the_next_blocks_output_of_the_adapted_pass(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(hid=0.1, detach=False, skip_first_token=False)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    ap = adapter_params(params)
    bank = predictor_bank(cfg, adapter, SEQ)
    # Independent path: intercept every block, apply the adapter on the way.
    every = tuple(range(substrate.architecture.num_layers))
    run = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=every
    )
    losses = []
    for i in LAYERS:
        source = adapter.apply(ap, run.hidden_state(i), i)
        pred = bank.predict(params, i, source)
        losses.append(
            hidden_prediction_loss(pred, run.hidden_state(i + 1), source, cfg)["loss"]
        )
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    assert float(terms["hid"]) == pytest.approx(float(np.mean(losses)), rel=1e-4)


def test_ignored_labels_leave_hid_finite_and_change_it(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(hid=0.1)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    labels = input_ids.at[:, SEQ // 2 :].set(-100)
    _, full = run_objective(substrate, adapter, cfg, params, input_ids)
    _, masked = run_objective(substrate, adapter, cfg, params, input_ids, labels=labels)
    assert bool(jnp.isfinite(masked["hid"]))
    assert float(masked["hid"]) != pytest.approx(float(full["hid"]), rel=1e-6)
    # All labels valid = the same as no labels.
    _, same = run_objective(
        substrate, adapter, cfg, params, input_ids, labels=input_ids
    )
    assert float(same["hid"]) == pytest.approx(float(full["hid"]), rel=1e-5)
    assert token_mask(labels) is not None


def hid_only_grad(substrate, adapter, input_ids, detach: bool):
    cfg = make_cfg(hid=1.0, detach=detach)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))

    def loss(p):
        return run_objective(substrate, adapter, cfg, p, input_ids)[1]["hid"]

    return jax.grad(loss)(params)


def test_coupled_gradient_matches_a_reference_with_a_stopped_target(
    gpt2, input_ids
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=1.0, detach=False)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    bank = predictor_bank(cfg, adapter, SEQ)
    every = tuple(range(gpt2.architecture.num_layers))

    def reference(p):
        ap = adapter_params(p)
        run = gpt2.run_with_interception(
            input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=every
        )
        losses = []
        for i in LAYERS:
            source = adapter.apply(ap, run.hidden_state(i), i)
            target = jax.lax.stop_gradient(run.hidden_state(i + 1))
            pred = bank.predict(p, i, source)
            losses.append(hidden_prediction_loss(pred, target, source, cfg)["loss"])
        return jnp.mean(jnp.stack(losses))

    want = jax.grad(reference)(params)
    got = jax.grad(lambda p: run_objective(gpt2, adapter, cfg, p, input_ids)[1]["hid"])(
        params
    )
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-3, atol=1e-6), got, want
    )


def test_monitor_changes_neither_the_total_nor_a_gradient(
    substrate, adapter, input_ids
) -> None:
    base = make_cfg(hid=0.2, detach=False, cpc=0.1, stab=0.1)
    watched = replace(base, monitor=MonitorConfig(mode="diagnostic"))
    params = with_random_predictors(with_random_b(init_a2_params(base, adapter)))
    t0, _ = run_objective(substrate, adapter, base, params, input_ids)
    t1, _ = run_objective(substrate, adapter, watched, params, input_ids)
    assert float(t0) == float(t1)
    g0 = grads_of(substrate, adapter, base, params, input_ids)
    g1 = grads_of(substrate, adapter, watched, params, input_ids)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), g0, g1)


@pytest.mark.parametrize("target", ["base_anchor", "corrected_teacher"])
def test_diagnostics_follow_selected_target(gpt2, input_ids, target):
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=0.1, hidden_target=target)
    bank = predictor_bank(cfg, adapter, SEQ)
    params = with_random_b(init_a2_params(cfg, adapter, SEQ))
    row = a2_diagnostics(gpt2, adapter, bank, cfg, params, [input_ids])
    suffix = "_base" if target == "base_anchor" else ""
    for name in ("pred_rel", "copy_rel"):
        assert row[f"{name}_selected_mean"] == pytest.approx(
            row[f"{name}{suffix}_mean"]
        )
        for layer in LAYERS:
            assert row[f"{name}_selected_l{layer}"] == pytest.approx(
                row[f"{name}{suffix}_l{layer}"]
            )


@pytest.mark.parametrize("detach", [True, False])
def test_audit_passes_with_hid_and_the_monitor(gpt2, input_ids, detach) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=0.1, detach=detach, monitor="diagnostic", cpc=0.05, stab=0.1)
    checks = a2_audit(gpt2, adapter, cfg, input_ids)
    by_name = {c.name: c for c in checks}
    failed = [c for c in checks if c.ok is False]
    assert audit_ok(checks), failed
    assert by_name["param_groups"].value["pred"] > 0
    assert "monitor_measurement_only" in by_name
    assert "hid_vs_copy" in by_name and by_name["hid_vs_copy"].ok is None
    routing = by_name["grad_routing_hid"].value
    assert routing["pred"] > 0 and (routing["adapter"] > 0) == (not detach)
    assert ("grad_ratio_hid" in by_name) == (not detach)


def _domain(name: str, token: int, num_blocks: int = 8) -> Domain:
    rng = np.random.default_rng(token)
    blocks = np.full((num_blocks, SEQ), token, dtype=np.int32)
    noise = rng.random(blocks.shape) < 0.1
    blocks[noise] = rng.integers(0, NUM_TOKENS, int(noise.sum()))
    return Domain(
        name=name,
        train_blocks=blocks,
        eval_set=make_eval_set(blocks, BATCH, 1),
        epochs=1,
    )


def test_run_sequential_logs_the_hid_terms_and_checkpoints_the_predictors(
    gpt2, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=0.2, detach=False, monitor="diagnostic", cpc=0.1, stab=0.1)
    params = init_a2_params(cfg, adapter)
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)
    with RunWriter(tmp_path, "run") as writer:
        out, summary = run_sequential(
            gpt2,
            adapter,
            params,
            [_domain("a", 5)],
            a2_objective,
            cfg,
            training,
            writer,
        )
    assert set(out) == set(params)
    assert np.all(np.isfinite(np.asarray(summary["loss"])))
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    train = [e for e in map(json.loads, lines) if e["event"] == "train"]
    expected = {"loss_hid", "loss_hid_rel", "loss_hid_copy_rel", "loss_res_smooth"}
    assert expected <= set(train[0])
    assert all(np.isfinite(e["loss_hid"]) for e in train)
    saved = load_params((tmp_path / "run" / "params").glob("*.npz").__next__())
    assert set(saved) == set(params)
    for k in predictor_params(params):
        assert set(saved[k]) == set(params[k])
    assert gpt2.params_unchanged()


def test_predictor_lr_mult_produces_different_updates(gpt2) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg_m1 = make_cfg(hid=0.2, detach=False, predictor_lr_mult=1.0)
    cfg_m2 = make_cfg(hid=0.2, detach=False, predictor_lr_mult=2.0)
    params0 = with_random_predictors(with_random_b(init_a2_params(cfg_m1, adapter)))
    # Both configs must produce the same initial parameter tree.
    params0_m2 = with_random_predictors(with_random_b(init_a2_params(cfg_m2, adapter)))
    for k in params0:
        for n in params0[k]:
            np.testing.assert_array_equal(
                np.asarray(params0[k][n]), np.asarray(params0_m2[k][n])
            )

    lr = 1e-2
    opt_m1 = make_a2_optimizer(cfg_m1, lr)
    opt_m2 = make_a2_optimizer(cfg_m2, lr)

    # Compute a shared gradient.
    rng = np.random.default_rng(42)
    ids = jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)

    def loss_fn(p, cfg):
        return a2_objective(
            p, substrate=gpt2, adapter=adapter, config=cfg, input_ids=ids
        )[0]

    grads = jax.grad(lambda p: loss_fn(p, cfg_m1))(params0)

    # One step with multiplier 1.
    state_m1 = opt_m1.init(params0)
    updates_m1, _ = opt_m1.update(grads, state_m1, params0)
    out_m1 = optax.apply_updates(params0, updates_m1)

    # One step with multiplier 2.
    state_m2 = opt_m2.init(params0)
    updates_m2, _ = opt_m2.update(grads, state_m2, params0)
    out_m2 = optax.apply_updates(params0, updates_m2)

    # 1) Adapter parameters must get identical updates.
    for k in adapter_params(params0):
        np.testing.assert_allclose(
            np.asarray(out_m1[k]["B"]),
            np.asarray(out_m2[k]["B"]),
            atol=1e-7,
            err_msg=f"adapter key {k} should have identical updates",
        )

    # 2) Predictor parameters must differ.
    pred_keys = list(predictor_params(params0))
    assert pred_keys, "expected predictor parameters in the tree"
    for k in pred_keys:
        for n in params0[k]:
            delta_m1 = np.asarray(out_m1[k][n]) - np.asarray(params0[k][n])
            delta_m2 = np.asarray(out_m2[k][n]) - np.asarray(params0[k][n])
            # Where the gradient is non-zero, multiplier 2 should produce
            # updates approximately 2× as large.
            mask = np.abs(delta_m1) > 1e-12
            if not mask.any():
                continue
            ratio = np.abs(delta_m2[mask]) / np.abs(delta_m1[mask])
            np.testing.assert_allclose(
                ratio,
                2.0,
                atol=1e-4,
                err_msg=f"predictor key {k}/{n}: expected update ratio ~2.0",
            )


def test_run_sequential_respects_predictor_lr_mult(gpt2, tmp_path) -> None:
    """End-to-end: ``run_sequential`` with ``make_a2_optimizer`` actually applies the
    multiplier, producing different predictor parameters from a run with multiplier 1.
    """
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)
    domains = [_domain("a", 5)]

    results = {}
    for mult in (1.0, 2.0):
        cfg = make_cfg(hid=0.2, detach=False, predictor_lr_mult=mult)
        params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
        tag = f"m{mult}"
        with RunWriter(tmp_path, tag) as writer:
            out, _ = run_sequential(
                gpt2,
                adapter,
                params,
                domains,
                a2_objective,
                cfg,
                training,
                writer,
                optimizer=make_a2_optimizer(cfg, training.learning_rate),
            )
        results[mult] = out

    pred_m1 = predictor_params(results[1.0])
    pred_m2 = predictor_params(results[2.0])
    assert pred_m1.keys() == pred_m2.keys()
    any_diff = False
    for k in pred_m1:
        for n in pred_m1[k]:
            if not np.allclose(
                np.asarray(pred_m1[k][n]), np.asarray(pred_m2[k][n]), atol=1e-6
            ):
                any_diff = True
    assert any_diff, "predictor params should differ between multiplier 1.0 and 2.0"
    assert gpt2.params_unchanged()
