"""Stage A2 as a whole: ``A2Config``, the flat parameter tree, ``a2_objective`` and its use
by ``run_sequential`` and the runner script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import ResidualAdapter, ResidualConfig, layer_key
from residual.cpc_heads import a_key, q_key
from residual.hep import hep_key
from stages.results import RunWriter, load_params
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_A.a2 import (
    ARMS,
    A2Config,
    CPCConfig,
    HEPConfig,
    StabLossConfig,
    a2_objective,
    adapter_params,
    apply_arm,
    cpc_term,
    init_a2_params,
    param_counts,
    stability_term,
    token_mask,
)
from stages.stage_base import (
    Domain,
    Objective,
    TrainingConfig,
    base_logits,
    evaluate,
    load_configs,
    make_eval_set,
    run_sequential,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED = REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml"
FAMILIES = ("gpt2", "neox")
SEQ = 12
LAYERS = [5, 6]


def make_cfg(
    *,
    kl: float = 0.0,
    wd: float = 0.0,
    cpc: float = 0.0,
    stab: float = 0.0,
    hep: str | None = None,
    w_hep: float = 0.0,
    positive: str = "base",
    seed: int = 0,
) -> A2Config:
    return A2Config(
        lambda_kl=kl,
        lambda_wd=wd,
        seed=seed,
        cpc=CPCConfig(weight=cpc, dim=8, positive=positive),
        stab=StabLossConfig(
            weight=stab, w_hep=w_hep, hep=HEPConfig(kind=hep, rank=4, alpha=1.0)
        ),
    )


# --- configuration ----------------------------------------------------------------------------


def test_defaults_are_the_a1_baseline() -> None:
    cfg = A2Config()
    assert (cfg.lambda_kl, cfg.lambda_wd, cfg.seed) == (0.0, 0.0, 0)
    assert cfg.cpc == CPCConfig() and cfg.stab == StabLossConfig()
    assert not cfg.aux_enabled and not cfg.needs_base_states


def test_flags() -> None:
    assert make_cfg(cpc=0.1).aux_enabled and make_cfg(cpc=0.1).needs_base_states
    assert not make_cfg(cpc=0.1, positive="corrected").needs_base_states
    assert make_cfg(stab=0.1).needs_base_states
    assert make_cfg(cpc=0.1, positive="corrected", stab=0.1).needs_base_states
    assert not make_cfg().needs_base_states


def test_from_dict_builds_the_nested_sections() -> None:
    cfg = A2Config.from_dict(
        {
            "lambda_kl": 0.1,
            "cpc": {"weight": 0.2, "offsets": [1, 2]},
            "stab": {"weight": 0.3, "hep": {"kind": "lowrank", "rank": 4}},
        }
    )
    assert cfg.lambda_kl == 0.1
    assert cfg.cpc.weight == 0.2 and cfg.cpc.offsets == (1, 2)
    assert cfg.stab.weight == 0.3 and cfg.stab.hep == HEPConfig("lowrank", rank=4)
    assert A2Config.from_dict({}) == A2Config()
    # A null section (``cpc:`` with nothing under it in YAML) means the defaults.
    assert A2Config.from_dict({"cpc": None, "stab": None}) == A2Config()


@pytest.mark.parametrize(
    "data",
    [
        {"lambda_k1": 0.1},
        {"cpc": {"wieght": 0.1}},
        {"stab": {"wieght": 0.1}},
        {"stab": {"hep": {"knd": "dense"}}},
    ],
)
def test_unknown_keys_are_rejected_at_every_level(data: dict) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        A2Config.from_dict(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_kl": -1.0},
        {"lambda_wd": -0.1},
        {"lambda_kl": "0.1"},
        {"lambda_kl": True},
        {"seed": 0.5},
        {"seed": True},
    ],
)
def test_invalid_values_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        A2Config(**kwargs)


def test_nested_validation_still_applies() -> None:
    with pytest.raises(ValueError, match="offsets"):
        A2Config.from_dict({"cpc": {"offsets": [0]}})
    with pytest.raises(ValueError, match="fixed"):
        A2Config.from_dict({"stab": {"w_hep": 1.0, "hep": {"kind": "fixed_random"}}})


def test_tied_highway_is_rejected_until_the_lm_head_is_wired() -> None:
    with pytest.raises(ValueError, match="tied"):
        A2Config.from_dict({"stab": {"hep": {"kind": "tied"}}})


def test_asdict_round_trip() -> None:
    cfg = make_cfg(kl=0.1, wd=0.2, cpc=0.3, stab=0.4, hep="lowrank", w_hep=0.5, seed=3)
    assert A2Config.from_dict(asdict(cfg)) == cfg


def test_shipped_yaml_loads_and_round_trips_through_the_writer(tmp_path: Path) -> None:
    shipped = load_configs(SHIPPED, A2Config)
    assert isinstance(shipped.objective, A2Config)
    assert [d.name for d in shipped.domains] == ["eurlex", "ledgar", "scotus"]
    # The shipped arm switches every loss term on and logs the stability metrics.
    obj = shipped.objective
    assert obj.cpc.enabled and obj.stab.enabled and obj.uses_hidden_pred
    assert obj.stability.enabled
    cfg = replace(shipped, objective=make_cfg(cpc=0.2, stab=0.3, hep="lowrank"))
    with RunWriter(tmp_path, "run") as writer:
        path = writer.write_config(cfg.to_dict())
    assert load_configs(path, A2Config) == cfg


def test_a1_style_yaml_loads_as_a2_with_everything_off(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "objective: {lambda_kl: 0.5, lambda_wd: 0.1}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    cfg = load_configs(path, A2Config)
    assert cfg.objective == A2Config(lambda_kl=0.5, lambda_wd=0.1)


# --- ablation arms and shared sections ------------------------------------------------------

A1_YAML = REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml"


def test_a2_yaml_shares_every_non_objective_section_with_a1() -> None:
    # Same adapter, optimiser, data and seeds across A1 and A2: the point of the comparison.
    a1, a2 = load_configs(A1_YAML, A1Config), load_configs(SHIPPED, A2Config)
    assert (a2.adapter, a2.training, a2.domains) == (
        a1.adapter,
        a1.training,
        a1.domains,
    )
    assert (a2.objective.lambda_kl, a2.objective.lambda_wd) == (
        a1.objective.lambda_kl,
        a1.objective.lambda_wd,
    )


def shipped_objective() -> A2Config:
    obj = load_configs(SHIPPED, A2Config).objective
    assert isinstance(obj, A2Config)
    return obj


def test_every_arm_gives_a_valid_config_and_unknown_arms_are_rejected() -> None:
    for arm in ARMS:
        out = apply_arm(shipped_objective(), arm)
        assert A2Config.from_dict(asdict(out)) == out  # re-validated, round-trips
    with pytest.raises(ValueError, match="unknown arm"):
        apply_arm(shipped_objective(), "hid_B_flow")


def test_a1_mode_switches_everything_off() -> None:
    out = apply_arm(shipped_objective(), "a1_mode")
    assert not out.aux_enabled and not out.needs_base_states
    assert out.stab.w_hep == 0 and out.stab.hep.kind is None


@pytest.mark.parametrize(
    ("arm", "cpc_on", "stab_on", "hid_on"),
    [
        ("a1_mode", False, False, False),
        ("stab_anch", False, True, False),
        ("stab_hep", False, True, False),
        ("stab_smooth", False, True, False),
        ("stab_all", False, True, False),
        ("cpc_base", True, False, False),
        ("cpc_corrected", True, False, False),
        ("cpc_multi", True, False, False),
        ("hid_detached", False, False, True),
        ("hid_coupled", False, False, True),
        ("full", True, True, True),
    ],
)
def test_arms_switch_exactly_their_terms_on(arm, cpc_on, stab_on, hid_on) -> None:
    out = apply_arm(shipped_objective(), arm)
    assert (out.cpc.enabled, out.stab.enabled, out.uses_hidden_pred) == (
        cpc_on,
        stab_on,
        hid_on,
    )
    assert set(ARMS) >= {arm}


def test_arm_variants() -> None:
    cfg = shipped_objective()
    assert apply_arm(cfg, "cpc_base").cpc.positive == "base"
    assert apply_arm(cfg, "cpc_corrected").cpc.positive == "corrected"
    assert apply_arm(cfg, "cpc_multi").cpc.offsets == (1, 2, 4)
    hep = apply_arm(cfg, "stab_hep")
    assert hep.stab.hep.trainable and hep.stab.w_hep > 0 and hep.stab.w_anch == 1.0
    anch = apply_arm(cfg, "stab_anch")
    assert anch.stab.w_hep == 0 and not anch.stab.hep.trainable
    full = apply_arm(cfg, "full")
    assert (
        full.cpc.positive == "base" and full.stab.w_anch == 1.0 and full.stab.w_hep == 0
    )
    assert full.hid_coupled and full.uses_hidden_pred


def test_arms_keep_the_shared_settings_and_the_configured_weights() -> None:
    cfg = replace(
        shipped_objective(),
        lambda_kl=0.3,
        lambda_wd=0.02,
        seed=4,
        cpc=CPCConfig(weight=0.4, dim=16, tau=0.2),
        stab=StabLossConfig(weight=0.6, normalize="none"),
    )
    for arm in ARMS:
        out = apply_arm(cfg, arm)
        assert (out.lambda_kl, out.lambda_wd, out.seed) == (0.3, 0.02, 4)
        assert (out.cpc.dim, out.cpc.tau) == (16, 0.2)
        assert out.stab.normalize == "none"
    assert apply_arm(cfg, "cpc_base").cpc.weight == 0.4
    assert apply_arm(cfg, "stab_anch").stab.weight == 0.6
    # A zero weight falls back to the arm default instead of silently running A1.
    zero = replace(cfg, cpc=CPCConfig(weight=0.0), stab=StabLossConfig(weight=0.0))
    assert apply_arm(zero, "cpc_base").cpc.weight > 0
    assert apply_arm(zero, "stab_anch").stab.weight > 0


def test_a_trainable_highway_in_the_yaml_does_not_leak_into_anchor_arms() -> None:
    cfg = replace(
        shipped_objective(),
        stab=StabLossConfig(weight=0.1, hep=HEPConfig(kind="lowrank", rank=4)),
    )
    for arm in ("stab_anch", "full"):
        assert apply_arm(cfg, arm).stab.hep.kind is None
    assert apply_arm(cfg, "stab_hep").stab.hep.kind == "lowrank"


# --- model fixtures ---------------------------------------------------------------------------


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


def with_random_b(params: dict, scale: float = 0.05) -> dict:
    """Give the adapter a non-zero residual (B = 0 at init); heads are left alone."""
    rng = np.random.default_rng(1)
    out = {k: dict(v) for k, v in params.items()}
    for k in adapter_params(params):
        out[k]["B"] = jnp.asarray(
            scale * rng.normal(size=out[k]["B"].shape), jnp.float32
        )
    return out


def run_objective(substrate, adapter, cfg, params, ids, **kw):
    return a2_objective(
        params,
        substrate=substrate,
        adapter=adapter,
        config=cfg,
        input_ids=ids,
        **kw,
    )


def grads_of(substrate, adapter, cfg, params, ids, **kw):
    def total(p):
        return run_objective(substrate, adapter, cfg, p, ids, **kw)[0]

    return jax.grad(total)(params)


def max_abs(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return max(float(jnp.max(jnp.abs(x))) for x in leaves) if leaves else 0.0


# --- parameters -------------------------------------------------------------------------------


def test_with_every_aux_term_off_the_tree_is_the_adapter_tree(
    substrate, adapter
) -> None:
    params = init_a2_params(make_cfg(kl=0.1, wd=0.1), adapter)
    expected = adapter.init_params()
    assert set(params) == set(expected)
    for k in expected:
        for name in expected[k]:
            np.testing.assert_array_equal(params[k][name], expected[k][name])


def test_cpc_heads_are_added_per_site_and_offset(substrate, adapter) -> None:
    cfg = A2Config(cpc=CPCConfig(weight=0.1, dim=8, offsets=(1, 2)))
    params = init_a2_params(cfg, adapter)
    expected = {layer_key(layer) for layer in LAYERS}
    for layer in LAYERS:
        expected |= {q_key(layer), a_key(layer, 1), a_key(layer, 2)}
    assert set(params) == expected
    d = substrate.architecture.hidden_size
    assert params[q_key(5)]["W"].shape == (d, 8)


@pytest.mark.parametrize(
    ("kind", "leaves"), [("dense", {"V"}), ("lowrank", {"U", "W"})]
)
def test_trainable_highways_are_added_and_fixed_ones_are_not(
    substrate, adapter, kind, leaves
) -> None:
    cfg = make_cfg(stab=0.1, hep=kind)
    params = init_a2_params(cfg, adapter)
    for layer in LAYERS:
        assert set(params[hep_key(layer)]) == leaves
    fixed = init_a2_params(make_cfg(stab=0.1, hep="fixed_random"), adapter)
    assert set(fixed) == set(adapter.init_params())
    # Disabled stability (weight 0) allocates nothing, whatever the kind.
    off = init_a2_params(make_cfg(stab=0.0, hep="lowrank"), adapter)
    assert set(off) == set(adapter.init_params())


def test_tree_is_flat_two_level_and_checkpointable(
    substrate, adapter, tmp_path
) -> None:
    cfg = make_cfg(cpc=0.1, stab=0.1, hep="lowrank", w_hep=0.1)
    params = init_a2_params(cfg, adapter)
    assert all(isinstance(v, dict) for v in params.values())
    assert all(
        isinstance(leaf, jax.Array) for v in params.values() for leaf in v.values()
    )
    with RunWriter(tmp_path, "run") as writer:
        path = writer.save_params("t", params)
    saved = load_params(path)
    assert set(saved) == set(params)
    for k in params:
        for name in params[k]:
            np.testing.assert_array_equal(saved[k][name], np.asarray(params[k][name]))


def test_adapter_params_selects_only_the_adapter(substrate, adapter) -> None:
    params = init_a2_params(make_cfg(cpc=0.1, stab=0.1, hep="lowrank"), adapter)
    assert set(adapter_params(params)) == {layer_key(layer) for layer in LAYERS}


def test_param_counts(substrate, adapter) -> None:
    cfg = make_cfg(cpc=0.1, stab=0.1, hep="lowrank")
    params = init_a2_params(cfg, adapter)
    counts = param_counts(params)
    assert counts["adapter"] == ResidualAdapter.num_params(adapter.init_params())
    d, v = substrate.architecture.hidden_size, substrate.architecture.vocab_size
    assert counts["cpc"] == len(LAYERS) * (d * 8 + 8 * 8)
    assert counts["hep"] == len(LAYERS) * (v * 4 + 4 * d)
    assert counts["total"] == counts["adapter"] + counts["cpc"] + counts["hep"]
    assert param_counts(adapter.init_params())["total"] == counts["adapter"]
    with pytest.raises(ValueError, match="no known group"):
        param_counts({**params, "stray": {"w": jnp.ones(3)}})


def test_seed_controls_the_heads_and_not_the_adapter(substrate, adapter) -> None:
    a = init_a2_params(make_cfg(cpc=0.1, seed=0), adapter)
    b = init_a2_params(make_cfg(cpc=0.1, seed=0), adapter)
    c = init_a2_params(make_cfg(cpc=0.1, seed=1), adapter)
    np.testing.assert_array_equal(a[q_key(5)]["W"], b[q_key(5)]["W"])
    assert not np.allclose(a[q_key(5)]["W"], c[q_key(5)]["W"])
    np.testing.assert_array_equal(a[layer_key(5)]["A"], c[layer_key(5)]["A"])


# --- objective: the A1 contract and parity ----------------------------------------------------


@pytest.mark.parametrize(
    ("kl", "wd"), [(0.0, 0.0), (0.5, 0.0), (0.0, 0.01), (0.3, 0.02)]
)
def test_equals_a1_when_the_aux_terms_are_off(
    substrate, adapter, input_ids, kl, wd
) -> None:
    cfg = make_cfg(kl=kl, wd=wd)
    params = with_random_b(init_a2_params(cfg, adapter))

    def a1_total(p):
        return a1_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=A1Config(lambda_kl=kl, lambda_wd=wd),
            input_ids=input_ids,
        )

    t1, terms1 = a1_total(params)
    t2, terms2 = run_objective(substrate, adapter, cfg, params, input_ids)
    assert set(terms2) == set(terms1)
    np.testing.assert_allclose(t2, t1, rtol=1e-6)
    for k in terms1:
        np.testing.assert_allclose(terms2[k], terms1[k], rtol=1e-6, atol=1e-7)
    g1 = jax.grad(lambda p: a1_total(p)[0])(params)
    g2 = grads_of(substrate, adapter, cfg, params, input_ids)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8), g1, g2
    )


def test_contract_terms_are_present_and_finite(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(cpc=0.1, stab=0.1, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, adapter))
    total, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    assert {"total", "task", "wd", "cpc", "stab"} <= set(terms)
    assert terms["total"] is total or float(terms["total"]) == float(total)
    assert all(np.isfinite(float(v)) for v in terms.values())
    assert all(jnp.ndim(v) == 0 for v in terms.values())
    assert substrate.params_unchanged()


@pytest.mark.parametrize(
    ("cpc", "stab", "present", "absent"),
    [
        (0.1, 0.0, {"cpc", "cpc_top1", "cpc_pos_sim"}, {"stab", "stab_anch"}),
        (0.0, 0.1, {"stab", "stab_anch", "anch_ratio"}, {"cpc", "cpc_top1"}),
        (0.1, 0.1, {"cpc", "stab"}, set()),
        (0.0, 0.0, set(), {"cpc", "stab", "kl"}),
    ],
)
def test_only_enabled_terms_are_returned(
    substrate, adapter, input_ids, cpc, stab, present, absent
) -> None:
    cfg = make_cfg(cpc=cpc, stab=stab)
    params = init_a2_params(cfg, adapter)
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    assert present <= set(terms)
    assert not absent & set(terms)


def test_total_is_the_weighted_sum_of_the_logged_terms(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(kl=0.3, wd=0.02, cpc=0.4, stab=0.7, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, adapter))
    total, t = run_objective(substrate, adapter, cfg, params, input_ids)
    expected = (
        t["task"] + 0.3 * t["kl"] + 0.02 * t["wd"] + 0.4 * t["cpc"] + 0.7 * t["stab"]
    )
    np.testing.assert_allclose(total, expected, rtol=1e-5)
    # Every logged term is the unweighted one: doubling a weight moves the total by it.
    total2, t2 = run_objective(
        substrate,
        adapter,
        replace(cfg, cpc=replace(cfg.cpc, weight=0.8)),
        params,
        input_ids,
    )
    np.testing.assert_allclose(t2["cpc"], t["cpc"], rtol=1e-6)
    np.testing.assert_allclose(total2 - total, 0.4 * t["cpc"], rtol=1e-4, atol=1e-6)
    total3, t3 = run_objective(
        substrate,
        adapter,
        replace(cfg, stab=replace(cfg.stab, weight=1.4)),
        params,
        input_ids,
    )
    np.testing.assert_allclose(t3["stab"], t["stab"], rtol=1e-6)
    np.testing.assert_allclose(total3 - total, 0.7 * t["stab"], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("positive", ["base", "corrected"])
def test_terms_match_an_independent_path_through_the_model(
    substrate, adapter, input_ids, positive
) -> None:
    cfg = make_cfg(cpc=0.5, stab=0.5, hep="fixed_random", positive=positive)
    params = with_random_b(init_a2_params(cfg, adapter))
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)

    ap = adapter_params(params)
    run = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
    )
    corrected = {k: adapter.apply(ap, run.hidden_state(k), k) for k in adapter.layers}
    base_run = substrate.run_with_interception(
        input_ids, modify_fn=identity_modify, intercept_layers=adapter.layers
    )
    base = {k: base_run.hidden_state(k) for k in adapter.layers}
    l_cpc, m_cpc = cpc_term(params, config=cfg.cpc, corrected=corrected, base=base)
    l_stab, m_stab = stability_term(
        params,
        config=cfg.stab,
        corrected=corrected,
        base=base,
        logits=run.logits,
        labels=input_ids,
        seed=cfg.seed,
    )
    np.testing.assert_allclose(terms["cpc"], l_cpc, rtol=1e-5)
    np.testing.assert_allclose(terms["stab"], l_stab, rtol=1e-5)
    for name, value in {**m_cpc, **m_stab}.items():
        np.testing.assert_allclose(terms[name], value, rtol=1e-5, atol=1e-7)


def test_precomputed_base_logits_give_the_same_result(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(kl=0.5, wd=0.01, cpc=0.2, stab=0.2)
    params = with_random_b(init_a2_params(cfg, adapter))
    t0, terms0 = run_objective(substrate, adapter, cfg, params, input_ids)
    base = base_logits(substrate, input_ids)
    t1, terms1 = run_objective(substrate, adapter, cfg, params, input_ids, base=base)
    np.testing.assert_allclose(t1, t0, rtol=1e-5)
    for k in terms0:
        np.testing.assert_allclose(terms1[k], terms0[k], rtol=1e-5, atol=1e-7)


def test_weight_decay_covers_the_adapter_and_not_the_heads(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(wd=0.5, cpc=0.1, stab=0.1, hep="lowrank", w_hep=0.1)
    params = with_random_b(init_a2_params(cfg, adapter))
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    expected = ResidualAdapter.l2_norm_sq(adapter_params(params))
    np.testing.assert_allclose(terms["wd"], expected, rtol=1e-6)
    everything = ResidualAdapter.l2_norm_sq(params)
    assert float(everything) > 1.01 * float(expected)  # the heads do carry weight


def test_the_seed_reaches_the_fixed_random_highway(
    substrate, adapter, input_ids
) -> None:
    def nudge(seed: int) -> float:
        cfg = make_cfg(stab=0.1, hep="fixed_random", seed=seed)
        params = with_random_b(init_a2_params(cfg, adapter))
        return float(
            run_objective(substrate, adapter, cfg, params, input_ids)[1][
                "hep_nudge_norm"
            ]
        )

    assert nudge(0) == pytest.approx(nudge(0))
    assert nudge(0) != pytest.approx(nudge(1), rel=1e-3)


def test_kl_is_skipped_without_a_weight_and_without_base(
    substrate, adapter, input_ids
) -> None:
    # Like A1: no KL term (not a NaN) when nothing needs it, even with aux terms on.
    cfg = make_cfg(cpc=0.1, positive="corrected")
    _, terms = run_objective(
        substrate, adapter, cfg, init_a2_params(cfg, adapter), input_ids
    )
    assert "kl" not in terms
    # Base states are needed for the anchor, so the KL comes for free (weight 0).
    cfg = make_cfg(stab=0.1)
    total, terms = run_objective(
        substrate, adapter, cfg, init_a2_params(cfg, adapter), input_ids
    )
    assert np.isfinite(float(terms["kl"]))
    np.testing.assert_allclose(total, terms["task"] + 0.1 * terms["stab"], rtol=1e-5)


def test_the_last_layer_can_be_adapted(substrate, input_ids) -> None:
    last = substrate.architecture.num_layers - 1
    ad = ResidualAdapter(
        ResidualConfig(rank=4, layers=[last - 1, last]), substrate.architecture
    )
    cfg = make_cfg(cpc=0.1, stab=0.1, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, ad))
    _, terms = run_objective(substrate, ad, cfg, params, input_ids)
    assert all(np.isfinite(float(v)) for v in terms.values())


# --- objective: padding -----------------------------------------------------------------------


def test_all_valid_labels_equal_no_labels(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(cpc=0.3, stab=0.3, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, adapter))
    t0, terms0 = run_objective(substrate, adapter, cfg, params, input_ids)
    t1, terms1 = run_objective(
        substrate, adapter, cfg, params, input_ids, labels=input_ids
    )
    np.testing.assert_allclose(t1, t0, rtol=1e-5)
    for k in terms0:
        np.testing.assert_allclose(terms1[k], terms0[k], rtol=1e-5, atol=1e-7)


def test_ignored_labels_leave_the_auxiliary_terms(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(cpc=0.3, stab=0.3, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, adapter))
    labels = input_ids.at[:, SEQ // 2 :].set(-100)
    mask = token_mask(labels)
    assert mask is not None and int(mask.sum()) == BATCH * (SEQ // 2)
    _, full = run_objective(substrate, adapter, cfg, params, input_ids)
    _, part = run_objective(substrate, adapter, cfg, params, input_ids, labels=labels)
    assert all(np.isfinite(float(v)) for v in part.values())
    assert float(part["cpc"]) != pytest.approx(float(full["cpc"]), rel=1e-4)
    assert float(part["stab"]) != pytest.approx(float(full["stab"]), rel=1e-4)
    # Changing the ignored tail's tokens cannot change the masked terms: the causal
    # model's real positions do not see later tokens.
    other = input_ids.at[:, SEQ // 2 :].set((input_ids[:, SEQ // 2 :] + 7) % NUM_TOKENS)
    _, again = run_objective(substrate, adapter, cfg, params, other, labels=labels)
    for k in ("cpc", "stab", "task", "stab_hep", "hep_nudge_norm"):
        np.testing.assert_allclose(again[k], part[k], rtol=1e-4, atol=1e-6)
    # The HEP error is masked by the same labels (the full-label value differs).
    assert float(part["stab_hep"]) != pytest.approx(float(full["stab_hep"]), rel=1e-4)


# --- objective: which term trains which parameter -----------------------------------------------


def tree_diff(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def test_task_only_gradient_reaches_only_the_adapter(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg()
    params = with_random_b(init_a2_params(cfg, adapter))
    g = grads_of(substrate, adapter, cfg, params, input_ids)
    assert set(g) == set(params)
    assert max_abs(g) > 0


def test_cpc_trains_the_heads_and_the_adapter(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(cpc=0.5)
    params = with_random_b(init_a2_params(cfg, adapter))
    g = grads_of(substrate, adapter, cfg, params, input_ids)
    assert set(g) == set(params)
    for layer in LAYERS:
        assert max_abs(g[q_key(layer)]) > 0
        assert max_abs(g[a_key(layer, 1)]) > 0
    # Only L_cpc reaches the heads.
    g_off = grads_of(substrate, adapter, make_cfg(), adapter_params(params), input_ids)
    for layer in LAYERS:
        diff = max_abs(tree_diff(g[layer_key(layer)], g_off[layer_key(layer)]))
        assert diff > 0


@pytest.mark.parametrize("term", ["cpc", "stab"])
def test_a_term_enters_the_gradient_linearly_with_its_weight(
    substrate, adapter, input_ids, term
) -> None:
    hep = "fixed_random" if term == "stab" else None
    params = with_random_b(
        init_a2_params(make_cfg(cpc=1.0, stab=1.0, hep=hep), adapter)
    )

    def grad(w: float) -> Any:
        cfg = make_cfg(
            cpc=w if term == "cpc" else 0.0,
            stab=w if term == "stab" else 0.0,
            hep=hep,
        )
        return grads_of(substrate, adapter, cfg, params, input_ids)

    g0, g1, g2 = grad(0.0), grad(0.25), grad(0.5)
    for layer in LAYERS:
        key = layer_key(layer)
        d1 = tree_diff(g1[key], g0[key])
        d2 = tree_diff(g2[key], g0[key])
        assert max_abs(d1) > 0
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(2 * a, b, rtol=1e-3, atol=1e-7),
            d1,
            d2,
        )


def test_anchor_alone_adds_no_parameters_and_pulls_the_adapter(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(stab=1.0)
    params = with_random_b(init_a2_params(cfg, adapter))
    assert set(params) == set(adapter.init_params())
    # Along the anchor gradient the stability term decreases.
    g = grads_of(substrate, adapter, cfg, params, input_ids)
    stepped = jax.tree_util.tree_map(lambda p, gr: p - 1e-2 * gr, params, g)
    _, before = run_objective(substrate, adapter, cfg, params, input_ids)
    _, after = run_objective(substrate, adapter, cfg, stepped, input_ids)
    assert float(after["stab"]) < float(before["stab"])


def test_hep_loss_trains_only_the_highway(substrate, adapter, input_ids) -> None:
    with_loss = make_cfg(stab=0.5, hep="lowrank", w_hep=1.0)
    metric_only = make_cfg(stab=0.5, hep="lowrank", w_hep=0.0)
    params = with_random_b(init_a2_params(with_loss, adapter))
    g_loss = grads_of(substrate, adapter, with_loss, params, input_ids)
    g_metric = grads_of(substrate, adapter, metric_only, params, input_ids)
    for layer in LAYERS:
        assert max_abs(g_loss[hep_key(layer)]) > 0
        assert max_abs(g_metric[hep_key(layer)]) == 0
        # eps is under stop-gradient: the HEP term never reaches the adapter.
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-9),
            g_loss[layer_key(layer)],
            g_metric[layer_key(layer)],
        )
    _, terms = run_objective(substrate, adapter, with_loss, params, input_ids)
    assert {"stab_hep", "hep_nudge_norm", "hep_nudge_ratio"} <= set(terms)


def test_jit_matches_eager(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(kl=0.2, wd=0.01, cpc=0.3, stab=0.3, hep="lowrank", w_hep=0.2)
    params = with_random_b(init_a2_params(cfg, adapter))

    def f(p, ids):
        return run_objective(substrate, adapter, cfg, p, ids)

    eager_total, eager_terms = f(params, input_ids)
    jit_total, jit_terms = jax.jit(f)(params, input_ids)
    np.testing.assert_allclose(jit_total, eager_total, rtol=1e-4)
    for k in eager_terms:
        np.testing.assert_allclose(jit_terms[k], eager_terms[k], rtol=1e-4, atol=1e-6)
    g_eager = jax.grad(lambda p: f(p, input_ids)[0])(params)
    g_jit = jax.jit(jax.grad(lambda p, ids: f(p, ids)[0]))(params, input_ids)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6),
        g_eager,
        g_jit,
    )


# --- training with the objective --------------------------------------------------------------


def test_optimising_the_objective_lowers_every_enabled_term(gpt2, input_ids) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(cpc=0.5, stab=0.5, hep="lowrank", w_hep=0.5)
    params = init_a2_params(cfg, adapter)
    opt = optax.adam(1e-2)
    state = opt.init(params)

    @jax.jit
    def step(p, s, ids):
        (_, terms), g = jax.value_and_grad(
            lambda q: run_objective(gpt2, adapter, cfg, q, ids), has_aux=True
        )(p)
        updates, s = opt.update(g, s, p)
        return optax.apply_updates(p, updates), s, terms

    first = None
    start = params
    for _ in range(40):
        params, state, terms = step(params, state, input_ids)
        first = terms if first is None else first
    assert first is not None
    assert float(terms["total"]) < float(first["total"])
    assert float(terms["task"]) < float(first["task"])
    assert float(terms["cpc"]) < float(first["cpc"])
    # Every group moved; the frozen base did not.
    for key in params:
        assert max_abs(tree_diff(params[key], start[key])) > 0, key
    assert gpt2.params_unchanged()


# --- run_sequential and the runner ------------------------------------------------------------


def _domain(name: str, token: int, epochs: int = 1, num_blocks: int = 8) -> Domain:
    rng = np.random.default_rng(token)
    blocks = np.full((num_blocks, SEQ), token, dtype=np.int32)
    noise = rng.random(blocks.shape) < 0.1
    blocks[noise] = rng.integers(0, NUM_TOKENS, int(noise.sum()))
    return Domain(
        name=name,
        train_blocks=blocks,
        eval_set=make_eval_set(blocks, BATCH, 1),
        epochs=epochs,
    )


def _events(run: Path) -> list[dict]:
    return [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]


def test_run_sequential_trains_logs_and_checkpoints_the_whole_tree(
    gpt2, tmp_path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(kl=0.1, wd=0.01, cpc=0.2, stab=0.2, hep="lowrank", w_hep=0.2)
    params = init_a2_params(cfg, adapter)
    domains = [_domain("a", 5), _domain("b", 9, num_blocks=6)]
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)
    with RunWriter(tmp_path, "run") as writer:
        out, summary = run_sequential(
            gpt2, adapter, params, domains, a2_objective, cfg, training, writer
        )

    assert set(out) == set(params)
    loss = np.asarray(summary["loss"])
    assert loss.shape == (2, 2) and np.all(np.isfinite(loss))
    assert loss[0, 0] < summary["base_loss"][0]
    assert summary["steps"] == 7

    events = _events(tmp_path / "run")
    train = [e for e in events if e["event"] == "train"]
    expected = {
        "loss_total",
        "loss_task",
        "loss_wd",
        "loss_kl",
        "loss_cpc",
        "loss_stab",
    }
    assert expected <= set(train[0])
    assert {"loss_cpc_top1", "loss_stab_anch", "loss_stab_hep"} <= set(train[0])
    assert all(np.isfinite(e["loss_cpc"]) for e in train)

    saved = load_params(tmp_path / "run" / "params" / "task_1_b.npz")
    assert set(saved) == set(params)
    for key in params:
        for name in params[key]:
            np.testing.assert_allclose(saved[key][name], np.asarray(out[key][name]))
    assert gpt2.params_unchanged()


def test_evaluation_ignores_the_auxiliary_heads(gpt2) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(cpc=0.2, stab=0.2, hep="lowrank", w_hep=0.2)
    params = with_random_b(init_a2_params(cfg, adapter))
    eval_set = _domain("a", 5).eval_set
    full = evaluate(gpt2, adapter, params, eval_set)
    only = evaluate(gpt2, adapter, adapter_params(params), eval_set)
    assert full == pytest.approx(only)


def test_with_aux_terms_off_a_run_is_the_a1_run(gpt2, tmp_path) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)
    summaries, trains = [], []
    runs: list[tuple[str, Objective, Any]] = [
        ("a1", a1_objective, A1Config(lambda_kl=0.1, lambda_wd=0.01)),
        ("a2", a2_objective, make_cfg(kl=0.1, wd=0.01)),
    ]
    for name, objective, cfg in runs:
        with RunWriter(tmp_path, name) as writer:
            _, summary = run_sequential(
                gpt2,
                adapter,
                init_a2_params(make_cfg(), adapter),
                [_domain("a", 5)],
                objective,
                cfg,
                training,
                writer,
            )
        summaries.append(summary)
        trains.append([e for e in _events(tmp_path / name) if e["event"] == "train"])
    np.testing.assert_allclose(summaries[0]["loss"], summaries[1]["loss"], rtol=1e-4)
    for a, b in zip(trains[0], trains[1], strict=True):
        for key in ("loss_total", "loss_task", "loss_kl", "loss_wd"):
            assert b[key] == pytest.approx(a[key], rel=1e-4, abs=1e-6)


@pytest.fixture(scope="module")
def runner() -> Any:
    path = REPO_ROOT / "experiments" / "run_stage_a2.py"
    spec = importlib.util.spec_from_file_location("run_stage_a2_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        sys.modules.pop(spec.name, None)
        pytest.skip("experiments/run_stage_a2.py is not implemented yet (runner PR)")
    yield module
    sys.modules.pop(spec.name, None)


def test_runner_overrides_reach_the_objective(runner: Any) -> None:
    cfg = load_configs(SHIPPED, A2Config)
    args = runner.parse_args(
        ["--epochs", "2", "--rank", "4", "--lambda-cp", "0.5", "--lambda-stab", "0"]
    )
    out = runner.apply_overrides(cfg, args)
    assert out.training.epochs == 2 and out.adapter.rank == 4
    assert out.objective.cpc.weight == 0.5 and out.objective.stab.weight == 0.0
    assert out.objective.cpc.dim == cfg.objective.cpc.dim
    assert not out.objective.stab.enabled
    untouched = runner.apply_overrides(cfg, runner.parse_args([]))
    assert untouched == cfg
    kl = runner.apply_overrides(cfg, runner.parse_args(["--lambda-kl", "0.3"]))
    assert kl.objective.lambda_kl == 0.3 and kl.objective.cpc == cfg.objective.cpc


def test_runner_arm_and_seed_overrides(runner: Any) -> None:
    cfg = load_configs(SHIPPED, A2Config)
    args = runner.parse_args(["--arm", "cpc_multi", "--seed", "7"])
    out = runner.apply_overrides(cfg, args)
    assert out.objective.cpc.offsets == (1, 2, 4) and not out.objective.stab.enabled
    assert out.adapter.seed == out.training.seed == out.objective.seed == 7
    assert (out.adapter.rank, out.domains) == (cfg.adapter.rank, cfg.domains)
    # Explicit weights win over the arm; the arm wins over the YAML.
    args = runner.parse_args(["--arm", "full", "--lambda-stab", "0"])
    out = runner.apply_overrides(cfg, args)
    assert out.objective.cpc.enabled and not out.objective.stab.enabled
    with pytest.raises(SystemExit):
        runner.parse_args(["--arm", "nope"])


@pytest.mark.parametrize("training", ["batch_size: 1", "seq_len: 2"])
def test_runner_rejects_inactive_cpc_before_loading_model(
    runner, tmp_path, monkeypatch, training
):
    cfg = tmp_path / "invalid.yaml"
    cfg.write_text(
        "objective:\n  cpc: {weight: 0.1, offsets: [2]}\n"
        f"training:\n  {training}\n"
        "domains:\n  - {name: a, dataset: unused}\n"
    )

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("Invalid CPC settings must fail before loading a model")

    monkeypatch.setattr(runner, "FrozenSubstrate", unexpected_model_load)
    with pytest.raises(ValueError, match="CPC"):
        runner.main(["--config", str(cfg)])


def test_runner_predictor_and_stability_overrides(runner: Any) -> None:
    cfg = load_configs(SHIPPED, A2Config)
    args = runner.parse_args(
        [
            "--lambda-hid", "0.4",
            "--hidden-target", "base_anchor",
            "--ff-dim", "8",
            "--no-detach-input",
            "--predictor-lr-mult", "2",
            "--stability-mode", "off",
        ]
    )  # fmt: skip
    obj = runner.apply_overrides(cfg, args).objective
    assert (obj.lambda_hid, obj.hidden_target) == (0.4, "base_anchor")
    assert obj.predictor.ff_dim == 8 and obj.hid_coupled
    assert obj.predictor_lr_mult == 2.0 and not obj.stability.enabled
    assert obj.cpc == cfg.objective.cpc and obj.stab == cfg.objective.stab


def test_runner_main_end_to_end(
    runner: Any,
    gpt2: FrozenSubstrate,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    # No dataset download: the loader and the model constructor are stubbed.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "adapter: {rank: 4, layers: [5, 6]}\n"
        "objective:\n  lambda_kl: 0.1\n"
        "  cpc: {weight: 0.2, dim: 8}\n"
        "  stab: {weight: 0.2, hep: {kind: fixed_random, rank: 4}}\n"
        f"training: {{model: gpt2, seq_len: {SEQ}, batch_size: {BATCH}, epochs: 1, "
        "learning_rate: 1.0e-2, eval_batches: 1, log_every: 1}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    blocks = _domain("a", 5, num_blocks=4).train_blocks
    monkeypatch.setattr(runner, "FrozenSubstrate", lambda _name: gpt2)
    monkeypatch.setattr(runner, "load_token_blocks", lambda *a, **k: blocks)

    code = runner.main(
        [
            "--arm",
            "full",
            "--seed",
            "3",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 0
    run = tmp_path / "runs" / "r"
    results = json.loads((run / "results.json").read_text())
    assert results["params_unchanged"] is True
    assert np.isfinite(results["test"]["a"]["loss"])
    meta = json.loads((run / "meta.json").read_text())
    assert meta["stage"] == "A2" and meta["arm"] == "full" and meta["seed"] == 3
    groups = meta["trainable_params_by_group"]
    assert groups["cpc"] > 0 and groups["hep"] == 0
    assert groups["pred"] > 0 and meta["predictor_params"] == groups["pred"]
    assert meta["predictor_layers"] == [5, 6]
    assert groups["total"] == meta["trainable_params"]
    saved = load_params(run / "params" / "task_0_a.npz")
    assert q_key(5) in saved and layer_key(5) in saved
    config = load_configs(run / "config.yaml", A2Config)
    assert config.objective.cpc.weight == 0.2
    assert config.objective.seed == config.training.seed == config.adapter.seed == 3
    # Exercise the real CLI load/filter/strict-validate/evaluate path after training
    # a combined hidden + CPC checkpoint. Patch only external model/data loading.
    import stages.stage_base as stage_base

    monkeypatch.setattr(stage_base, "load_token_blocks", lambda *a, **k: blocks)
    checkpoint = run / "params" / "task_0_a.npz"
    capsys.readouterr()
    assert (
        runner.main(
            ["--config", str(run / "config.yaml"), "--eval-checkpoint", str(checkpoint)]
        )
        == 0
    )
    reloaded = json.loads(capsys.readouterr().out)
    expected = evaluate(
        gpt2,
        ResidualAdapter(config.adapter, gpt2.architecture),
        saved,
        make_eval_set(blocks, BATCH, 1),
    )
    assert reloaded["a"] == pytest.approx(expected)
    # A1's validator still rejects auxiliary keys.
    with pytest.raises(ValueError, match="extra layers"):
        stage_base.validate_checkpoint_params(
            ResidualAdapter(config.adapter, gpt2.architecture), saved
        )
    assert gpt2.params_unchanged()
