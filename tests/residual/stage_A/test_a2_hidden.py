"""Stage A2 hidden-prediction loss and stability monitor: config, parameters, the term, the
full path through the frozen substrate and the A1 adapter, gradient routing and the audit
(GPT-2 and Pythia fixtures)."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
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
    merge_params,
    predictor_key,
    split_params,
)
from stages.results import RunWriter, load_params
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_A.a2 import (
    ARM_DEFAULT_WEIGHTS,
    ARMS,
    A2Config,
    CPCConfig,
    HiddenConfig,
    MonitorConfig,
    StabLossConfig,
    a2_audit,
    a2_objective,
    adapter_params,
    apply_arm,
    audit_ok,
    hid_term,
    hidden_prediction_loss,
    init_a2_params,
    param_counts,
    predictor_bank,
    predictor_params,
    token_mask,
)
from stages.stage_base import (
    Domain,
    TrainingConfig,
    evaluate,
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
    detach: bool = True,
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
        cpc=CPCConfig(weight=cpc, dim=8),
        stab=StabLossConfig(weight=stab),
        hid=HiddenConfig(
            weight=hid,
            predictor=PredictorConfig(ff_dim=FF, detach_input=detach),
            **hid_kw,
        ),
        monitor=MonitorConfig(mode=monitor),
    )


# --- configuration ----------------------------------------------------------------------------


def test_defaults_are_off() -> None:
    cfg = HiddenConfig()
    assert not cfg.enabled and cfg.weight == 0.0 and cfg.lambda_hid is None
    assert (cfg.hidden_target, cfg.hidden_norm, cfg.skip_first_token) == (
        "corrected_teacher",
        "target_var",
        True,
    )
    assert cfg.predictor == PredictorConfig() and cfg.predictor.detach_input is True
    assert cfg.predictor_lr_mult == 1.0
    a2 = A2Config()
    assert a2.hid == cfg and a2.monitor == MonitorConfig() and not a2.monitor.enabled
    assert not a2.aux_enabled and not a2.needs_corrected and not a2.uses_hidden_pred


def test_flags() -> None:
    assert make_cfg(hid=0.1).aux_enabled and make_cfg(hid=0.1).needs_corrected
    assert make_cfg(hid=0.1).uses_hidden_pred
    # The corrected-teacher target is the next block, not h^base; base_anchor needs it.
    assert not make_cfg(hid=0.1).needs_base_states
    assert not make_cfg(hid=0.1).needs_base_targets
    anchor = make_cfg(hid=0.1, hidden_target="base_anchor")
    assert anchor.needs_base_targets and anchor.needs_base_states
    assert not make_cfg(hidden_target="base_anchor").needs_base_targets
    assert not make_cfg(monitor="diagnostic").aux_enabled
    assert make_cfg(monitor="diagnostic").needs_corrected
    assert make_cfg(hid=1.0, detach=False).hid_coupled
    assert make_cfg(hid=1.0, detach=False).hid.coupled
    assert not make_cfg(hid=1.0).hid_coupled
    assert not make_cfg(hid=0.0, detach=False).hid_coupled


def test_flat_read_access_mirrors_the_nested_sections() -> None:
    cfg = make_cfg(
        hid=0.3, hidden_target="base_anchor", hidden_norm="none", monitor="diagnostic"
    )
    cfg = replace(
        cfg, hid=replace(cfg.hid, predictor_lr_mult=2.0, skip_first_token=False)
    )
    assert cfg.lambda_hid == 0.3 and cfg.hidden_target == "base_anchor"
    assert cfg.hidden_norm == "none" and cfg.predictor_lr_mult == 2.0
    assert cfg.skip_first_token is False and cfg.predictor == cfg.hid.predictor
    assert cfg.stability is cfg.monitor and cfg.stability.enabled


def test_from_dict_builds_the_nested_sections() -> None:
    cfg = A2Config.from_dict(
        {
            "hid": {
                "lambda_hid": 0.2,
                "hidden_target": "base_anchor",
                "hidden_norm": "none",
                "skip_first_token": False,
                "predictor_lr_mult": 2.0,
                "predictor": {"ff_dim": 8, "detach_input": False, "layers": [5]},
            },
            "monitor": {"mode": "diagnostic", "layers": [6]},
        }
    )
    assert cfg.hid.weight == 0.2 and cfg.hid.hidden_norm == "none"
    assert cfg.hid.hidden_target == "base_anchor" and cfg.hid.predictor_lr_mult == 2.0
    assert not cfg.hid.skip_first_token
    assert cfg.hid.predictor == PredictorConfig(
        ff_dim=8, detach_input=False, layers=[5]
    )
    assert cfg.monitor == MonitorConfig(mode="diagnostic", layers=[6])
    # Null sections (``hid:`` with nothing under it in YAML) mean the defaults; a bare
    # ``mode: off`` is read by YAML as False.
    assert A2Config.from_dict({"hid": None, "monitor": None}) == A2Config()
    assert A2Config.from_dict({"monitor": {"mode": False}}).monitor.mode == "off"


def test_lambda_hid_is_an_alias_of_weight() -> None:
    assert HiddenConfig(lambda_hid=0.4).weight == 0.4
    assert HiddenConfig(lambda_hid=0.4).lambda_hid is None  # folded into ``weight``
    assert HiddenConfig.from_dict({"lambda_hid": 0.4}).weight == 0.4
    assert HiddenConfig.from_dict({"weight": 0.4, "lambda_hid": 0.4}).weight == 0.4
    with pytest.raises(ValueError, match="must match"):
        HiddenConfig.from_dict({"weight": 0.1, "lambda_hid": 0.4})
    # ``replace`` works with either spelling and asdict round-trips.
    base = HiddenConfig(weight=0.1)
    assert replace(base, lambda_hid=0.7).weight == 0.7
    assert replace(replace(base, lambda_hid=0.7), weight=0.0).weight == 0.0
    assert not replace(HiddenConfig(lambda_hid=0.4), weight=0.0).enabled
    assert HiddenConfig.from_dict(asdict(base)) == base
    with pytest.raises(ValueError):
        HiddenConfig(lambda_hid=-1.0)


def test_the_earlier_flat_yaml_spelling_is_still_read() -> None:
    flat = A2Config.from_dict(
        {
            "lambda_hid": 0.2,
            "hidden_target": "base_anchor",
            "predictor_lr_mult": 3.0,
            "hidden_norm": "none",
            "skip_first_token": False,
            "predictor": {"ff_dim": 8},
            "stability": {"mode": "diagnostic"},
        }
    )
    nested = A2Config.from_dict(
        {
            "hid": {
                "lambda_hid": 0.2,
                "hidden_target": "base_anchor",
                "predictor_lr_mult": 3.0,
                "hidden_norm": "none",
                "skip_first_token": False,
                "predictor": {"ff_dim": 8},
            },
            "monitor": {"mode": "diagnostic"},
        }
    )
    assert flat == nested
    with pytest.raises(ValueError, match="both"):
        A2Config.from_dict({"lambda_hid": 0.1, "hid": {"lambda_hid": 0.2}})
    with pytest.raises(ValueError, match="either"):
        A2Config.from_dict({"stability": {}, "monitor": {}})


@pytest.mark.parametrize(
    "data",
    [
        {"hid": {"wieght": 0.1}},
        {"hid": {"predictor": {"ffdim": 4}}},
        {"monitor": {"modee": "off"}},
        {"lambda_hidd": 0.1},
    ],
)
def test_unknown_keys_are_rejected(data: dict) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        A2Config.from_dict(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": -0.1},
        {"weight": "0.1"},
        {"weight": True},
        {"hidden_norm": "std"},
        {"hidden_target": "teacher"},
        {"skip_first_token": 1},
        {"predictor_lr_mult": 0.0},
        {"predictor_lr_mult": True},
    ],
)
def test_invalid_hidden_values_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HiddenConfig(**kwargs)


def test_nested_validation_still_applies() -> None:
    with pytest.raises(ValueError, match="ff_dim"):
        A2Config.from_dict({"hid": {"predictor": {"ff_dim": 0}}})
    with pytest.raises(ValueError, match="mode"):
        A2Config.from_dict({"monitor": {"mode": "full"}})


def test_asdict_round_trip() -> None:
    cfg = make_cfg(hid=0.3, detach=False, monitor="diagnostic", cpc=0.1, stab=0.2)
    assert A2Config.from_dict(asdict(cfg)) == cfg


# --- ablation arms ----------------------------------------------------------------------------


def test_the_hidden_arms_exist_and_switch_only_hid_on() -> None:
    assert {"hid_detached", "hid_coupled"} <= set(ARMS)
    cfg = make_cfg(monitor="diagnostic")
    for arm, detach in (("hid_detached", True), ("hid_coupled", False)):
        out = apply_arm(cfg, arm)
        assert out.hid.enabled and not out.cpc.enabled and not out.stab.enabled
        assert out.hid.predictor.detach_input is detach
        assert out.hid.weight == ARM_DEFAULT_WEIGHTS["hid"]
        assert out.monitor == cfg.monitor
        assert not out.needs_base_states


def test_arms_without_hid_switch_it_off_and_full_has_everything() -> None:
    cfg = make_cfg(hid=0.4)
    for arm in ("a1_mode", "stab_anch", "stab_hep", "cpc_base", "cpc_multi"):
        assert not apply_arm(cfg, arm).uses_hidden_pred
    full = apply_arm(cfg, "full")
    assert full.hid.enabled and full.cpc.enabled and full.stab.enabled
    assert full.hid.weight == 0.4 and full.hid.coupled
    assert not apply_arm(cfg, "a1_mode").aux_enabled


def test_arms_keep_the_predictor_settings() -> None:
    cfg = replace(
        make_cfg(hid=0.4),
        hid=HiddenConfig(
            weight=0.4,
            hidden_target="base_anchor",
            hidden_norm="none",
            skip_first_token=False,
            predictor_lr_mult=3.0,
            predictor=PredictorConfig(ff_dim=8, predict_delta=True, layers=[5]),
        ),
    )
    out = apply_arm(cfg, "hid_coupled").hid
    assert (out.hidden_norm, out.skip_first_token) == ("none", False)
    assert (out.hidden_target, out.predictor_lr_mult) == ("base_anchor", 3.0)
    assert (out.predictor.ff_dim, out.predictor.predict_delta) == (8, True)
    assert out.predictor.layers == [5]
    zero = replace(cfg, hid=replace(cfg.hid, weight=0.0))
    assert (
        apply_arm(zero, "hid_coupled").hid.weight > 0
    )  # falls back to the arm default


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


# --- parameters -------------------------------------------------------------------------------


# --- the local predictor in isolation (from the predictor's own suite) ------------------------


def test_bank_sites_default_to_adapted_layers(substrate, adapter) -> None:
    """Bank layers match the adapter; init_params produces one key per site."""
    bank = _bank(substrate, adapter)
    assert bank.layers == adapter.layers
    assert set(bank.init_params()) == {predictor_key(i) for i in adapter.layers}


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


def test_bank_matches_ln_gelu_mlp_formula(substrate, adapter) -> None:
    """bank.predict reproduces the manual LayerNorm → GELU → linear formula."""
    bank = _bank(substrate, adapter)
    params = bank.init_params()
    layer = LAYERS[0]
    pkey = predictor_key(layer)
    # Non-trivial biases/gain so the check is not satisfied by zeros/ones.
    q = {
        k: v + 0.1 * jax.random.normal(jax.random.PRNGKey(i), v.shape)
        for i, (k, v) in enumerate(params[pkey].items())
    }
    params = {**params, pkey: q}
    u = _u(substrate.architecture.hidden_size)
    mu, var = u.mean(-1, keepdims=True), u.var(-1, keepdims=True)
    xn = (u - mu) / jnp.sqrt(var + 1e-5) * q["pred_hidden.w.ln_gamma"] + q[
        "pred_hidden.b.ln_beta"
    ]
    hid = jax.nn.gelu(xn @ q["pred_hidden.w.W_ff1"] + q["pred_hidden.b.b_ff1"])
    expected = hid @ q["pred_output.w.W"] + q["pred_output.b.b"]
    got = bank.predict(params, layer, u)
    assert got.shape == (BATCH, SEQ, substrate.architecture.hidden_size)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_bank_predict_delta_adds_the_skip(substrate, adapter) -> None:
    """predict_delta=True ⇒ output = input + predict(input)."""
    layer = LAYERS[0]
    plain = _bank(substrate, adapter)
    delta = _bank(substrate, adapter, predict_delta=True)
    params = plain.init_params()
    u = _u(substrate.architecture.hidden_size)
    assert jnp.allclose(
        delta.predict(params, layer, u),
        u + plain.predict(params, layer, u),
        atol=1e-6,
    )


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


def test_bank_rejects_wrong_input_shape_and_missing_layer(substrate, adapter) -> None:
    """Wrong sequence length → ValueError; un-adapted layer → KeyError."""
    bank = _bank(substrate, adapter)
    params = bank.init_params()
    d = substrate.architecture.hidden_size
    layer = LAYERS[0]
    with pytest.raises(ValueError, match="predictor input"):
        bank.predict(params, layer, jnp.zeros((BATCH, SEQ + 1, d)))
    with pytest.raises(KeyError):
        bank.predict(params, 2, jnp.zeros((BATCH, SEQ, d)))


def test_bank_last_block_cannot_have_a_predictor(substrate) -> None:
    """The last transformer block has no following block to predict."""
    last = substrate.architecture.num_layers - 1
    tmp_adapter = ResidualAdapter(
        ResidualConfig(rank=4, layers=[LAYERS[0], last]), substrate.architecture
    )
    with pytest.raises(ValueError, match="following block"):
        LocalPredictorBank(
            PredictorConfig(layers=[last]),
            substrate.architecture,
            SEQ,
            tmp_adapter.layers,
        )


def test_bank_split_and_merge_roundtrip(substrate, adapter) -> None:
    """merge_params → split_params recovers the original adapter & predictor dicts."""
    bank = _bank(substrate, adapter)
    a, p = adapter.init_params(), bank.init_params()
    merged = merge_params(a, p)
    a2, p2 = split_params(merged)
    assert set(a2) == set(a) and set(p2) == set(p)
    with pytest.raises(ValueError, match="overlap"):
        merge_params(a, a)


def test_predictor_config_validation() -> None:
    """PredictorConfig rejects ff_dim=0 and unknown keys."""
    with pytest.raises(ValueError, match="ff_dim"):
        PredictorConfig(ff_dim=0)
    with pytest.raises(ValueError, match="Unknown PredictorConfig"):
        PredictorConfig.from_dict({"bogus": 1})


def test_predictors_are_added_per_site_only_when_enabled(substrate, adapter) -> None:
    off = init_a2_params(make_cfg(), adapter)
    assert set(off) == set(adapter.init_params())
    on = init_a2_params(make_cfg(hid=0.1), adapter)
    assert set(on) - set(off) == {predictor_key(layer) for layer in LAYERS}
    assert set(predictor_params(on)) == {predictor_key(layer) for layer in LAYERS}
    assert set(adapter_params(on)) == {layer_key(layer) for layer in LAYERS}


def test_predictor_sites_default_to_the_adapted_layers_and_can_be_restricted(
    substrate, adapter
) -> None:
    cfg = replace(
        make_cfg(hid=0.1),
        hid=HiddenConfig(weight=0.1, predictor=PredictorConfig(ff_dim=FF, layers=[5])),
    )
    assert set(predictor_params(init_a2_params(cfg, adapter))) == {predictor_key(5)}


def test_the_last_block_gets_no_predictor_by_default(substrate) -> None:
    last = substrate.architecture.num_layers - 1
    adapter = ResidualAdapter(
        ResidualConfig(rank=4, layers=[5, last]), substrate.architecture
    )
    cfg = make_cfg(hid=0.1)
    assert set(predictor_params(init_a2_params(cfg, adapter))) == {predictor_key(5)}


def test_predictor_sites_must_be_adapted_layers(substrate, adapter) -> None:
    cfg = replace(
        make_cfg(hid=0.1),
        hid=HiddenConfig(weight=0.1, predictor=PredictorConfig(ff_dim=FF, layers=[2])),
    )
    with pytest.raises(ValueError, match="adapted layers"):
        init_a2_params(cfg, adapter)


def test_predictors_are_independent_of_the_sequence_length(substrate, adapter) -> None:
    params = init_a2_params(make_cfg(hid=0.1), adapter)
    for key, sub in predictor_params(params).items():
        assert all(SEQ not in v.shape for v in sub.values()), key


def test_param_counts_include_the_predictors(substrate, adapter) -> None:
    params = init_a2_params(make_cfg(hid=0.1), adapter)
    counts = param_counts(params)
    d = substrate.architecture.hidden_size
    per_site = d * FF + FF + 2 * d + FF * d + d
    assert counts["pred"] == len(LAYERS) * per_site
    assert counts["total"] == sum(counts[g] for g in ("adapter", "cpc", "hep", "pred"))
    assert param_counts(adapter.init_params())["pred"] == 0


def test_predictor_seed_controls_the_predictors_and_not_the_adapter(
    substrate, adapter
) -> None:
    def cfg(seed):
        return replace(
            make_cfg(hid=0.1),
            hid=HiddenConfig(
                weight=0.1, predictor=PredictorConfig(ff_dim=FF, seed=seed)
            ),
        )

    a, b, c = (init_a2_params(cfg(s), adapter) for s in (0, 0, 1))
    name = "pred_hidden.w.W_ff1"
    np.testing.assert_array_equal(a[predictor_key(5)][name], b[predictor_key(5)][name])
    assert not np.allclose(a[predictor_key(5)][name], c[predictor_key(5)][name])
    np.testing.assert_array_equal(a[layer_key(5)]["A"], c[layer_key(5)]["A"])


def test_the_objective_seed_does_not_move_the_predictors(substrate, adapter) -> None:
    name = "pred_hidden.w.W_ff1"
    a = init_a2_params(make_cfg(hid=0.1, seed=0), adapter)
    b = init_a2_params(make_cfg(hid=0.1, seed=3), adapter)
    np.testing.assert_array_equal(a[predictor_key(5)][name], b[predictor_key(5)][name])


def test_init_does_not_depend_on_the_sequence_length_argument(
    substrate, adapter
) -> None:
    cfg = make_cfg(hid=0.1)
    a = init_a2_params(cfg, adapter)
    b = init_a2_params(cfg, adapter, seq_len=SEQ)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        np.testing.assert_array_equal(x, y)


def test_the_bank_is_built_once_per_configuration(substrate, adapter) -> None:
    a = predictor_bank(make_cfg(hid=1.0), adapter, SEQ)
    b = predictor_bank(make_cfg(hid=2.0), adapter, SEQ)
    c = predictor_bank(
        replace(
            make_cfg(hid=1.0),
            hid=HiddenConfig(weight=1.0, predictor=PredictorConfig(ff_dim=FF + 1)),
        ),
        adapter,
        SEQ,
    )
    d = predictor_bank(make_cfg(hid=1.0), adapter, SEQ + 1)
    assert a is b and a is not c and a is not d
    assert a.seq_len == SEQ and tuple(a.layers) == tuple(LAYERS)


def test_tree_is_flat_two_level_and_checkpointable(
    substrate, adapter, tmp_path
) -> None:
    params = init_a2_params(make_cfg(hid=0.1, cpc=0.1, stab=0.1), adapter)
    assert all(isinstance(v, dict) for v in params.values())
    assert all(isinstance(x, jax.Array) for v in params.values() for x in v.values())
    with RunWriter(tmp_path, "run") as writer:
        path = writer.save_params("t", params)
    saved = load_params(path)
    assert set(saved) == set(params)
    for k in params:
        assert set(saved[k]) == set(params[k])
        for name in params[k]:
            np.testing.assert_array_equal(saved[k][name], np.asarray(params[k][name]))


# --- the term ---------------------------------------------------------------------------------


def test_hid_term_is_the_mean_of_the_site_losses(substrate, adapter) -> None:
    cfg = make_cfg(hid=0.1, detach=False)
    params = with_random_predictors(init_a2_params(cfg, adapter))
    bank = predictor_bank(cfg, adapter, SEQ)
    rng = np.random.default_rng(3)
    d = substrate.architecture.hidden_size
    corrected = {
        i: jnp.asarray(rng.normal(size=(BATCH, SEQ, d)), jnp.float32) for i in LAYERS
    }
    targets = {
        i: jnp.asarray(rng.normal(size=(BATCH, SEQ, d)), jnp.float32) for i in LAYERS
    }
    loss, metrics = hid_term(
        params, config=cfg, bank=bank, corrected=corrected, targets=targets
    )
    parts = [
        hidden_prediction_loss(
            bank.predict(params, i, corrected[i]), targets[i], corrected[i], cfg
        )
        for i in LAYERS
    ]
    assert float(loss) == pytest.approx(
        np.mean([float(p["loss"]) for p in parts]), rel=1e-5
    )
    for name in ("mse", "rel", "copy_rel"):
        assert float(metrics[f"hid_{name}"]) == pytest.approx(
            np.mean([float(p[name]) for p in parts]), rel=1e-5
        )
    assert set(metrics) == {"hid_mse", "hid_rel", "hid_copy_rel"}


def test_hid_term_needs_every_site(substrate, adapter) -> None:
    cfg = make_cfg(hid=0.1)
    params = init_a2_params(cfg, adapter)
    bank = predictor_bank(cfg, adapter, SEQ)
    d = substrate.architecture.hidden_size
    z = jnp.zeros((BATCH, SEQ, d))
    with pytest.raises(KeyError, match="corrected"):
        hid_term(params, config=cfg, bank=bank, corrected={5: z}, targets={5: z, 6: z})
    with pytest.raises(KeyError, match="next-block"):
        hid_term(params, config=cfg, bank=bank, corrected={5: z, 6: z}, targets={5: z})


# --- objective: parity and contract ------------------------------------------------------------


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


def test_hid_terms_are_returned_and_finite(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(hid=0.1)
    params = with_random_b(init_a2_params(cfg, adapter))
    total, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    assert {"total", "task", "wd", "hid", "hid_mse", "hid_rel", "hid_copy_rel"} <= set(
        terms
    )
    assert all(bool(jnp.isfinite(v)) for v in terms.values())
    assert total.shape == ()


def test_total_is_the_weighted_sum(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(hid=0.3, wd=0.2, kl=0.1, cpc=0.2, stab=0.1)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    total, t = run_objective(substrate, adapter, cfg, params, input_ids)
    expected = (
        t["task"]
        + 0.1 * t["kl"]
        + 0.2 * t["wd"]
        + 0.2 * t["cpc"]
        + 0.1 * t["stab"]
        + 0.3 * t["hid"]
    )
    assert float(total) == pytest.approx(float(expected), rel=1e-5)


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


def test_hid_does_not_need_the_base_model_states() -> None:
    # The target is the next frozen block of the adapted pass, so no extra base forward.
    assert not make_cfg(hid=0.1).needs_base_states
    assert make_cfg(hid=0.1, cpc=0.1).needs_base_states  # cpc.positive == "base"


def test_copy_baseline_is_the_untrained_hidden_error_scale(
    substrate, adapter, input_ids
) -> None:
    cfg = make_cfg(hid=0.1)
    params = init_a2_params(cfg, adapter)
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    # An untrained predictor is a poor guess; copying h~ is the trivial baseline.
    assert float(terms["hid_rel"]) > 0.0 and float(terms["hid_copy_rel"]) > 0.0


def test_precomputed_base_logits_do_not_change_hid(
    substrate, adapter, input_ids
) -> None:
    from stages.stage_base import base_logits

    cfg = make_cfg(hid=0.1, kl=0.1)
    params = with_random_b(init_a2_params(cfg, adapter))
    a = run_objective(substrate, adapter, cfg, params, input_ids)[1]
    b = run_objective(
        substrate,
        adapter,
        cfg,
        params,
        input_ids,
        base=base_logits(substrate, input_ids),
    )[1]
    assert float(a["hid"]) == pytest.approx(float(b["hid"]), rel=1e-6)
    assert float(a["total"]) == pytest.approx(float(b["total"]), rel=1e-5)


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


def test_jit_matches_eager(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(hid=0.2, detach=False, monitor="diagnostic")
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    eager = run_objective(substrate, adapter, cfg, params, input_ids)
    jitted = jax.jit(lambda p: run_objective(substrate, adapter, cfg, p, input_ids))(
        params
    )
    for k in eager[1]:
        assert float(jitted[1][k]) == pytest.approx(
            float(eager[1][k]), rel=1e-4, abs=1e-6
        ), k
    g_eager = grads_of(substrate, adapter, cfg, params, input_ids)
    g_jit = jax.jit(lambda p: grads_of(substrate, adapter, cfg, p, input_ids))(params)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5),
        g_eager,
        g_jit,
    )


# --- gradient routing -------------------------------------------------------------------------


def hid_only_grad(substrate, adapter, input_ids, detach: bool):
    cfg = make_cfg(hid=1.0, detach=detach)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))

    def loss(p):
        return run_objective(substrate, adapter, cfg, p, input_ids)[1]["hid"]

    return jax.grad(loss)(params)


def test_detached_hid_trains_only_the_predictors(substrate, adapter, input_ids) -> None:
    g = hid_only_grad(substrate, adapter, input_ids, detach=True)
    assert max_abs(predictor_params(g)) > 0.0
    assert max_abs(adapter_params(g)) == 0.0


def test_coupled_hid_trains_the_predictors_and_the_adapter(
    substrate, adapter, input_ids
) -> None:
    g = hid_only_grad(substrate, adapter, input_ids, detach=False)
    assert max_abs(predictor_params(g)) > 0.0
    assert max_abs(adapter_params(g)) > 0.0


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


def test_hid_enters_the_gradient_linearly_with_its_weight(gpt2, input_ids) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    params = with_random_predictors(
        with_random_b(init_a2_params(make_cfg(hid=1.0, detach=False), adapter))
    )
    g1 = grads_of(gpt2, adapter, make_cfg(hid=1.0, detach=False), params, input_ids)
    g3 = grads_of(gpt2, adapter, make_cfg(hid=3.0, detach=False), params, input_ids)
    g0 = grads_of(gpt2, adapter, make_cfg(hid=1e-12, detach=False), params, input_ids)
    jax.tree_util.tree_map(
        lambda a, b, c: np.testing.assert_allclose(
            np.asarray(c) - np.asarray(b),
            3 * (np.asarray(a) - np.asarray(b)),
            rtol=2e-3,
            atol=1e-5,
        ),
        g1,
        g0,
        g3,
    )


def test_task_only_gradient_never_reaches_the_predictors(gpt2, input_ids) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=0.1)
    params = with_random_b(init_a2_params(cfg, adapter))
    g = jax.grad(lambda p: run_objective(gpt2, adapter, cfg, p, input_ids)[1]["task"])(
        params
    )
    assert max_abs(predictor_params(g)) == 0.0 and max_abs(adapter_params(g)) > 0.0


def test_weight_decay_covers_the_adapter_and_not_the_predictors(
    gpt2, input_ids
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    params = with_random_predictors(init_a2_params(make_cfg(hid=0.1), adapter))
    _, t = run_objective(gpt2, adapter, make_cfg(hid=0.1, wd=1.0), params, input_ids)
    assert float(t["wd"]) == pytest.approx(
        float(ResidualAdapter.l2_norm_sq(adapter_params(params))), rel=1e-6
    )


# --- the monitor inside the objective ---------------------------------------------------------


def test_monitor_metrics_are_logged_only_when_enabled(
    substrate, adapter, input_ids
) -> None:
    params = with_random_b(init_a2_params(make_cfg(), adapter))
    keys = {
        "res_smooth",
        "input_smooth",
        "res_vs_input",
        *("step_loss", "step_noncontract", "step_last_norm"),
    }
    _, off = run_objective(substrate, adapter, make_cfg(), params, input_ids)
    assert not keys & set(off)
    _, on = run_objective(
        substrate, adapter, make_cfg(monitor="diagnostic"), params, input_ids
    )
    assert keys <= set(on) and {f"res_smooth_l{i}" for i in LAYERS} <= set(on)
    assert all(bool(jnp.isfinite(on[k])) for k in keys)
    assert all(
        float(on[k]) == 0.0 for k in ("step_loss", "step_noncontract", "step_last_norm")
    )
    assert float(on["res_smooth"]) > 0.0 and float(on["input_smooth"]) > 0.0


def test_monitor_matches_the_manual_residual_smoothness(
    substrate, adapter, input_ids
) -> None:
    from residual.a2_losses import residual_smoothness

    params = with_random_b(init_a2_params(make_cfg(), adapter))
    ap = adapter_params(params)
    run = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
    )
    rows = []
    for i in LAYERS:
        h = run.hidden_state(i)
        rows.append(residual_smoothness(adapter.apply(ap, h, i) - h, h))
    _, on = run_objective(
        substrate, adapter, make_cfg(monitor="diagnostic"), params, input_ids
    )
    for key in ("res_smooth", "input_smooth"):
        want = np.mean([float(r[key]) for r in rows])
        assert float(on[key]) == pytest.approx(want, rel=1e-4)


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


def test_monitor_alone_equals_a1(substrate, adapter, input_ids) -> None:
    cfg = make_cfg(monitor="diagnostic", kl=0.1, wd=0.1)
    assert not cfg.aux_enabled
    params = with_random_b(init_a2_params(cfg, adapter))
    total, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    a1_total, _ = a1_objective(
        adapter_params(params),
        substrate=substrate,
        adapter=adapter,
        config=A1Config(lambda_kl=0.1, lambda_wd=0.1),
        input_ids=input_ids,
    )
    assert float(total) == pytest.approx(float(a1_total), rel=1e-6)
    assert "res_smooth" in terms


def test_monitor_sites_must_be_adapted(substrate, adapter, input_ids) -> None:
    cfg = replace(make_cfg(), monitor=MonitorConfig(mode="diagnostic", layers=[2]))
    with pytest.raises(ValueError, match="adapted sites"):
        run_objective(substrate, adapter, cfg, init_a2_params(cfg, adapter), input_ids)


# --- training ---------------------------------------------------------------------------------


@pytest.mark.parametrize("detach", [True, False])
def test_optimising_lowers_the_hidden_prediction_error(gpt2, input_ids, detach) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=1.0, detach=detach)
    params = init_a2_params(cfg, adapter)
    start = params
    opt = optax.adam(1e-2)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        (_, terms), g = jax.value_and_grad(
            lambda q: run_objective(gpt2, adapter, cfg, q, input_ids), has_aux=True
        )(p)
        updates, s = opt.update(g, s, p)
        return optax.apply_updates(p, updates), s, terms

    first = None
    for _ in range(40):
        params, state, terms = step(params, state)
        first = terms if first is None else first
    assert float(terms["hid"]) < 0.8 * float(first["hid"])
    moved = {
        g: max_abs(jax.tree_util.tree_map(lambda a, b: a - b, fn(params), fn(start)))
        for g, fn in (("pred", predictor_params), ("adapter", adapter_params))
    }
    assert moved["pred"] > 0.0
    # (The adapter moves in both arms because of the task loss; who gets the hid gradient is
    # tested above.)
    assert moved["adapter"] > 0.0
    assert gpt2.params_unchanged()


def test_every_arm_runs_through_the_objective(gpt2, input_ids) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    for arm in ARMS:
        cfg = apply_arm(make_cfg(monitor="diagnostic"), arm)
        params = with_random_b(init_a2_params(cfg, adapter))
        total, terms = run_objective(gpt2, adapter, cfg, params, input_ids)
        assert bool(jnp.isfinite(total)), arm
        assert ("hid" in terms) == cfg.uses_hidden_pred, arm


# --- audit ------------------------------------------------------------------------------------


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


# --- run_sequential ---------------------------------------------------------------------------


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


def test_evaluation_ignores_the_predictors(gpt2) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    params = with_random_predictors(
        with_random_b(init_a2_params(make_cfg(hid=0.2), adapter))
    )
    eval_set = _domain("a", 5).eval_set
    full = evaluate(gpt2, adapter, params, eval_set)
    only = evaluate(gpt2, adapter, adapter_params(params), eval_set)
    assert full == pytest.approx(only)