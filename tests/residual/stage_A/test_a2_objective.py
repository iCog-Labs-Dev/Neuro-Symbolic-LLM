"""Essential A2 contracts: detach routing, live task terms, and runner wiring."""

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from losses.a2_loss import drift_metrics
from residual import ResidualAdapter, ResidualConfig, layer_key
from residual.cpc_heads import q_key
from residual.local_predictor import PredictorConfig
from stages.results import load_params
from stages.stage_A.a2 import (
    A2Config,
    CPCConfig,
    HiddenConfig,
    StabLossConfig,
    a2_objective,
    adapter_params,
    init_a2_params,
)
from stages.stage_base import Domain, evaluate, load_configs, make_eval_set
from tests.frozenllm.conftest import BATCH, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]
SEQ = 12


def _domain(name, token, num_blocks=8):
    blocks = np.random.default_rng(token).integers(
        0, 64, (num_blocks, SEQ), dtype=np.int32
    )
    return Domain(name, blocks, make_eval_set(blocks, BATCH, 1), 1)


def tree_norm(tree):
    return sum(float(jnp.sum(x**2)) for x in jax.tree_util.tree_leaves(tree))


@pytest.fixture(scope="module")
def setup():
    substrate = make_substrate("gpt2")[1]
    adapter = ResidualAdapter(
        ResidualConfig(rank=2, layers=[5]), substrate.architecture
    )
    config = A2Config(
        lambda_kl=0.1,
        cpc=CPCConfig(weight=0.1, dim=4, max_anchors_per_seq=3),
        hid=HiddenConfig(weight=0.1, predictor=PredictorConfig(ff_dim=8)),
        stab=StabLossConfig(weight=0.1, w_anch=1.0, w_smooth=0.1),
    )
    ids = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    params = init_a2_params(config, adapter, seq_len=4)
    params["layer_5"]["B"] = 0.05 * jax.random.normal(
        jax.random.PRNGKey(7), params["layer_5"]["B"].shape
    )
    for index, block in enumerate(k for k in params if k.startswith("pred_")):
        for offset, name in enumerate(params[block]):
            params[block][name] = 0.1 * jax.random.normal(
                jax.random.PRNGKey(100 + 20 * index + offset),
                params[block][name].shape,
            )
    return substrate, adapter, config, ids, params


def test_detach_flags_block_column_gradients_but_heads_and_task_terms_stay_live(setup):
    """Each auxiliary loss is checked independently, with a nonzero adapter probe."""
    substrate, adapter, config, ids, params = setup

    def terms(p, cfg=config):
        return a2_objective(
            p, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids
        )[1]

    for term, prefix in (("hid", "pred_"), ("cpc", "cpc")):
        grad = jax.grad(lambda p, term=term: terms(p)[term])(params)
        assert tree_norm(adapter_params(grad)) == 0.0
        assert tree_norm({k: v for k, v in grad.items() if k.startswith(prefix)}) > 0
        coupled = replace(config, hid_detach_input=False, cpc_detach_query=False)
        grad = jax.grad(lambda p, term=term, cfg=coupled: terms(p, cfg)[term])(params)
        assert tree_norm(adapter_params(grad)) > 0

    for term in ("task", "kl", "stab_anch", "stab_smooth"):
        grad = jax.grad(lambda p, term=term: terms(p)[term])(params)
        assert tree_norm(adapter_params(grad)) > 0
    assert substrate.params_unchanged()


def test_targets_are_detached_and_drift_is_measurement_only(setup):
    from losses.a2_loss import (
        anchor_loss,
        contrastive_future_loss,
        hidden_prediction_loss,
    )
    from residual.cpc_heads import a_key, init_cpc_heads, q_key

    substrate, _, config, _, _ = setup
    source = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8))
    target = source + 0.5
    heads = init_cpc_heads(0, [5], 8, 4)
    for loss in (
        lambda y: hidden_prediction_loss(source, y, source, config)["loss"],
        lambda y: contrastive_future_loss(
            heads[q_key(5)], heads[a_key(5, 1)], source, y
        )[0],
        lambda y: anchor_loss(source, y)[0],
        lambda y: drift_metrics({5: y}, {5: source})["drift_mean"],
    ):
        np.testing.assert_array_equal(jax.grad(loss)(target), jnp.zeros_like(target))
    from stages.stage_base import model_loss_terms

    labels = jnp.zeros((2, 4), dtype=jnp.int32)
    base_grad = jax.grad(
        lambda base: model_loss_terms(substrate, source, labels, 0.1, base)[0]
    )(target)
    np.testing.assert_array_equal(base_grad, jnp.zeros_like(target))


def test_a2_schema_defaults_and_settling_separation():
    import ast

    root = Path(__file__).resolve().parents[3]
    cfg = load_configs(
        root / "configs" / "stage_A" / "a2_pc_inspired.yaml", A2Config
    ).objective
    assert cfg.hid_detach_input is True and cfg.cpc_detach_query is True
    assert A2Config.from_dict(asdict(cfg)) == cfg
    for name in ("w_step", "w_hep", "hep"):
        with pytest.raises(ValueError, match="Unknown"):
            A2Config.from_dict({"stab": {name: 1}})
    for name in ("hid_detach_input", "cpc_detach_query"):
        with pytest.raises(ValueError, match="boolean"):
            A2Config.from_dict({name: 1})
    for path in (root / "stages" / "stage_A").glob("a2*.py"):
        imports = [
            n.module
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.ImportFrom)
        ]
        assert "losses.stability_loss" not in imports


def test_runner_detach_overrides_and_predictor_multiplier(runner):
    cfg = load_configs(REPO_ROOT / "configs/stage_A/a2_pc_inspired.yaml", A2Config)
    args = runner.parse_args(
        [
            "--no-hid-detach-input",
            "--no-cpc-detach-query",
            "--predictor-lr-mult",
            "2",
        ]
    )
    out = runner.apply_overrides(cfg, args).objective
    assert not out.hid_detach_input and not out.cpc_detach_query
    assert out.hid_coupled and not out.predictor.detach_input
    assert out.predictor_lr_mult == 2.0


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


def test_runner_main_end_to_end(
    runner: Any,
    setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    gpt2 = setup[0]
    import stages.stage_base as stage_base

    # No dataset download: the loader and the model constructor are stubbed.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "adapter: {rank: 4, layers: [5, 6]}\n"
        "objective:\n  lambda_kl: 0.1\n"
        "  cpc: {weight: 0.2, dim: 8}\n"
        "  stab: {weight: 0.2}\n"
        f"training: {{model: gpt2, seq_len: {SEQ}, batch_size: {BATCH}, epochs: 1, "
        "learning_rate: 1.0e-2, eval_batches: 1, log_every: 1}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    blocks = _domain("a", 5, num_blocks=4).train_blocks
    monkeypatch.setattr(runner, "FrozenSubstrate", lambda _name: gpt2)
    monkeypatch.setattr(stage_base, "load_token_blocks", lambda *a, **k: blocks)

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
    assert groups["cpc"] > 0 and "hep" not in groups
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
